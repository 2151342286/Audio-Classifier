# ast_model_adapted.py
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import os
import timm
from timm.models.layers import to_2tuple, trunc_normal_
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR
import torch.optim as optim
import time

# 将频谱图特征提取的类
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        # img_size：输入图像（或频谱图）的尺寸
        # patch_size：小块的尺寸（论文中为 16×16）
        # in_chans：输入通道数（图像为 3，音频频谱图为 1）
        # embed_dim：嵌入维度（论文中为 768）
        img_size = to_2tuple(img_size)  # 将输入转换为2元组。若输入是整数x，返回(x, x)；
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) # 计算块数
        self.img_size = img_size # 图片尺寸
        self.patch_size = patch_size # 块尺寸
        self.num_patches = num_patches # 块数量

        # 卷积层，起到线性投影的作用
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x.shape = (B, C, H, W)
        # 其中 B=批大小, C=通道数, H=高度, W=宽度

        # 卷积操作：将输入的x转化为B个768维的特征，其中，一个特征是原图经16*16卷积核以16步长卷积结果
        x = self.proj(x) # (B, embed_dim, H', W')

        # 将二维的空间网格展平为一维序列
        x = x.flatten(2) # (B, embed_dim, H', W') → (B, embed_dim, H'×W')

        # 将 (B, embed_dim, num_patches) → (B, num_patches, embed_dim)
        # 这是为了适应 Transformer 的输入格式：
        # - batch_size × sequence_length × embedding_dim
        x = x.transpose(1, 2)

        return x

class ASTModel(nn.Module):
    """
    AST模型。
    :param num_classes: 标签维度，即总类别数。ESC-50为50。
    :param fstride: 频谱图在频率维度上分割的步长。对于16*16块，fstride=16表示无重叠，fstride=10表示重叠6。
    :param tstride: 频谱图在时间维度上分割的步长。对于16*16块，tstride=16表示无重叠，tstride=10表示重叠6。
    :param input_fdim: 输入频谱图的频率bin数量。
    :param input_tdim: 输入频谱图的时间帧数量。
    :param imagenet_pretrain: 是否使用ImageNet预训练模型。
    :param model_size: AST模型大小，只有[base384]。
    """
    def __init__(self, num_classes=50, input_tdim=216, input_fdim=128, 
                 fstride=10, tstride=10, imagenet_pretrain=True, 
                 model_size='base384', verbose=True):
        super(ASTModel, self).__init__()
        
        if verbose:
            print('--------------- AST Model Summary ---------------')
            print(f'ImageNet pretraining: {imagenet_pretrain}')
            print(f'Input shape: {input_fdim}x{input_tdim}')
            print(f'Number of classes: {num_classes}')
        
        # 将timm库中Vision Transformer的PatchEmbed类替换为自定义的PatchEmbed类，以解除对输入形状的限制，使其能处理非方形输入和可变尺寸输入。
        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        
        # 创建DeiT模型
        if model_size == 'base384':
            self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=imagenet_pretrain)
        else:
            raise ValueError('仅支持base384模型')
        
        self.original_num_patches = self.v.patch_embed.num_patches
        self.original_hw = int(self.original_num_patches ** 0.5)
        self.original_embedding_dim = self.v.pos_embed.shape[2]
        
        # 分类头
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.original_embedding_dim),
            nn.Linear(self.original_embedding_dim, num_classes)
        )
        
        # 计算输出形状
        f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
        num_patches = f_dim * t_dim
        self.v.patch_embed.num_patches = num_patches
        
        if verbose:
            print(f'Frequency stride={fstride}, Time stride={tstride}')
            print(f'Number of patches={num_patches} ({f_dim}x{t_dim})')
        
        # 创建新的投影层（1通道输入）
        new_proj = nn.Conv2d(1, self.original_embedding_dim, 
                             kernel_size=(16, 16), stride=(fstride, tstride))
        
        if imagenet_pretrain:
            # 平均3通道权重为1通道
            new_proj.weight = nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
            new_proj.bias = self.v.patch_embed.proj.bias
        
        self.v.patch_embed.proj = new_proj
        
        # 位置嵌入适配
        if imagenet_pretrain:
            # 从DeiT模型中获取位置嵌入，跳过前两个token（分类token和蒸馏token），将其重塑为原始的2D形状（24*24）。
            new_pos_embed = self.v.pos_embed[:, 2:, :].detach()
            new_pos_embed = new_pos_embed.reshape(1, self.original_num_patches, self.original_embedding_dim)
            new_pos_embed = new_pos_embed.transpose(1, 2)
            new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, self.original_hw, self.original_hw)
            
            # 适配时间维度
            if t_dim <= self.original_hw:
                # 裁剪中间部分
                start_t = int(self.original_hw / 2) - int(t_dim / 2)
                new_pos_embed = new_pos_embed[:, :, :, start_t:start_t + t_dim]
            else:
                # 插值
                new_pos_embed = torch.nn.functional.interpolate(
                    new_pos_embed, size=(self.original_hw, t_dim), mode='bilinear')
            
            # 适配频率维度
            if f_dim <= self.original_hw:
                # 裁剪中间部分
                start_f = int(self.original_hw / 2) - int(f_dim / 2)
                new_pos_embed = new_pos_embed[:, :, start_f:start_f + f_dim, :]
            else:
                # 插值
                new_pos_embed = torch.nn.functional.interpolate(
                    new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
            
            # 展平
            new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches)
            new_pos_embed = new_pos_embed.transpose(1, 2)
            
            # 合并分类token的位置嵌入
            self.v.pos_embed = nn.Parameter(
                torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1)
            )
        
        # 打印参数量
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if verbose:
            print(f'Total parameters: {total_params:,}')
            print(f'Trainable parameters: {trainable_params:,}')
            print(f'Model size: {total_params * 4 / 1024 / 1024:.2f} MB')
            print('-----------------------------------------------')
    
    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        """计算卷积输出形状"""
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, 
                             kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim
    
    def forward(self, x):
        """
        x: [batch, 1, frequency_bins, time_frames] or [batch, time_frames, frequency_bins]
        """
        # 确保输入形状正确
        if x.dim() == 3:  # [batch, time, freq]
            x = x.unsqueeze(1)  # [batch, 1, time, freq]
            x = x.transpose(2, 3)  # [batch, 1, freq, time]
        elif x.dim() == 4 and x.shape[1] == 1:  # [batch, 1, freq, time]
            pass  # 已经是正确形状
        else:
            raise ValueError(f'输入形状错误: {x.shape}')
        
        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        
        for blk in self.v.blocks:
            x = blk(x)
        
        x = self.v.norm(x)
        # 使用两个分类token的平均
        x = (x[:, 0] + x[:, 1]) / 2
        
        x = self.mlp_head(x)
        return x

# 新增的SpecAugment类（放在train函数前面或单独文件）
class SpecAugment:
    """SpecAugment数据增强，在训练时应用"""
    def __init__(self, freq_mask_param=48, time_mask_param=192, num_freq_masks=1, num_time_masks=1):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        
    def __call__(self, x):
        # x: [batch, 1, freq, time]
        batch, channels, freq, time = x.shape
        
        # 频率遮蔽（ESC-50数据集较小，用温和的参数）
        for _ in range(self.num_freq_masks):
            f = min(self.freq_mask_param, freq // 4)  # 不超过1/4频率
            if f > 0:
                f0 = torch.randint(0, max(1, freq - f), (1,)).item()
                x[:, :, f0:f0+f, :] = 0
        
        # 时间遮蔽
        for _ in range(self.num_time_masks):
            t = min(self.time_mask_param, time // 4)  # 不超过1/4时间
            if t > 0:
                t0 = torch.randint(0, max(1, time - t), (1,)).item()
                x[:, :, :, t0:t0+t] = 0
            
        return x
    
# 训练函数（适配你的main.py）
def train(model, train_loader, val_loader, fold, epochs=20, lr=5e-4,patience=5):

    model_save_path = './model_param/ast/' + str(fold) + '.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running on {device}')

    # 初始化SpecAugment
    spec_augment = SpecAugment(
        freq_mask_param=24,  
        time_mask_param=48,
        num_freq_masks=1,
        num_time_masks=1
    )

    # 使用DataParallel
    if not isinstance(model, nn.DataParallel):
        model = nn.DataParallel(model)
    
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    

    trainables = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in trainables)
    
    print(f'Total parameters: {total_params / 1e6:.3f} million')
    print(f'Trainable parameters: {trainable_params / 1e6:.3f} million')
    
    optimizer = optim.Adam(trainables, lr=lr, weight_decay=5e-5, betas=(0.95, 0.999))
    
    # ESC-50的学习率调度：第5个epoch后每epoch衰减0.85
    scheduler = MultiStepLR(optimizer, milestones=list(range(3, 26)), gamma=0.85)
    
    # 训练历史
    history = {
        'train_loss': [], 'train_accuracy': [], 
        'val_loss': [], 'val_accuracy': [],
        'lr': [],'time':[]
    }
    
    # 早停和最佳模型追踪
    best_val_accuracy = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    early_stop = False
    back_best = False

    print(f'开始训练，共{epochs}个epoch')
    
    for epoch in range(epochs):

        if early_stop:
            print(f'早停触发，训练提前结束')
            break
        
        # 检查是否需要回滚 
        if back_best:
            # 无改进，回滚到最佳模型
            checkpoint = torch.load(model_save_path,weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  ↺ 回滚到epoch {best_epoch}的最佳模型状态")
            back_best = False
          
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        batch_count = 0
        
        start_time = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # 应用SpecAugment
            if model.training:
                inputs = spec_augment(inputs)
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 统计
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            batch_count += 1
            
            # 每50个batch打印一次
            if (batch_idx + 1) % 50 == 0:
                batch_acc = correct / total if total > 0 else 0
                batch_loss = epoch_loss / batch_count if batch_count > 0 else 0
                print(f'  Batch {batch_idx+1}/{len(train_loader)}, Loss: {batch_loss:.4f}, Acc: {batch_acc:.4f}')
        
        # 计算epoch指标
        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
        accuracy = correct / total if total > 0 else 0
        current_lr = optimizer.param_groups[0]['lr']
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_batch_count = 0

        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                
                val_outputs = model(val_inputs)
                val_batch_loss = criterion(val_outputs, val_targets)
                
                val_loss += val_batch_loss.item()
                _, val_predicted = torch.max(val_outputs, 1)
                val_total += val_targets.size(0)
                val_correct += (val_predicted == val_targets).sum().item()
                val_batch_count += 1
        
        val_accuracy = val_correct / val_total if val_total > 0 else 0
        avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else 0
        model.train()

        # 更新学习率
        scheduler.step()
        
        # 记录历史
        history['train_loss'].append(avg_loss)
        history['train_accuracy'].append(accuracy)
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['lr'].append(current_lr)
        
        # 1. 检测过拟合
        overfitting_warning = ""
        if epoch >= 2:
            train_acc_trend = history['train_accuracy'][-1] - history['train_accuracy'][-2]
            val_acc_trend = history['val_accuracy'][-1] - history['val_accuracy'][-2]
            if train_acc_trend > 0 and val_acc_trend < 0:
                overfitting_warning = "⚠️ 过拟合警告：训练精度上升但验证精度下降"
        
        # 2. 保存最佳模型
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
                'val_loss': avg_val_loss,
                'train_accuracy': accuracy,
                'train_loss': avg_loss,
            }, model_save_path)

            print(f'  ✓ 保存最佳模型 (Val Acc: {val_accuracy:.4f})')

        else:
            epochs_without_improvement += 1
            back_best = True
        
        # 3. 早停检测
        if epochs_without_improvement >= patience:
            early_stop = True
        
        # 计算epoch时间
        epoch_time = time.time() - start_time
        history['time'].append(epoch_time)
      

        # 打印epoch结果
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  训练损失: {avg_loss:.4f} | 训练准确率: {accuracy:.4f}')
        print(f'  验证损失: {avg_val_loss:.4f} | 验证准确率: {val_accuracy:.4f}')
        print(f'  学习率: {current_lr:.6f}')
        print(f'  最佳验证准确率: {best_val_accuracy:.4f} (epoch {best_epoch})')
        print(f'  无改进epoch数: {epochs_without_improvement}/{patience}')
        if overfitting_warning:
            print(f'  {overfitting_warning}')
        print(f'  训练时间: {epoch_time:.2f}秒')
        print('-' * 50)
    
    print(f'训练完成!')
    print(f'最终最佳验证准确率: {best_val_accuracy:.4f} (epoch {best_epoch})')
    
    # 加载最佳模型
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
        
    return history


# 测试代码
if __name__ == "__main__":
    # 测试模型
    model = ASTModel(
        num_classes=50,
        input_tdim=216,
        input_fdim=128,
        imagenet_pretrain=True,
        verbose=True
    )
    
    # 测试前向传播
    test_input = torch.randn(4, 1, 128, 216)
    output = model(test_input)
    print(f'输出形状: {output.shape}')
    
    test_input2 = torch.randn(4, 216, 128)
    output2 = model(test_input2)
    print(f'输出形状: {output2.shape}')