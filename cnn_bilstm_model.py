import torch.nn as nn
import torch.optim as optim
import torch
import os
import torch
import time

class CnnBilstmModule(nn.Module):

    def __init__(self, num_classes=50):
        super().__init__()
        # 输入: (batch, 1, 128, 216)
        # 定义卷积层序列,CNN层
        self.cnn = nn.Sequential(
            # 输入1通道，输出32通道，3×3卷积，边缘填充1
            # 即采用32个不同卷积核对原数据边缘填充1后结果进行卷积操作，输出32个尺寸与原数据相同的特征图
            nn.Conv2d(1, 32, 3, padding=1), 

            # 批量归一化，32通道；即对32个特征图归一化处理，计算均值方差，标准化为均值0方差1
            nn.BatchNorm2d(32),

            nn.ReLU(), # ReLU激活函数

            # 最大池化，每个2×2区域取最大值，输出尺寸减半
            nn.MaxPool2d(2),  # 64*108
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # 32*54
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.Dropout2d(0.3),
            nn.MaxPool2d(2), # 16*27
        )

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=128,      # 特征维度
            hidden_size=256,     # LSTM隐层
            num_layers=1,        # LSTM层数
            batch_first=True,
            bidirectional=True,
        )

         # 注意力机制层
        self.attention = nn.Sequential(
            nn.Linear(512, 512),  # 双向LSTM输出是hidden_size*2
            nn.Tanh(),
            nn.Linear(512, 1)
        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(512, 256),  # 双向LSTM输出512维
            nn.ReLU(),
            nn.Dropout(0.1), # 随机失活防过拟合
            
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):

        # CNN特征提取
        cnn_features = self.cnn(x)  # [batch, 128, 16, 27]
        
        # 重塑为序列格式 [batch, seq_len, features]
        batch, channels, height, width = cnn_features.shape
        cnn_features = cnn_features.permute(0, 2, 3, 1)  # [batch, height, width, channels]
        sequence = cnn_features.reshape(batch, height * width, channels)
        
        # LSTM处理
        lstm_out, _ = self.lstm(sequence)  # [batch, seq_len, 256]
        
        # 注意力机制
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # [batch, seq_len, 1]
        weighted = (lstm_out * attn_weights).sum(dim=1)  # [batch, 512]

        # 分类
        output = self.fc(weighted)
        return output


def train(model, train_loader, val_loader, fold, epochs=30, lr=0.001, patience=50):
    """
    训练模型并返回训练历史
    参数:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        fold: 交叉验证的折数（用于保存模型）
        epochs: 训练轮数
        lr: 学习率
        patience: 早停耐心值
    """
    
    # 最佳模型保存路径
    model_save_path = './model_param/cnn_bilstm/' + str(fold) + '.pth'
    
    # 创建保存目录
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss() # 使用交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=3.5e-4) 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='max',           # 监控验证准确率（最大越好）
                factor=0.5,           # 每次降低为原来的0.5倍
                patience=3,           # 连续3个epoch没提升才降低
                threshold=0.01,       # 至少提升0.01才算"提升"
                min_lr=1e-6           # 最小学习率
            )

    # 训练历史记录
    history = {
        'train_loss': [], 'train_accuracy': [],
        'val_loss': [], 'val_accuracy': [],
        'lr': [], 'time': []
    }

    # 早停和最佳模型追踪
    best_val_accuracy = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    early_stop = False

    back_best = False

    print(f'开始训练第 {fold} 折，共{epochs}个epoch，初始学习率: {lr}')

    for epoch in range(epochs):

        if early_stop:
            print(f'早停触发，训练提前结束')
            break

        epoch_start = time.time()
        
        # ========== 训练阶段 ==========
        model.train() # 将模型调至训练模式
        total_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for X,y in train_loader:
            X, y = X.to(device), y.to(device) # 将数据迁移到gpu
            optimizer.zero_grad() # 清空梯度
            output = model(X) # 前向传播
            loss = criterion(output,y) # 计算交叉熵
            loss.backward() # 反向传播
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 梯度裁剪（防止梯度爆炸）
            optimizer.step() # 更新参数
            
            total_loss += loss.item()

            # 计算训练准确率
            _, predicted = torch.max(output, 1)
            train_total += y.size(0)
            train_correct += (predicted == y).sum().item()

        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = train_correct / train_total if train_total > 0 else 0

        # ========== 验证阶段 ==========
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for val_X, val_y in val_loader:
                val_X, val_y = val_X.to(device), val_y.to(device)
                
                val_output = model(val_X)
                val_batch_loss = criterion(val_output, val_y)
                
                val_loss += val_batch_loss.item()
                
                _, val_predicted = torch.max(val_output, 1)
                val_total += val_y.size(0)
                val_correct += (val_predicted == val_y).sum().item()
        
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_accuracy = val_correct / val_total if val_total > 0 else 0
        scheduler.step(val_accuracy)  # 调整学习率

        model.train()
        
        current_lr = optimizer.param_groups[0]['lr']

        # ========== 记录历史数据 ==========
        history['train_loss'].append(avg_train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['lr'].append(current_lr)
        history['time'].append(time.time() - epoch_start)

        # ========== 检测过拟合 ==========
        overfitting_warning = ""
        if epoch >= 2:
            train_acc_trend = history['train_accuracy'][-1] - history['train_accuracy'][-2]
            val_acc_trend = history['val_accuracy'][-1] - history['val_accuracy'][-2]
            if train_acc_trend > 0 and val_acc_trend < 0:
                overfitting_warning = "⚠️ 过拟合警告：训练精度上升但验证精度下降"
                
            # 记录下降
            if history['val_accuracy'][-1] - best_val_accuracy < -0.05:  # 明显下降
                back_best = True

        # ========== 保存最佳模型 ==========
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            
            # 保存最佳模型到文件
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
                'val_loss': avg_val_loss,
                'train_accuracy': train_accuracy,
                'train_loss': avg_train_loss,
                'lr': current_lr,
            }, model_save_path)
            print(f'  ✓ 保存最佳模型到 {model_save_path} (Val Acc: {val_accuracy:.4f})')
        else:
            epochs_without_improvement += 1

        # ========== 早停检测 ==========
        if epochs_without_improvement >= patience:
            early_stop = True

        # ========== 打印训练信息 ==========
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  训练损失: {avg_train_loss:.4f} | 训练准确率: {train_accuracy:.4f}')
        print(f'  验证损失: {avg_val_loss:.4f} | 验证准确率: {val_accuracy:.4f}')
        print(f'  学习率: {current_lr:.6f}')
        print(f'  最佳验证准确率: {best_val_accuracy:.4f} (epoch {best_epoch})')
        print(f'  无改进epoch数: {epochs_without_improvement}/{patience}')
        
        if overfitting_warning:
            print(f'  {overfitting_warning}')
        
    
        # ========== 回退机制检测 ==========
        # 检测性能持续下降
        if back_best:
            print(f"验证精度下降超过5%，触发回退机制")

            checkpoint = torch.load(model_save_path,weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])

            print(f"  ↺ 回滚到epoch {best_epoch}({best_val_accuracy})的最佳模型状态")
            back_best = False

        print(f'  训练时间: {history["time"][-1]:.2f}秒')
        print('-' * 50)

       

    print(f'第 {fold} 折训练完成!')
    print(f'最终最佳验证准确率: {best_val_accuracy:.4f} (epoch {best_epoch})')
    
    # 加载最佳模型
    checkpoint = torch.load(model_save_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

    return history