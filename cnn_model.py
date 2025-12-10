import torch.nn as nn
import torch.optim as optim
import torch
import time

class CNNModel(nn.Module):

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
            nn.MaxPool2d(2), # 16*27
        )
        
        # 全局平均池化，输出1×1
        # 对每个128×16×27的特征图，计算所有16×27=432个值的平均值，输出128×1×1。
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 定义全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3), # 30%随机失活防过拟合
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        '''
            输入(batch, 1, 128, 216)
            卷积层后(batch, 128, 16, 27)
            全局平均池化后(batch, 128, 1, 1)
            展平后(batch, 128)
            全连接层后(batch, 50)
        '''
        x = self.cnn(x) # 输入通过卷积层提取特征
        x = self.global_pool(x) # 全局平均池化，每个特征图变1×1
        x = x.view(x.size(0), -1) # 展平，保持批次维度，合并其他维度
        x = self.fc_layers(x) # 通过全连接层分类

        return x
    


def train(model, train_loader, val_loader, fold, epochs=30, lr=1e-4):
    """
    训练模型并返回训练历史
    
    参数:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        fold: 交叉验证的折数（用于保存模型）
        epochs: 训练轮数
        lr: 学习率
    """
    
    # 训练历史
    history = {
        'train_loss': [], 'train_accuracy': [], 
        'val_loss': [], 'val_accuracy': [],
        'lr': [], 'time': []
    }
    
    # 最佳模型保存路径
    model_save_path = './model_param/cnn/' + str(fold) + '.pth'
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 用于保存最佳模型
    best_val_accuracy = 0.0
    best_model_state = None
    
    print(f"开始训练第 {fold} 折，共 {epochs} 个epoch")
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += y.size(0)
            train_correct += (predicted == y).sum().item()
        
        train_accuracy = train_correct / train_total if train_total > 0 else 0
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += y.size(0)
                val_correct += (predicted == y).sum().item()
        
        val_accuracy = val_correct / val_total if val_total > 0 else 0
        avg_val_loss = val_loss / len(val_loader)
        
        # 记录历史
        history['train_loss'].append(avg_train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        history['time'].append(time.time() - epoch_start)
        
        # 保存最佳模型
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, model_save_path)
            print(f'  ✓ 保存最佳模型到 {model_save_path} (Val Acc: {val_accuracy:.4f})')
        
        # 打印进度
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  训练损失: {avg_train_loss:.4f} | 训练准确率: {train_accuracy:.4f}')
        print(f'  验证损失: {avg_val_loss:.4f} | 验证准确率: {val_accuracy:.4f}')
        print(f'  最佳验证准确率: {best_val_accuracy:.4f}')
        print(f'  时间: {history["time"][-1]:.2f}秒')
        print('-' * 50)
    
    # 训练完成后加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print(f'第 {fold} 折训练完成，最佳验证准确率: {best_val_accuracy:.4f}')
    
    return history
