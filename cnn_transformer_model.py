import torch.nn as nn
import torch.optim as optim
import torch
import time
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    
class CNNTransformerModel(nn.Module):

    def __init__(self, num_classes=50, d_model=128, nhead=4, num_layers=2, dropout=0.3):
        super().__init__()
        
        # CNN 部分（保持不变，用于提取局部特征）
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
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
        
        # 全局平均池化（可选，用于降维）
        self.global_pool = nn.AdaptiveAvgPool2d((1, None))  # 保持时间维度
        
        # Transformer 部分
        self.d_model = d_model
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dropout=dropout,
                batch_first=True  # 输入形状: (batch, seq_len, d_model)
            ),
            num_layers=num_layers
        )
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # 全连接分类头
        self.fc = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        """
        输入: (batch, 1, 128, 216)
        输出: (batch, num_classes)
        """
        # CNN 特征提取
        x = self.cnn(x)  # (batch, 128, 16, 27)
        
        # 重新排列维度，适应 Transformer
        # 将高度维度（频带）视为通道，时间维度视为序列长度
        x = x.squeeze(1) if x.size(1) == 1 else x  # (batch, 128, 16, 27)
        x = x.permute(0, 2, 3, 1)  # (batch, 16, 27, 128)
        batch_size, h, w, c = x.shape
        x = x.reshape(batch_size, h * w, c)  # (batch, seq_len=432, d_model=128)
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # Transformer 编码
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        
        # 全局平均池化（对序列维度）
        x = x.mean(dim=1)  # (batch, d_model)
        
        # 分类
        x = self.fc(x)  # (batch, num_classes)
        
        return x


def train(model, train_loader, val_loader, fold, epochs=30, lr=3e-4):
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
    model_save_path = './model_param/cnn_transformer/' + str(fold) + '.pth'
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay = 5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max',           # 监控验证准确率（最大越好）
            factor=0.5,           # 每次降低为原来的0.5倍
            patience=3,           # 连续3个epoch没提升才降低
            threshold=0.01,       # 至少提升0.01才算"提升"
            min_lr=1e-6           # 最小学习率
        )
    
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
        scheduler.step(val_accuracy)  # 调整学习率
        current_lr = optimizer.param_groups[0]['lr']
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
        print(f'  学习率: {current_lr:.6f}')
        print(f'  最佳验证准确率: {best_val_accuracy:.4f}')
        print(f'  时间: {history["time"][-1]:.2f}秒')
        print('-' * 50)
    
    # 训练完成后加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print(f'第 {fold} 折训练完成，最佳验证准确率: {best_val_accuracy:.4f}')
    
    return history