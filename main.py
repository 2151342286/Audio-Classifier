from data_pre import Datapre
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch.nn as nn
import json
import matplotlib.pyplot as plt
from collections import OrderedDict
import argparse
import os  

# 导入自己写的模型
import ast_model
import cnn_model
import cnn_bilstm_model
import cnn_transformer_model

def create_datasets(fold=1, batch_size=32):
    """创建训练集,验证集和测试集"""
    date = Datapre()
    
    # 获取训练数据
    X_train, y_train = date.get_train_datas(fold)
    X_train = torch.tensor(X_train).float().unsqueeze(1)  # [batch, 128, 216] 或 [batch, 216, 128]
    y_train = torch.tensor(y_train).long()
    
    # 获取验证数据
    X_val, y_val = date.get_val_datas(fold)
    X_val = torch.tensor(X_val).float().unsqueeze(1)
    y_val = torch.tensor(y_val).long()

    # 获取测试数据
    X_test, y_test = date.get_test_datas(fold)
    X_test = torch.tensor(X_test).float().unsqueeze(1)
    y_test = torch.tensor(y_test).long()

    print(f"训练集形状: {X_train.shape}, 标签形状: {y_train.shape}")
    print(f"验证集形状: {X_val.shape}, 标签形状: {y_val.shape}")
    print(f"测试集形状: {X_test.shape}, 标签形状: {y_test.shape}")
    print(f"类别数量: {len(torch.unique(y_train))}")
    
    # 创建数据集和数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader

def create_model(name):

    if name == 'ast':
        return ast_model.ASTModel(
            num_classes=50,
            input_tdim=216,  # 新的时间维度
            input_fdim=128,
            fstride=10,
            tstride=10,
            imagenet_pretrain=True, # 使用imageNet预训练模型
            model_size='base384', # 只能用这个模型，其他没写
            verbose=True # 打印模型总结
        )
    
    if name == 'cnn':
        return cnn_model.CNNModel()
    
    if name == 'cnn_bilstm':
        return cnn_bilstm_model.CnnBilstmModule()
    
    if name == 'cnn_transformer':
        return cnn_transformer_model.CNNTransformerModel()
    
def train(name, model, train_loader,val_loader, fold, epochs=40, lr=1e-3):

    if name == 'ast':
        history = ast_model.train(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    epochs=epochs,              
                    lr=lr,      
                    fold=fold
                )
        return history
    
    if name == 'cnn':
        history = cnn_model.train(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    epochs=epochs,               
                    lr=lr,      
                    fold=fold
                )
        return history
    
    if name == 'cnn_bilstm':
        history = cnn_bilstm_model.train(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    epochs=epochs,               
                    lr=lr,      
                    fold=fold
                )
        return history
    
    if name == 'cnn_transformer':
        history = cnn_transformer_model.train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,             
            lr=lr,      
            fold=fold
        )
        return history
    
def evaluate(model, test_loader):
    """
    评估模型
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not isinstance(model, nn.DataParallel):
        model = nn.DataParallel(model)
    
    model = model.to(device)
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    test_loss = 0.0
    batch_count = 0
    
    print('开始评估...')
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            
            # 收集结果
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # 统计准确率
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            batch_count += 1
            
            if (batch_idx + 1) % 20 == 0:
                print(f'  处理批次 {batch_idx+1}/{len(test_loader)}')
    
    accuracy = correct / total if total > 0 else 0
    avg_loss = test_loss / batch_count if batch_count > 0 else 0
    
    print('评估完成!')
    print(f'  测试样本数: {total}')
    print(f'  正确数: {correct}')
    print(f'  测试准确率: {accuracy:.4f}')
    print(f'  测试损失: {avg_loss:.4f}')
    
    return accuracy, np.array(all_predictions), np.array(all_targets)

def save_training_logs(history, filename, fold):
    """保存训练日志"""
    # 1. 保存为JSON
    with open(filename, 'w') as f:
        json.dump(history, f, indent=2)
    
    # 2. 绘制损失准确率图
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.title('Loss Curve')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_accuracy'], label='Train')
    plt.plot(history['val_accuracy'], label='Val')
    plt.title('Accuracy Curve')
    plt.legend()
    
    plt.savefig('training_curves_' + str(fold) + '.png')
    plt.close()
    
    print(f"✓ 日志已保存: {filename}, training_curves_{fold}.png")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='音频分类模型训练与评估')
    parser.add_argument('--fold', type=int, default=1, help='交叉验证折数，默认1')
    parser.add_argument('--model', type=str, default='cnn', choices=['ast', 'cnn', 'cnn_bilstm', 'cnn_transformer'], 
                        help='模型名称，可选: ast, cnn, cnn_bilstm, cnn_transformer，默认cnn')
    parser.add_argument('--train', action='store_true', help='是否训练模型，如果指定则训练，否则加载预训练模型')
    parser.add_argument('--batch_size', type=int, default=32, help='批大小，默认32')
    parser.add_argument('--epochs', type=int, default=40, help='训练轮次，默认40')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率，默认0.001')
    
    args = parser.parse_args()
    fold = args.fold
    model_name = args.model
    trainModel = args.train
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr

    path = './model_param/' + model_name + '/' + str(fold) + '.pth'
    history_path = './model_history/' + model_name + '/' + str(fold) + '.json'

    print(f"开始音频分类任务 - {model_name}模型")
    
    # 1. 创建数据集
    print("\n1. 加载数据集...")
    train_loader, val_loader, test_loader = create_datasets(fold=fold, batch_size=batch_size)
    
    # 2. 创建模型
    print(f"\n2. 创建{model_name}模型...") 

    model = create_model(model_name)
    
    if(trainModel):
        # 3. 训练模型
        print("\n3. 开始训练...")
        history = train(model_name, model,train_loader, lr=lr, val_loader=val_loader, epochs=epochs, fold = fold)
        # 保存日志
        save_training_logs(history, history_path, fold)

    else:
        # 直接调用模型参数
        print("\n3. 开始加载参数...")
        checkpoint = torch.load(path, weights_only=True)

        # 提取state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # 移除'module.'前缀
        if isinstance(state_dict, dict) and state_dict and list(state_dict.keys())[0].startswith('module.'):
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k[7:]] = v  # 移除'module.'
        else:
            new_state_dict = state_dict

        # 加载模型
        model.load_state_dict(new_state_dict, strict=False)
        print("✓ 模型加载完成")
    # 4. 评估模型
    print("\n4. 评估模型...")
    accuracy, predictions, targets = evaluate(model, test_loader)
    print(f"正确率：{accuracy}")

if __name__ == "__main__":
    main()