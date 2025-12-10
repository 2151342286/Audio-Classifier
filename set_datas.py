import pandas as pd
import os
# 数据结构
# ['文件名', '交叉验证分组号', '数字类别ID', '类别名称', '是否属于10类简化版子集', '原始音频文件编号', '同一文件的不同剪辑片段']
#     filename  fold  target  category  esc10      src_file  take
# 0  1-100032-A.wav     1       0  dog       True  100032  A
# 1  1-100038-A.wav     1      36  chirping_birds  False  100038  A
# 2  1-100210-A.wav     1      19  thunderstorm    False  100210  A
meta_path = r'ESC-50/meta/esc50.csv' # # 主标签文件
df = pd.read_csv(meta_path) # 读取主标签文件

# 创建数据集
train_datas = []
val_datas = []
test_datas = []

# 创建data目录
os.makedirs('data', exist_ok=True)
# 5折交叉验证循环
for test_fold in [1, 2, 3, 4, 5]:
    # 划分数据
    train_data = df[df['fold'] != test_fold]  # 1600条
    test_data = df[df['fold'] == test_fold]   # 400条
    train_datas.append(train_data)
    test_datas.append(test_data)
    fold_dir = f'data/fold{test_fold}'
    os.makedirs(fold_dir, exist_ok=True)
    # 保存CSV
    train_data.to_csv(f'{fold_dir}/train.csv', index=False)
    test_data.to_csv(f'{fold_dir}/test.csv', index=False)

# 5折交叉验证循环
for test_fold in [1, 2, 3, 4, 5]:
    # 测试集：当前fold
    test_data = df[df['fold'] == test_fold]  # 400条
    
    # 剩余数据作为训练+验证集
    remaining_data = df[df['fold'] != test_fold]  # 1600条
    
    # 从剩余数据中划分出验证集（400条）和训练集（1200条）
    # 这里使用另一个fold作为验证集
    # 例如：如果test_fold=1，使用fold=2作为验证集
    val_fold = test_fold % 5 + 1  # 循环获取下一个fold
    train_data = remaining_data[remaining_data['fold'] != val_fold]  # 1200条
    val_data = remaining_data[remaining_data['fold'] == val_fold]  # 400条
    
    # 验证一下数据量
    print(f"Fold {test_fold}: 训练集={len(train_data)}, 验证集={len(val_data)}, 测试集={len(test_data)}")
    
    train_datas.append(train_data)
    val_datas.append(val_data)
    test_datas.append(test_data)
    
    # 创建fold目录
    fold_dir = f'data/fold{test_fold}'
    os.makedirs(fold_dir, exist_ok=True)
    
    # 保存CSV文件
    train_data.to_csv(f'{fold_dir}/train.csv', index=False)
    val_data.to_csv(f'{fold_dir}/val.csv', index=False)
    test_data.to_csv(f'{fold_dir}/test.csv', index=False)

print("数据集划分完成！")