import pandas as pd
import librosa
import numpy as np

class Datapre:
    def __init__(self):
        meta_path = r'ESC-50/meta/esc50.csv' # # 主标签文件
        df = pd.read_csv(meta_path) # 读取主标签文件
        # 创建标签映射
        self.id_to_label = dict(zip(df['target'], df['category']))
        self.label_to_id = {label:id for id,label in self.id_to_label.items()}

        # 保存训练集统计量
        self.train_mean = None
        self.train_std = None
        
    # 加载某个fold的训练数据
    def load_fold_train(self,fold_num):

        train_df = pd.read_csv(f'data/fold{fold_num}/train.csv')
        
        # 获取文件名和标签
        X_train = train_df['filename'].tolist()
        y_train = train_df['target'].tolist()

        return X_train, y_train
    
    # 加载某个fold的测试数据
    def load_fold_test(self,fold_num):

        test_df = pd.read_csv(f'data/fold{fold_num}/test.csv')
        
        X_test = test_df['filename'].tolist()
        y_test = test_df['target'].tolist()
        
        return  X_test, y_test

    # 加载某个fold的验证数据
    def load_fold_val(self,fold_num):

        test_df = pd.read_csv(f'data/fold{fold_num}/val.csv')
        
        X_test = test_df['filename'].tolist()
        y_test = test_df['target'].tolist()
        
        return  X_test, y_test
    
    # 加载音频文件
    def load_audio(self,filename, base_path='ESC-50/audio/'):
        """根据元数据行加载音频"""

        filepath = base_path + filename
        audio, sr = librosa.load(filepath, sr=22050)  # ESC-50默认采样率

        return audio,sr

    # 计算梅尔频谱图
    def extract_mel_spectrogram(self,audio, sr=22050, n_mels=128):
        """提取梅尔频谱图"""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,          # 采样率
            n_fft=2048,        # FFT点数
            hop_length=512,    # 帧移
            n_mels=n_mels,        # 梅尔带数
            win_length=2048,   # 窗口长度
            window='hann',     # 窗函数：汉宁窗，减少边缘效应
            center=True,       # 中心化：确保帧在时间上居中
            pad_mode='reflect',# 填充模式：反射填充，边界处理更平滑
            power=2.0,         # 功率谱：平方幅度，符合音频处理惯例
            fmin=0,            # 最低频率：从0Hz开始
            fmax=sr/2,        # 最高频率：尼奎斯特频率（sr/2）
            htk=False,         # 使用Slaney梅尔尺度（更接近人耳感知）
            norm='slaney',     # 滤波器归一化：Slaney方法，能量均匀分布
            dtype=np.float32   # 数据类型：32位浮点，节省内存并保证精度
        )

        # 转换为分贝单位（更符合人耳感知）
        mel_spec_db = librosa.power_to_db(mel_spec, ref=1.0, top_db=None)

        return mel_spec_db  # 形状: (n_mels, 时间帧数)

    # 计算单个音频的梅尔频谱图
    def audio_mel_spectogram(self, filename, base_path='ESC-50/audio/'):
        audio,sr = self.load_audio(filename)
        features = self.extract_mel_spectrogram(audio)
        return features

    # 批量处理音频文件为特征矩阵
    def create_feature_dataset(self, files, base_path='ESC-50/audio/'):
        X_features = []
        
        for filename in files:
            try:
                # 提取特征
                features = self.audio_mel_spectogram(filename, base_path)
                
                # Z-score标准化：（特征-特征均值）/（特征矩阵的标准差+极小值（防止除零））
                # features = (features - np.mean(features)) / (np.std(features) + 1e-8)
                
                X_features.append(features)
                
            except Exception as e:
                print(f"处理文件 {filename} 失败: {e}")
                continue
        
        # 转换为numpy数组
        X = np.array(X_features)

        return X
    
    # 读取一个文件夹，获取训练数据，并转化为特征值和标签
    def get_train_datas(self,fold):

        X_train_audio, y_train = self.load_fold_train(fold) # x音频文件名，y为数字标签
        X_train = self.create_feature_dataset(X_train_audio)

        # 计算训练集统计量
        self.train_mean = np.mean(X_train)
        self.train_std = np.std(X_train)
        print(f"Fold {fold} 训练集统计量: mean={self.train_mean:.4f}, std={self.train_std:.4f}")
        
        # 归一化训练集
        X_train = (X_train - self.train_mean) / (self.train_std + 1e-8) * 0.5

        return X_train,y_train

    # 读取一个文件夹，获取测试数据，并转化为特征值和标签
    def get_test_datas(self,fold):
        
        """读取测试数据，用训练集的参数归一化"""
        if self.train_mean is None or self.train_std is None:
            raise ValueError("请先调用get_train_datas()获取训练集统计量")
        
        X_test_audio, y_test = self.load_fold_test(fold)
        X_test_raw = self.create_feature_dataset(X_test_audio)
        
        # 用训练集的统计量归一化测试集
        X_test = (X_test_raw - self.train_mean) / (self.train_std + 1e-8) * 0.5
        
        return X_test, y_test
    
    
    # 读取一个文件夹，获取验证数据，并转化为特征值和标签
    def get_val_datas(self,fold):
        """读取验证数据，用训练集的参数归一化"""
        if self.train_mean is None or self.train_std is None:
            raise ValueError("请先调用get_train_datas()获取训练集统计量")
        
        X_val_audio, y_val = self.load_fold_val(fold)
        X_val_raw = self.create_feature_dataset(X_val_audio)
        
        # 用训练集的统计量归一化验证集
        X_val = (X_val_raw - self.train_mean) / (self.train_std + 1e-8) * 0.5

        return X_val,y_val

