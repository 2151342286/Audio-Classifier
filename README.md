# 基于傅里叶变换和神经网络的音频分类器
## 技术说明：
将数据处理成梅尔频谱图  
共四个模型，cnn，cnn+lstm，cnn+transformer，ast  
数据集和部分模型参数太大，数据集自行下载，模型参数提供cnn作为示例，其余可以自己训练测试  
## 运行main.py --help可得到运行指令详情：
options:
  -h, --help            show this help message and exit
  --fold FOLD           交叉验证折数，默认1
  --model {ast,cnn,cnn_bilstm,cnn_transformer}
                        模型名称，可选: ast, cnn, cnn_bilstm, cnn_transformer，默认cnn
  --train               是否训练模型，如果指定则训练，否则加载预训练模型
  --batch_size BATCH_SIZE
                        批大小，默认32
  --epochs EPOCHS       训练轮次，默认40
  --lr LR               学习率，默认0.001
  
