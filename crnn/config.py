import torch


class Config:
    # 训练参数
    batch_size = 128
    lr = 0.001
    epochs = 50
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 数据参数
    train_data_path = '../mydataset_crnn/train'
    val_data_path = '../mydataset_crnn/val'
    characters = 'ABCDEFGHJKLMNOPQRSTUVWXYZ0123456789皖沪津渝冀晋蒙辽吉黑苏浙京闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新警学-'
    log_csv = 'results.csv'
    num_workers = 4
    img_height = 32
    img_width = 256  # 训练时统一缩放到的宽度

    # 模型参数
    hidden_size = 256
    num_layers = 2
    num_channels = 64

    # 保存路径
    crnn_model_path = 'best_crnn.pth'
    saved_model_path = 'last.pth'


config = Config()
