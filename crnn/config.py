import torch


class Config:
    # 在原有配置基础上新增
    checkpoint_path = 'checkpoint.pth'  # 用于保存继续训练所需的全量信息
    resume_training = False  # 是否从检查点恢复训练
    autosave_interval = 10  # 自动保存间隔（epoch数）

    # 训练参数
    batch_size = 128
    lr = 0.001
    epochs = 50
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 数据参数
    train_data_path = '../mydataset_crnn/train'
    val_data_path = '../mydataset_crnn/val'
    characters = 'ABCDEFGHJKLMNOPQRSTUVWXYZ0123456789皖沪津渝冀晋蒙辽吉黑苏浙京闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新警学-'
    num_workers = 4
    img_height = 32
    img_width = 256  # 训练时统一缩放到的宽度

    # 模型参数
    hidden_size = 256
    num_layers = 2
    num_channels = 64

    # 保存路径
    saved_model_path = 'checkpoint.pth'
    crnn_model_path = 'crnn/crnn.pth'


config = Config()
