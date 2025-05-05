import os
import cv2
import torch
from torchvision import transforms
from torch.utils.data import Dataset

from crnn.config import config


class OCRDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.char2idx = {c: i + 1 for i, c in enumerate(config.characters)}
        self.idx2char = {i + 1: c for i, c in enumerate(config.characters)}
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.labels = []
        labels_path = os.path.join(data_dir, 'labels.txt')
        with open(labels_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    name, label = line.split(' ', 1)
                    img_path = os.path.join(self.data_dir, 'images', name)
                    if os.path.exists(img_path):
                        self.labels.append((img_path, label))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path, label = self.labels[idx]
        # 读取并预处理图像
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (config.img_width, config.img_height))
        img = self.transform(img)  # (1, H, W)
        # 转换标签为索引
        target = [self.char2idx[c] for c in label]
        return img, torch.LongTensor(target), label


def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    targets = torch.cat([item[1] for item in batch])
    target_lengths = torch.LongTensor([len(item[1]) for item in batch])
    labels = [item[2] for item in batch]
    return images, targets, target_lengths, labels
