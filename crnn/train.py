import signal
import sys
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from crnn.dataset import OCRDataset, collate_fn
from crnn.model import CRNN
from crnn.config import config
import numpy as np

from ModelCRNN import Decoder  # 导入解码器

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true',
                        help='从最新检查点继续训练')
    parser.add_argument('--checkpoint', type=str,
                        help='指定使用的检查点文件路径')
    return parser.parse_args()


# 注册信号处理函数
def signal_handler(sig, frame):
    print("\n捕获中断信号，正在保存检查点...")
    save_checkpoint(force_save=True)
    sys.exit(0)


def save_checkpoint(epoch=None, model=None, optimizer=None,
                    best_loss=None, force_save=False):
    """保存训练状态检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'best_loss': best_loss,
        'config': vars(config)
    }

    # 两种保存情形：
    if force_save:  # 强制保存（如中断时）
        torch.save(checkpoint, f"interrupted_{config.checkpoint_path}")
        print(f"紧急检查点已保存到：interrupted_{config.checkpoint_path}")
    else:  # 常规保存
        torch.save(checkpoint, config.checkpoint_path)
        if epoch % config.autosave_interval == 0:
            dated_copy = f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M')}.pth"
            torch.save(checkpoint, dated_copy)


def load_checkpoint(model, optimizer):
    """加载检查点继续训练"""
    try:
        if config.resume_training:
            checkpoint = torch.load(config.checkpoint_path)
        else:  # 优先加载中断检查点
            checkpoint = torch.load(f"interrupted_{config.checkpoint_path}")
            print("检测到未完成的训练，自动恢复")
    except FileNotFoundError:
        print("未找到检查点，开始新训练")
        return 0, float('inf')

    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    start_epoch = checkpoint['epoch'] + 1  # 从下一个epoch开始
    best_loss = checkpoint['best_loss']

    print(f"成功恢复训练，从epoch {start_epoch}开始")
    return start_epoch, best_loss


def train():
    # 初始化数据集和数据加载器
    train_dataset = OCRDataset(config.train_data_path)
    val_dataset = OCRDataset(config.val_data_path)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, collate_fn=collate_fn, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                            shuffle=False, collate_fn=collate_fn)

    # 初始化模型
    num_classes = len(config.characters) + 1  # 包含空白符
    model = CRNN(num_classes).to(config.device)
    criterion = torch.nn.CTCLoss(blank=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 自动恢复机制
    start_epoch = 0
    if Path(config.checkpoint_path).exists() or Path(f"interrupted_{config.checkpoint_path}").exists():
        start_epoch, best_loss = load_checkpoint(model, optimizer)

    best_loss = float('inf')
    for epoch in range(start_epoch, config.epochs):
        # 训练阶段
        model.train()
        train_loss = []
        for images, targets, target_lengths, _ in tqdm(train_loader, desc=f'Epoch {epoch + 1}'):
            # 转移数据到设备
            images = images.to(config.device)
            targets = targets.to(config.device)
            target_lengths = target_lengths.to(config.device)

            logits = model(images)
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)

            input_lengths = torch.full(
                (images.size(0),),  # size参数
                logits.size(0),  # fill_value参数
                dtype=torch.long,
                device=config.device
            )

            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            train_loss.append(loss.item())

        # 定期保存检查点
        if epoch % config.autosave_interval == 0:
            save_checkpoint(epoch, model, optimizer, best_loss)

        # 每个epoch后更新学习率
        scheduler.step()

        # 验证阶段
        model.eval()
        val_loss = []
        total_edit_distance = 0
        total_chars = 0
        total_seq_accuracy = 0
        total_samples = 0
        total_correct_chars = 0
        with torch.no_grad():
            for images, targets, target_lengths, labels in val_loader:  # 获取labels
                # 转移数据到设备
                images = images.to(config.device)
                targets = targets.to(config.device)
                target_lengths = target_lengths.to(config.device)

                # 前向传播
                logits = model(images)
                log_probs = torch.nn.functional.log_softmax(logits, dim=2)

                # 计算损失
                seq_length = logits.size(0)
                input_lengths = torch.full(
                    (images.size(0),),
                    seq_length,
                    dtype=torch.long,
                    device=config.device
                )
                loss = criterion(log_probs, targets, input_lengths, target_lengths)
                val_loss.append(loss.item())

                # 解码预测结果
                decoder = Decoder(config.characters)
                pred_texts = []
                # 计算指标
                batch_correct_seq = 0
                batch_correct_chars = 0
                batch_total_chars = 0
                for i in range(logits.size(1)):
                    logit = logits[:, i:i + 1, :]
                    pred_text = decoder.decode(logit)
                    pred_texts.append(pred_text)

                for pred, true in zip(pred_texts, labels):
                    # 序列准确率
                    if pred == true:
                        batch_correct_seq += 1

                    # 字符准确率和编辑距离
                    min_len = min(len(pred), len(true))
                    max_len = max(len(pred), len(true))

                    # 计算匹配字符数
                    match_count = sum(1 for p, t in zip(pred, true) if p == t)
                    batch_correct_chars += match_count
                    batch_total_chars += len(true)


                # 累计指标
                total_seq_accuracy += batch_correct_seq
                total_correct_chars += batch_correct_chars
                total_chars += batch_total_chars
                total_samples += len(labels)

        # 计算整体指标
        avg_train_loss = np.mean(train_loss)
        avg_val_loss = np.mean(val_loss)
        seq_accuracy = total_seq_accuracy / total_samples
        char_accuracy = total_correct_chars / total_chars if total_chars > 0 else 0

        print(f'Epoch {epoch + 1}/{config.epochs} | '
              f'Train Loss: {avg_train_loss:.4f} | '
              f'Val Loss: {avg_val_loss:.4f} | '
              f'Seq Acc: {seq_accuracy:.4f} | '
              f'Char Acc: {char_accuracy:.4f} | ')
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), config.saved_model_path)
            print(f'Epoch {epoch + 1} ,Model saved')


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)  # 添加信号注册
    args = parse_args()
    if args.checkpoint:
        config.checkpoint_path = args.checkpoint
    config.resume_training = args.resume
    train()
