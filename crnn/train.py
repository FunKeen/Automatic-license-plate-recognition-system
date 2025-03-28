import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
# import Levenshtein

from crnn.dataset import OCRDataset, collate_fn
from crnn.model import CRNN
from crnn.config import config
from crnn.ModelCRNN import Decoder

from crnn.trainlog import TrainLog


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

    best_loss = float('inf')
    logger = TrainLog(config.log_csv)
    for epoch in range(config.epochs):
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
                batch_edit_distance = 0
                for i in range(logits.size(1)):
                    logit = logits[:, i:i + 1, :]
                    pred_text = decoder.decode(logit)
                    pred_texts.append(pred_text)

                for pred, true in zip(pred_texts, labels):
                    # 序列准确率
                    if pred == true:
                        batch_correct_seq += 1

                    # 计算匹配字符数
                    match_count = sum(1 for p, t in zip(pred, true) if p == t)
                    batch_correct_chars += match_count
                    batch_total_chars += len(true)

                    # # 计算编辑距离
                    # batch_edit_distance += Levenshtein.distance(pred, true)

                # 累计指标
                total_seq_accuracy += batch_correct_seq
                total_correct_chars += batch_correct_chars
                total_chars += batch_total_chars
                total_samples += len(labels)
                # total_edit_distance += batch_edit_distance

        # 计算整体指标
        avg_train_loss = np.mean(train_loss)
        avg_val_loss = np.mean(val_loss)
        seq_accuracy = total_seq_accuracy / total_samples
        char_accuracy = total_correct_chars / total_chars if total_chars > 0 else 0
        # avg_edit_distance = total_edit_distance / total_samples

        print(f'Epoch {epoch + 1}/{config.epochs} | '
              f'Train Loss: {avg_train_loss:.4f} | '
              f'Val Loss: {avg_val_loss:.4f} | '
              f'Seq Acc: {seq_accuracy:.4f} | '
              f'Char Acc: {char_accuracy:.4f} | '
              #
              )

        metrics = {
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'seq_acc': seq_accuracy,
            'char_acc': char_accuracy,
            # 'avg_edit': avg_edit_distance
        }
        logger.log_metrics(epoch + 1, metrics)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), config.crnn_model_path)
            print(f'Epoch {epoch + 1} ,Model saved')
        torch.save(model.state_dict(), config.saved_model_path)


if __name__ == '__main__':
    train()
