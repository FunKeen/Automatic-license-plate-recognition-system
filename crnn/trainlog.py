import csv
from pathlib import Path


class TrainLog:
    def __init__(self, csv_path):
        self.csv_path = Path(csv_path)
        self._init_csv_file()

    def _init_csv_file(self):
        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'epoch',
                    'train_loss',
                    'val_loss',
                    'seq_acc',
                    'char_acc',
                    'avg_edit'
                ])

    def log_metrics(self, epoch, metrics):
        """记录指标到CSV"""
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                metrics['train_loss'],
                metrics['val_loss'],
                metrics['seq_acc'],
                metrics['char_acc'],
                metrics['avg_edit']
            ])
