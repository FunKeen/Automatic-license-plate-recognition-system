import torch
import cv2

from crnn.model import CRNN
from crnn.config import config


class Decoder:
    def __init__(self, characters):
        self.characters = [''] + list(characters)  # 空白符索引为0

    def decode(self, logits):
        """兼容不同维度的解码方法"""
        # 自动检测输入维度
        if logits.dim() == 3:
            # 输入形状 [seq_len, batch=1, num_classes]
            _, max_indices = torch.max(logits, dim=2)
            indices = max_indices.squeeze(1).cpu().numpy()
        else:
            # 输入形状 [seq_len, num_classes]
            _, max_indices = torch.max(logits, dim=1)
            indices = max_indices.cpu().numpy()

        text = ''
        prev = -1
        for idx in indices:
            if idx != 0 and idx != prev:
                text += self.characters[idx]
            prev = idx
        return text


class ModelCRNN:
    def __init__(self, model_path=config.saved_model_path, device=config.device):
        self.model_path = model_path
        self.device = device
        self.decoder = Decoder(config.characters)

        # 加载模型
        self.model = CRNN(len(config.characters) + 1).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

        self._warmup()

    def _warmup(self):
        """预热模型，避免首次推理延迟"""
        dummy_input = torch.randn(1, 1, config.img_height, config.img_width).to(self.device)
        with torch.no_grad():
            _ = self.model(dummy_input)

    def __call__(self, img_list):
        if img_list:
            results = []
            for img in img_list:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (config.img_width, config.img_height))
                img = (img / 255.0 - 0.5) / 0.5  # 归一化
                img_tensor = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits = self.model(img_tensor)
                results.append(self.decoder.decode(logits))
            return results

    def release(self):
        """手动释放资源"""
        del self.model
        if self.device == 'cuda':
            torch.cuda.empty_cache()

    # def preprocess(self, image_path):
    #     """预处理图像"""
    #     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #     if img is None:
    #         raise ValueError(f"无法读取图像：{image_path}")
    #
    #     img = cv2.resize(img, (config.img_width, config.img_height))
    #     img = (img / 255.0 - 0.5) / 0.5  # 归一化
    #     return torch.FloatTensor(img).unsqueeze(0).unsqueeze(0).to(self.device)
    #
    # def __call__(self, image_path):
    #     """使实例可像函数一样调用"""
    #     img_tensor = self.preprocess(image_path)
    #     with torch.no_grad():
    #         logits = self.model(img_tensor)
    #     return self.decoder.decode(logits)
    #
    # def batch_inference(self, image_paths):
    #     """批量推理"""
    #     results = []
    #     for path in image_paths:
    #         results.append(self(path))
    #     return results
    #
    # def export(self, save_path):
    #     return torch.save(self.model.state_dict(), save_path)

# if __name__ == '__main__':
#
#     img_dir = '../../../Projects1/mydataset_crnn/test'
#     count = 0
#     totle = 0
#
#     # 初始化
#     ocr = CRNNInferencer('crnn.pth', 'cpu')
#     # 批量推理
#     with open(os.path.join(img_dir, 'labels.txt'), 'r', encoding='utf-8') as f:
#         for line in f:
#             line = line.strip()
#             if line:
#                 count += 1
#                 img_name, label = line.split(' ')
#                 result = ocr(os.path.join(img_dir, 'images', img_name))
#                 if label == result:
#                     totle += 1
#     print(f'{(totle / count):.6f}')
#     # 释放
#     ocr.release()
