import cv2
import torch

from crnn.ModelCRNN import ModelCRNN


class ModelYOLO:
    def __init__(self, model_path):
        self.size = 640

        # 加载模型
        self.model = torch.hub.load(
            'C:/Users/Keen/PycharmProjects/Alprs/yolov5',
            'custom',
            path=model_path,
            source='local')
        self.model.conf = 0.7  # 置信度阈值（过滤低置信度检测）
        self.model.eval()  # 设置为推理模式

        self.crnn = ModelCRNN('crnn/best_crnn.pth')

    def __call__(self, img):
        # 将图像转为模型输入格式
        results = self.model(img, self.size)
        # 提取检测结果
        detections = results.xyxy[0].cpu().numpy()
        # 遍历所有检测到的车牌裁剪车牌区域
        plate = []
        plates = []
        for i, det in enumerate(detections):
            x1, y1, x2, y2, conf, cls_id = det
            if conf < self.model.conf:  # 置信度低于阈值则跳过
                continue
            plate = self.crnn([img[int(y1):int(y2), int(x1):int(x2)]])
            plates.append([i, plate])
            # 在原始图像上绘制边界框（可视化）
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(img, f"Id:{i} cong:{conf:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return img, plate, plates
