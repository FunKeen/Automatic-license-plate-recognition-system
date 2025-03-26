import cv2
import torch


class YOLOInferencer:
    def __init__(self, model_path, device):
        self.model_path = model_path
        self.device = device
        self.size = 640

        # 加载模型
        self.model = torch.hub.load(
            'yolov5',
            'custom',
            path=self.model_path,
            source='local')
        self.model.conf = 0.7  # 置信度阈值（过滤低置信度检测）
        self.model.eval()  # 设置为推理模式

    def __call__(self, img_path):
        img = cv2.imread(img_path)
        assert img is not None, f"图像 {img_path} 读取失败！"
        # 将图像转为模型输入格式
        results = self.model(img, self.size)
        # 提取检测结果
        detections = results.xyxy[0].cpu().numpy()
        # 遍历所有检测到的车牌
        plates = []
        for i, det in enumerate(detections):
            x1, y1, x2, y2, conf, cls_id = det
            if conf < self.model.conf:  # 置信度低于阈值则跳过
                continue
            # 裁剪车牌区域
            plates.append(img[int(y1):int(y2), int(x1):int(x2)])
            # # 在原始图像上绘制边界框（可视化）
            # cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # cv2.putText(img, f"Plate {conf:.2f}", (int(x1), int(y1) - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return plates
