import cv2
import torch
from PIL.ImageQt import QImage, QPixmap
from PySide6.QtGui import QPainter, QPen, QColor, QFont
from PySide6.QtWidgets import (QMainWindow, QPushButton,
                               QLabel, QFileDialog, QWidget, QVBoxLayout, )
from PySide6.QtCore import Qt, QRect

from crnn.ModelCRNN import Decoder
from crnn.model import CRNN

characters = 'ABCDEFGHJKLMNOPQRSTUVWXYZ0123456789皖沪津渝冀晋蒙辽吉黑苏浙京闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新警学-'


class Alprs(QMainWindow):
    def __init__(self):
        super().__init__()
        self.btn_open = None
        self.label = None
        self.image_label = None
        # 配置画笔属性（颜色、线宽）
        self.pen = QPen(QColor(255, 0, 0))
        self.pen.setWidth(2)
        # 配置字体大小
        self.font = QFont()
        self.font.setPointSize(12)

        self.init_ui()
        self.yolo = self.load_yolo('yolov5/runs/train/exp2/weights/best.pt')
        self.crnn = self.load_crnn('crnn/crnn.pth', 'cpu')
        self.decoder = Decoder(characters)

    def load_yolo(self, model_path):
        # 加载模型
        model = torch.hub.load(
            'yolov5',
            'custom',
            path=model_path,
            source='local')
        model.conf = 0.5  # 置信度阈值（过滤低置信度检测）
        model.eval()  # 设置为推理模式
        return model

    def load_crnn(self, model_path, device):
        # 加载模型
        model = CRNN(len(characters) + 1).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model

    def init_ui(self):
        self.setWindowTitle('车牌自动识别系统')
        self.setGeometry(100, 100, 400, 650)

        # 创建主控件和布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # 顶部按钮
        self.btn_open = QPushButton('选择图片')
        self.btn_open.clicked.connect(self.open_image)
        main_layout.addWidget(self.btn_open, alignment=Qt.AlignmentFlag.AlignTop)

        # 添加标签
        self.label = QLabel('请选择图片!')
        main_layout.addWidget(self.label, alignment=Qt.AlignmentFlag.AlignCenter)

        # 图片显示区域
        self.image_label = QLabel()
        self.image_label.setMinimumSize(1, 1)
        main_layout.addWidget(self.image_label, alignment=Qt.AlignmentFlag.AlignBottom)

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            '选择图片文件',
            '',
            '图片文件 (*.png *.jpg *.jpeg *.bmp *.gif)'
        )
        if file_path:
            img, results = self.plate_positioning_recognition(file_path)
            out_img = self.draw(img, results)
            if not img is None:
                self.image_label.setPixmap(out_img)
                self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            else:
                self.label.setText(f'无法加载图片')
                print(f'无法加载图片')

    def plate_positioning_recognition(self, img_path):
        img = cv2.imread(img_path)
        # 图像预处理
        pre_img = self.yolo(img, 640)
        # 提取检测结果
        detections = pre_img.xyxy[0].cpu()
        # 遍历所有检测到的车牌裁剪车牌区域
        results = []
        for i, det in enumerate(detections):
            x1, y1, x2, y2, conf, cls_id = det
            if det[4] < self.yolo.conf:  # 置信度低于阈值则跳过
                continue
            plate = self.auto_recognition(img[int(y1):int(y2), int(x1):int(x2)])
            results.append((plate, det))
        return self.numpy2qimage(img), results

    def auto_recognition(self, plate):
        img = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (256, 32))
        img = (img / 255.0 - 0.5) / 0.5  # 归一化
        img_tensor = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0).to('cpu')
        with torch.no_grad():
            logits = self.crnn(img_tensor)
        return self.decoder.decode(logits)

    def numpy2qimage(self, img):
        if img.shape[2] == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img
        height, width, channel = img_rgb.shape
        bytes_per_line = 3 * width
        return QImage(img_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)

    # 可视化结果
    def draw(self, img, results):
        if results:
            # 缩小图片，方便显示
            scaled_width = int(img.width() * 0.5)
            scaled_height = int(img.height() * 0.5)
            img = img.scaled(
                scaled_width,
                scaled_height,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            painter = QPainter(img)
            painter.setFont(self.font)
            painter.setPen(self.pen)
            for plate, det in results:
                x1, y1, x2, y2 = list(map(lambda x: x * 0.5, det[:4]))
                conf, cls_id = det[4:]
                painter.drawRect(QRect(int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
                if cls_id == 1:
                    painter.drawText(int(x1), int(y1) - 10, f'蓝牌：{plate} {conf:.2f}')
                    print(f'蓝牌：{plate} {conf:.2f}')
                elif cls_id == 0:
                    painter.drawText(int(x1), int(y1) - 10, f'绿牌：{plate} {conf:.2f}')
                    print(f'绿牌：{plate} {conf:.2f}')
            painter.end()
            self.label.setText(f'结果如下')
            return QPixmap.fromImage(img)
        else:
            self.label.setText(f'未识别到车牌')
            print(f'未识别到车牌')
            return QPixmap.fromImage(img)
