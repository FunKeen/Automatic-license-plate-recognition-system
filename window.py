import cv2
import torch
from PIL.ImageQt import QImage, QPixmap
from PySide6.QtGui import QPainter, QPen, QColor
from PySide6.QtWidgets import (QMainWindow, QPushButton,
                               QLabel, QFileDialog, QWidget, QVBoxLayout,
                               QScrollArea)
from PySide6.QtCore import Qt

from crnn.ModelCRNN import Decoder
from crnn.model import CRNN

characters = 'ABCDEFGHJKLMNOPQRSTUVWXYZ0123456789皖沪津渝冀晋蒙辽吉黑苏浙京闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新警学-'


class Alprs(QMainWindow):
    def __init__(self):
        super().__init__()
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
        model.conf = 0.7  # 置信度阈值（过滤低置信度检测）
        model.eval()  # 设置为推理模式
        return model

    def load_crnn(self, model_path, device):
        # 加载模型
        model = CRNN(len(characters) + 1).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model

    def init_ui(self):
        self.setWindowTitle("Automatic-license-plate-recognition-system")
        self.setGeometry(100, 100, 400, 650)

        # 创建主控件和布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # 顶部按钮
        self.btn_open = QPushButton("选择图片")
        self.btn_open.clicked.connect(self.open_image)
        main_layout.addWidget(self.btn_open, alignment=Qt.AlignTop)

        # 添加标签
        self.label = QLabel("请选择图片!")
        main_layout.addWidget(self.label, alignment=Qt.AlignCenter)

        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(scroll_area)

        # 图片显示区域
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(1, 1)

        # 将标签放入滚动区域
        scroll_area.setWidget(self.image_label)

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片文件",
            "",
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.gif)"
        )
        if file_path:
            img, plates = self.plate_positioning(file_path)
            results = self.auto_recognition(plates)
            if not img is None:
                scaled_width = int(img.width() * 0.5)
                scaled_height = int(img.height() * 0.5)

                scaled_pixmap = img.scaled(
                    scaled_width,
                    scaled_height,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.label.setText(f'{results}')
                self.image_label.setPixmap(scaled_pixmap)
                self.image_label.setAlignment(Qt.AlignCenter)
            else:
                self.image_label.setText("无法加载图片")

    def plate_positioning(self, img_path):
        img = cv2.imread(img_path)
        # 图像预处理
        results = self.yolo(img, 640)
        # 提取检测结果
        detections = results.xyxy[0].cpu()
        # 遍历所有检测到的车牌裁剪车牌区域
        plates = []
        for i, det in enumerate(detections):
            x1, y1, x2, y2, conf, cls_id = det
            if conf < self.yolo.conf:  # 置信度低于阈值则跳过
                continue
            plates.append((i, img[int(y1):int(y2), int(x1):int(x2)]))
            # 在原始图像上绘制边界框（可视化）
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(img, f"Id:{i} cong:{conf:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return QPixmap.fromImage(self.numpy2qimage(img)), plates

    def auto_recognition(self, plates):
        if plates:
            results = []
            for i, plate in plates:
                img = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (256, 32))
                img = (img / 255.0 - 0.5) / 0.5  # 归一化
                img_tensor = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0).to('cpu')
                with torch.no_grad():
                    logits = self.crnn(img_tensor)
                results.append((i, self.decoder.decode(logits)))
            return results

    def numpy2qimage(self, img):
        height, width, channel = img.shape
        bytes_per_line = 3 * width
        return QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
