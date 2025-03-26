import cv2
from PySide6.QtWidgets import (QMainWindow, QPushButton,
                               QLabel, QFileDialog, QWidget, QVBoxLayout,
                               QScrollArea)
from PySide6.QtCore import Qt

from ModelYOLO import ModelYOLO

Modelyolo = ModelYOLO('C:/Users/Keen/PycharmProjects/Alprs/yolov5/runs/train/exp2/weights/best.pt')


class Alprs(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

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
            img, plates = self.Auto_recognition(file_path)
            if not img.isNull():
                scaled_width = int(img.width() * 0.5)
                scaled_height = int(img.height() * 0.5)

                scaled_pixmap = img.scaled(
                    scaled_width,
                    scaled_height,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.label.setText(f'{plates}')
                self.image_label.setPixmap(scaled_pixmap)
                self.image_label.setAlignment(Qt.AlignCenter)
            else:
                self.image_label.setText("无法加载图片")

    def Auto_recognition(self, img_path):
        return Modelyolo(cv2.imread(img_path))
