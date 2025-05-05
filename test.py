import os
import cv2
from ModelYOLO import ModelYOLO
import time

import warnings

warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.autocast.*")

yolo = ModelYOLO('yolov5/runs/train/exp2/weights/best.pt')

files = ['ccpd_base', 'ccpd_blur', 'ccpd_challenge', 'ccpd_db', 'ccpd_fn', 'ccpd_green', 'ccpd_rotate',
         'ccpd_tilt', 'ccpd_weather']


def fun(path):
    # 获取所有图片
    images_name = [f for f in os.listdir(path) if f.endswith('.jpg')]
    sum = len(images_name)
    count = 0
    # 遍历图片 获取图片信息
    start_time = time.time()  # 记录开始时间
    for image_name in images_name:
        img = cv2.imread(os.path.join(path, image_name))
        _, pre_plate, _ = yolo(img)
        # print(pre_plate[0], image_name[:-4])
        if pre_plate and pre_plate[0] == image_name[:-4]:
            count += 1
        # cv2.imwrite('show.jpg', img)
    end_time = time.time()  # 记录开始时间
    return count / sum, sum / (end_time - start_time)


acp = []
afps = []
# 调用函数
for file in files:
    cp, fps = fun(os.path.join('testimages', file))
    acp.append(cp)
    afps.append(fps)
for i in range(len(files)):
    print(f'场景：{files[i]} 准确率：{acp[i]:.4f}  FPS：{afps[i]:.4f}')
print(f'平均：准确率：{(sum(acp) / len(files)):.4f}  FPS：{(sum(afps) // len(files))}')
