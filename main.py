import torch
import cv2

from crnn.CRNNInferencer import CRNNInferencer
from YOLOInferencer import YOLOInferencer

yolo = YOLOInferencer('yolov5/runs/train/exp2/weights/best.pt', 'cpu')
crnn = CRNNInferencer('crnn/crnn.pth', 'cpu')
res = crnn(yolo(
    'C:/Users/Keen/tempfile/CCPD2019/ccpd_base/01-90_88-196&604_384&663-386&662_213&658_204&598_377&602-0_0_20_4_31_29_32-112-27.jpg'))
for i in res:
    print(i)
