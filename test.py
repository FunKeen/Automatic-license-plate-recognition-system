import torch
import cv2

from ModelYOLO import ModelYOLO

yolo = ModelYOLO('yolov5/runs/train/exp2/weights/best.pt')
img = cv2.imread('C:/Users/Keen/PycharmProjects/Alprs/testimages/1.jpg')
img, plates = yolo(img)

cv2.imwrite('show.jpg', img)
print(plates)
