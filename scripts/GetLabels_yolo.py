import os
import re
from multiprocessing import Process

import cv2


def get_labels(image_dir):
    # 获取所有图片
    images_name = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    print(f"{image_dir} start")
    # 遍历图片 获取图片信息

    for image_name in images_name:
        split_image_name = image_name.split('-')
        txt_name = os.path.splitext(image_name)[0] + '.txt'
        with open(os.path.join(image_dir.replace('images', 'labels'), txt_name), 'w', encoding='utf-8') as f:
            if len(split_image_name) == 7:
                x1, y1, x2, y2 = map(int, re.split('[_&]', split_image_name[2]))
                # 归一化处理
                img = cv2.imread(os.path.join(image_dir, image_name))
                h, w, _ = img.shape
                cx = (x1 + x2) / (2 * w)
                cy = (y1 + y2) / (2 * h)
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                # 保存
                if len(split_image_name[4].split('_')) == 7:
                    f.write(f"1 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                elif len(split_image_name[4].split('_')) == 8:
                    f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            elif len(split_image_name) == 1:
                f.write("\n")
    print(f"{image_dir} is completed")


def mytask(file_path):
    print(f"进程 {file_path} (PID: {os.getpid()}) 执行")
    print(file_path)
    get_labels(file_path)


if __name__ == '__main__':
    # 定义路径参数
    image_dir = r'../mydataset_yolo/images'

    # 创建文件
    os.makedirs(os.path.join(image_dir.replace('images', 'labels'), 'train'), exist_ok=True)
    os.makedirs(os.path.join(image_dir.replace('images', 'labels'), 'val'), exist_ok=True)
    os.makedirs(os.path.join(image_dir.replace('images', 'labels'), 'test'), exist_ok=True)

    # 调用函数
    files = ['train', 'val', 'test']
    processes = []
    for file in files:
        p = Process(target=mytask, args=(os.path.join(image_dir, file),))
        processes.append(p)
        p.start()  # 立即启动进程
    for p in processes:
        p.join()
