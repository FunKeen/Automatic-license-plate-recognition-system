import os
import random
import re
from multiprocessing import Process

import cv2


def select_images(image_dir, output_dir, pre_images=0.05):
    global totle

    # 获取所有图片
    base_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    sum_images = len(base_files)
    num_selected = int(pre_images * sum_images)

    # 打印调试信息
    print(f'{len(base_files)} images in: {image_dir}')

    # 随机选择图片
    selected_files = random.sample(base_files, num_selected)
    # 复制选中的图片
    for index, image_name in enumerate(selected_files):
        split_image_name = image_name.split('-')
        if len(split_image_name) == 1:
            continue
        x1, y1, x2, y2 = map(int, re.split('[_&]', split_image_name[2]))
        # 裁剪图片
        image = cv2.imread(os.path.join(image_dir, image_name))
        assert image is not None, f"图像 {image_name} 读取失败！"
        plate_region = image[int(y1):int(y2), int(x1):int(x2)]
        # 保存图片
        if index < 0.8 * num_selected:
            cv2.imwrite(os.path.join(output_dir, 'train/images', image_name), plate_region)
        elif index < 0.9 * num_selected:
            cv2.imwrite(os.path.join(output_dir, 'val/images', image_name), plate_region)
        else:
            cv2.imwrite(os.path.join(output_dir, 'test/images', image_name), plate_region)

    print(f'{len(selected_files)} to: {output_dir}')


def mytask(file_path, out_dir, pre):
    print(f"进程 {file_path} (PID: {os.getpid()}) 执行")
    print(file_path, out_dir, pre)
    select_images(file_path, out_dir, pre)


if __name__ == '__main__':
    # 定义路径和参数
    image_dir = r'C:/Users/Keen/tempfile/CCPD2019'  # CCPD数据集位置
    output_dir = r'../mydataset_crnn'  # 随机选择的图片存放位置
    pre_images = 0.5  # 选择图片占比

    # 创建输出目录
    os.makedirs(os.path.join(output_dir, 'train/images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val/images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test/images'), exist_ok=True)

    # 调用函数
    files = ['ccpd_base', 'ccpd_blur', 'ccpd_challenge', 'ccpd_db', 'ccpd_fn', 'ccpd_green', 'ccpd_rotate',
             'ccpd_tilt', 'ccpd_weather']
    processes = []
    for file in files:
        p = Process(target=mytask, args=(os.path.join(image_dir, file), output_dir, pre_images))
        processes.append(p)
        p.start()  # 立即启动进程
    for p in processes:
        p.join()
