import os
import random
from multiprocessing import Process
from shutil import copyfile

provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂",
             "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']


def select_images(image_dir, output_dir, pre_images=0.05):
    global totle

    os.makedirs(output_dir, exist_ok=True)
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
        # 读取车牌信息
        split_image_name = image_name.split('-')
        plate_code = list(map(int, split_image_name[4].split('_')))
        plate = '' + provinces[plate_code[0]]
        plate = plate + alphabets[plate_code[1]]
        for index in plate_code[2:]:
            plate = plate + ads[index]
        plate+='.jpg'
        # 复制图片
        copyfile(
            os.path.join(image_dir, image_name),
            os.path.join(output_dir, plate)
        )

    print(f'{len(selected_files)} to: {output_dir}')


def mytask(file_path, out_dir, pre):
    print(f"进程 {file_path} (PID: {os.getpid()}) 执行")
    print(file_path, out_dir, pre)
    select_images(file_path, out_dir, pre)


if __name__ == '__main__':
    # 定义路径和参数
    image_dir = r'C:/Users/Keen/tempfile/CCPD2019'  # CCPD数据集位置
    output_dir = r'../testimages'  # 随机选择的图片存放位置
    pre_images = 0.01  # 选择图片占比

    # 调用函数
    files = ['ccpd_base', 'ccpd_blur', 'ccpd_challenge', 'ccpd_db', 'ccpd_fn', 'ccpd_green', 'ccpd_rotate',
             'ccpd_tilt', 'ccpd_weather']
    processes = []
    for file in files:
        p = Process(target=mytask, args=(os.path.join(image_dir, file), os.path.join(output_dir, file), pre_images))
        processes.append(p)
        p.start()  # 立即启动进程
    for p in processes:
        p.join()
