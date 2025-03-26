import os
import random
from shutil import copyfile


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
    for index, file in enumerate(selected_files):
        if index < 0.8 * num_selected:
            copyfile(
                os.path.join(image_dir, file),
                os.path.join(output_dir, 'images/train', file)
            )
        elif index < 0.9 * num_selected:
            copyfile(
                os.path.join(image_dir, file),
                os.path.join(output_dir, 'images/val', file)
            )
        else:
            copyfile(
                os.path.join(image_dir, file),
                os.path.join(output_dir, 'images/test', file)
            )

    print(f'{len(selected_files)} to: {output_dir}')
    totle += len(selected_files)


if __name__ == '__main__':
    totle = 0
    # 定义路径和参数
    image_dir = r'C:/Users/Keen/tempfile/CCPD2019'  # CCPD数据集位置
    output_dir = r'../mydataset_yolo'  # 随机选择的图片存放位置
    pre_images = 0.2  # 选择图片占比

    # 创建输出目录
    os.makedirs(os.path.join(output_dir, 'images/train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images/val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images/test'), exist_ok=True)

    # 调用函数
    select_images(os.path.join(image_dir, 'ccpd_base'), output_dir, pre_images)
    select_images(os.path.join(image_dir, 'ccpd_blur'), output_dir, pre_images)
    select_images(os.path.join(image_dir, 'ccpd_challenge'), output_dir, pre_images)
    select_images(os.path.join(image_dir, 'ccpd_db'), output_dir, pre_images)
    select_images(os.path.join(image_dir, 'ccpd_fn'), output_dir, pre_images)
    select_images(os.path.join(image_dir, 'ccpd_green'), output_dir, 0.05)
    select_images(os.path.join(image_dir, 'ccpd_np'), output_dir, pre_images)
    select_images(os.path.join(image_dir, 'ccpd_rotate'), output_dir, pre_images)
    select_images(os.path.join(image_dir, 'ccpd_tilt'), output_dir, pre_images)
    select_images(os.path.join(image_dir, 'ccpd_weather'), output_dir, pre_images)

    print(totle)
