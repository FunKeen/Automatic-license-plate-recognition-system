import os
import re
import cv2

provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂",
             "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']


def get_labels(image_dir):
    # 创建文件
    txt_name = os.path.join(image_dir, 'labels.txt')
    with open(txt_name, 'w', encoding='utf-8') as f:
        # 获取所有图片
        images_name = [f for f in os.listdir(os.path.join(image_dir, 'images')) if f.endswith('.jpg')]
        print(f"{image_dir} start")

        # 遍历图片 获取图片信息
        for image_name in images_name:
            print(image_name)
            split_image_name = image_name.split('-')
            plate_code = list(map(int, split_image_name[4].split('_')))
            plate = '' + provinces[plate_code[0]]
            plate = plate + alphabets[plate_code[1]]
            for index in plate_code[2:]:
                plate = plate + ads[index]
            f.write(f"{image_name} {plate}\n")
    # print(f"{image_dir} is completed")


if __name__ == '__main__':
    # 定义路径参数
    image_dir = r'../testdataset_crnn'

    # 调用函数
    get_labels(os.path.join(image_dir, 'test'))
    get_labels(os.path.join(image_dir, 'train'))
    get_labels(os.path.join(image_dir, 'val'))
