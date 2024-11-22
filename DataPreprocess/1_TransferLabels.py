import os
import cv2
import glob
import shutil
from tqdm import tqdm


# 类别映射
dic = {'Car': 0, 'Van': 0, 'Truck': 0,
            'Tram': 1, 'Pedestrian': 2, 'Person_sitting': 2, 'Cyclist': 3, 'Misc': 4}


def change_format():
    # 路径配置
    images_path = "/home/zourz/work/Dataset/CarDetection/KITTI/Object/data_object_image_2/training/image_2/*"  # kitti 图像数据
    labels_path = "/home/zourz/work/Dataset/CarDetection/KITTI/Object/data_object_label_2/training/label_2/"  # kitti 标签数据
    filename_list = glob.glob(images_path)
    images_save_path = "/home/zourz/work/Dataset/CarDetection/KITTI/Object/data/images/"  # 图像文件保存路径
    labels_save_path = "/home/zourz/work/Dataset/CarDetection/KITTI/Object/data/labels/"  # yolo 格式标签文件保存路径

    # 如果保存路径不存在则创建
    if not os.path.exists(images_save_path):
        os.makedirs(images_save_path)
    if not os.path.exists(labels_save_path):
        os.makedirs(labels_save_path)

    # 检查是否找到图像
    if not filename_list:
        print("No images found in the specified path.")
        return

    # 统计类别的处理次数
    category_count = {key: 0 for key in dic.keys()}
    ignored_count = 0
    processed_images = 0

    # 遍历图像文件
    for img_name in tqdm(filename_list, desc='Processing'):
        image_name = os.path.basename(img_name).split('.')[0]  # 获取图片名称（无扩展名）
        label_file = os.path.join(labels_path, image_name + '.txt')  # 找到对应的标签
        label_save_path = os.path.join(labels_save_path, image_name + '.txt')  # 修改后标签保存路径

        # 检查标签文件是否存在
        if not os.path.exists(label_file):
            print(f"Label file not found: {label_file}")
            continue

        # 读取图像
        img = cv2.imread(img_name)
        if img is None:
            print(f"Failed to load image: {img_name}")
            continue

        h, w, _ = img.shape
        dw = 1.0 / w
        dh = 1.0 / h  # 归一化比例

        # 读取标签
        with open(label_file, 'r') as f:
            labels = f.readlines()

        # 清空标签文件内容以避免追加重复数据
        with open(label_save_path, 'w') as w:
            for label in labels:
                label = label.split(' ')

                # 检查标签长度是否正确
                if len(label) < 8:
                    print(f"Incomplete label in file {label_file}: {label}")
                    continue

                classname = label[0]
                if classname not in dic:
                    ignored_count += 1  # 统计被忽略的标签
                    continue

                # 获取位置信息并进行归一化
                x1, y1, x2, y2 = map(float, label[4:8])
                bx = ((x1 + x2) / 2.0) * dw
                by = ((y1 + y2) / 2.0) * dh
                bw = (x2 - x1) * dw
                bh = (y2 - y1) * dh

                # 保留6位小数
                bx = round(bx, 6)
                by = round(by, 6)
                bw = round(bw, 6)
                bh = round(bh, 6)

                # 获取类别索引
                classindex = dic[classname]
                category_count[classname] += 1  # 统计该类别的标签数量

                # 写入转换后的标签
                w.write(f'{classindex} {bx} {by} {bw} {bh}\n')

        # 将图像文件复制到指定路径
        shutil.copy(img_name, images_save_path)

        processed_images += 1

    # 打印统计信息
    print(f'Done processing {processed_images} images!')
    print(f'Ignored {ignored_count} labels not in the dictionary.')
    for classname, count in category_count.items():
        print(f'{classname}: {count} labels processed.')


# 调用函数
change_format()
