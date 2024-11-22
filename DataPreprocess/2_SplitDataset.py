import os
import random
import shutil


def mvfile(path):
    # 标签文件列表
    labels_list = os.listdir(path + "/labels/")
    # 标签文件路径
    labels_path = path + "/labels/"
    # 图像文件路径
    images_path = path + "/images/"
    # 保存标签文件的路径
    train_labels_path = path + "/labels/train/"
    val_labels_path = path + "/labels/val/"
    # 保存图像文件的路径
    train_images_path = path + "/images/train/"
    val_images_path = path + "/images/val/"

    if not os.path.exists(train_labels_path):
        os.makedirs(train_labels_path)
    if not os.path.exists(val_labels_path):
        os.makedirs(val_labels_path)
    if not os.path.exists(train_images_path):
        os.makedirs(train_images_path)
    if not os.path.exists(val_images_path):
        os.makedirs(val_images_path)

    val_labels_list = random.sample(labels_list, 1496)
    print("------ Split val ------")
    for label in val_labels_list:
        val_label_file = labels_path + label
        print(val_label_file)
        shutil.move(val_label_file , val_labels_path)
        val_images_file = images_path + label.replace("txt","png")
        print(val_images_file)
        shutil.move(val_images_file, val_images_path)

    val_labels_list_set = set(val_labels_list)
    train_labels_list = [item for item in labels_list if item not in val_labels_list_set]
    print("------ Split train ------")
    for label in train_labels_list:
        train_label_file = labels_path + label
        print(train_label_file)
        shutil.move(train_label_file , train_labels_path)
        train_images_file = images_path + label.replace("txt","png")
        print(train_images_file)
        shutil.move(train_images_file, train_images_path)


if __name__ == '__main__':
    path = "/home/zourz/work/Dataset/CarDetection/KITTI/Object/data"
    mvfile(path)
