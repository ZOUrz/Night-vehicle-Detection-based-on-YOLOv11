


## 1. 构建深度学习环境

  - 在终端输入如下代码, 创建一个名为 CarDetection 的虚拟环境, Python 版本为 3.8

    ```
    conda upgrade conda
    conda create -n CarDetection python=3.8
    ```

  - 安装带有 CUDA 版本的 Pytorch 2.0.1

    ```
    conda activate CarDetection
    conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
    ```

  - 验证是否安装成功

    - 在终端输入:
   
      ```
      python
      ```

    - 在 Python 中输入:

      ```
      import torch
      print(torch.cuda.is_available())
      print(torch.backends.cudnn.is_available())
      print(torch.version.cuda)
      print(torch.backends.cudnn.version())
      ```

  - 安装其他库

    ```
    pip install -i https://mirrors.aliyun.com/pypi/simple/ tqdm
    pip install -i https://mirrors.aliyun.com/pypi/simple/ opencv-python
    pip install -i https://mirrors.aliyun.com/pypi/simple/ matplotlib
    pip install -i https://mirrors.aliyun.com/pypi/simple/ pandas
    ```


## 2. KITTI 目标检测数据集下载

- KITTI 官网提供的链接国内无法下载, 且网上的资源也大部分实效, 因为我将数据集重新上传到百度网盘, 供大家下载

  - 百度网盘链接：https://pan.baidu.com/s/165m3yFxvHowZSyxAgZMjqQ?pwd=h1ar 提取码：h1ar

- 使用说明

  - 每个文件的内容被分成多个压缩包, 对于不同内容的压缩包, 只需要解压对应的 xxx.zip 文件

  - Ubuntu 系统最好使用 `7z` 进行分卷解压缩, 如果没有安装就使用命令 `sudo apt install p7zip-full` 下载安装

  - 在压缩包所在文件夹内打开终端, 输入命令 `7z x data_object_image_2.zip` 即可进行解压
 
- 解压后的文件存放格式如下:

  ```
  KITTI/
  └── Object
  │   ├── data_object_image_2/
  │   │   ├── testing/
  │   │   │   └── image_2/
  │   │   │   │   ├── 000000.png
  │   │   │   │   ├── 000001.png
  │   │   │   │   └── ...  # 总共 7518 个文件
  │   │   └── training/
  │   │   │   └── image_2/
  │   │   │   │   ├── 000000.png
  │   │   │   │   ├── 000001.png
  │   │   │   │   └── ...  # 总共 7481 个文件
  │   └── data_object_label_2/
  │   │   └── training/
  │   │   │   └── label_2/
  │   │   │   │   ├── 000000.txt
  │   │   │   │   ├── 000001.txt
  │   │   │   │   └── ...  # 总共 7481 个文件
  ```
       

## 3. KITTI 数据集预处理


- ### 3.1 标签格式转换 KITTI -> YOLO

- 将KITTI格式的标签转为YOLO格式的标签

```
data/
├── images/
│   ├── 000000.png
│   ├── 000001.png
│   │   └── ...  # 总共7841个文件
└── labels/
│   ├── 000000.txt
│   ├── 000001.txt
│   └── ...  # 总共7841个文件
```


### 划分训练集和测试集

```python
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
    path = r"E:/DataSets/KITTI/Object/data"
    mvfile(path)

```

```
data/
├── images/
│   ├── train/
│   │   ├── 000000.png
│   │   ├── 000001.png
│   │   └── ...  # 总共5985个文件
│   └── val/
│   │   ├── 000006.png
│   │   ├── 000023.png
│   │   └── ...  # 总共1496个文件
└── labels/
│   └── train/
│   │   ├── 000000.txt
│   │   ├── 000001.txt
│   │   └── ...  # 总共5985个文件
│   └── val/
│   │   ├── 000006.txt
│   │   ├── 000023.txt
│   │   └── ...  # 总共1496个文件
```


### 3. 将YOLO格式的标签转换成COCO格式的.json文件

```python
import os
import cv2
import json


# info, license, categories 结构初始化
# 在train.json, val.json, 里面信息是一致的

# info, license暂时用不到
info = {
    "year": 2024,
    "version": '1.0',
    "date_created": 2024 - 11 - 18
}

licenses = {
    "id": 1,
    "name": "null",
    "url": "null",
}

# 自己的标签类别, 跟yolo的数据集类别要对应好
categories = [
    {
        "id": 0,
        "name": 'Car',
    },
    {
        "id": 1,
        "name": 'Van',
    },
    {
        "id": 2,
        "name": 'Truck',
    },
    {
        "id": 3,
        "name": 'Tram',
    },
    {
        "id": 4,
        "name": 'Person',
    },
    {
        "id": 5,
        "name": 'DontCare',
    },
]


# 初始化train, val 数据字典
# info licenses categories 在 train 和 val 里面都是一致的；
train_data = {'info': info, 'licenses': licenses, 'categories': categories, 'images': [], 'annotations': []}
val_data = {'info': info, 'licenses': licenses, 'categories': categories, 'images': [], 'annotations': []}


# image_path 对应的图像路径，比如images/train；
# label_path 对应的label路径，比如labels/train 跟images要对应；
def yolo_covert_coco_format(image_path, label_path):
    images = []
    annotations = []
    for index, img_file in enumerate(os.listdir(image_path)):
        if img_file.endswith('.png'):
            image_info = {}
            img = cv2.imread(os.path.join(image_path, img_file))
            height, width, channel = img.shape
            image_info['id'] = index
            image_info['file_name'] = img_file
            image_info['width'], image_info['height'] = width, height
        else:
            continue
        if image_info != {}:
            images.append(image_info)
        # 处理 label 信息-------
        label_file = os.path.join(label_path, img_file.replace('.png', '.txt'))
        with open(label_file, 'r') as f:
            for idx, line in enumerate(f.readlines()):
                info_annotation = {}
                class_num, xs, ys, ws, hs = line.strip().split(' ')
                class_id, xc, yc, w, h = int(class_num), float(xs), float(ys), float(ws), float(hs)
                xmin = (xc - w / 2) * width
                ymin = (yc - h / 2) * height
                xmax = (xc + w / 2) * width
                ymax = (yc + h / 2) * height
                bbox_w = int(width * w)
                bbox_h = int(height * h)

                info_annotation["category_id"] = class_id  # 类别的id
                info_annotation['bbox'] = [xmin, ymin, bbox_w, bbox_h]  # bbox的坐标
                info_annotation['area'] = bbox_h * bbox_w  # area
                info_annotation['image_id'] = index  # bbox的id
                info_annotation['id'] = index * 100 + idx  # bbox的id
                info_annotation['segmentation'] = [[xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]]  # 四个点的坐标
                info_annotation['iscrowd'] = 0  # 单例
                annotations.append(info_annotation)
    return images, annotations


# key == train, val
# 对应要生成的json文件，比如instances_train, instances_val.json
def gen_json_file(yolov8_data_path, coco_format_path, key):
    # json path
    json_path = os.path.join(coco_format_path, f'annotations/instances_{key}.json')
    if not os.path.exists(os.path.dirname(json_path)):
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
    data_path = os.path.join(yolov8_data_path, f'images/{key}')
    label_path = os.path.join(yolov8_data_path, f'labels/{key}')
    images, anns = yolo_covert_coco_format(data_path, label_path)
    if key == 'train':
        train_data['images'] = images
        train_data['annotations'] = anns
        with open(json_path, 'w') as f:
            json.dump(train_data, f, indent=2)
    elif key == 'val':
        val_data['images'] = images
        val_data['annotations'] = anns
        with open(json_path, 'w') as f:
            json.dump(val_data, f, indent=2)
    else:
        print(f'key is {key}')
    print(f'generate {key} json success!')
    return


if __name__ == '__main__':
    data_path = r"E:/DataSets/KITTI/Object/data"
    coco_format_path = r"E:/DataSets/KITTI/Object/data"
    gen_json_file(data_path, coco_format_path, key='train')
    gen_json_file(data_path, coco_format_path, key='val')

```

```
data/
├── images/
│   ├── train/
│   │   ├── 000000.png
│   │   ├── 000001.png
│   │   └── ...  # 总共5985个文件
│   └── val/
│   │   ├── 000006.png
│   │   ├── 000023.png
│   │   └── ...  # 总共1496个文件
├── labels/
│   └── train/
│   │   ├── 000000.txt
│   │   ├── 000001.txt
│   │   └── ...  # 总共5985个文件
│   └── val/
│   │   ├── 000006.txt
│   │   ├── 000023.txt
│   │   └── ...  # 总共1496个文件
└── annotations/
│   ├── instances_train.json
│   └── instances_val.json
```

```
cd ultralytics-main
```

```
pip install -e .
```

```yaml
# Ultralytics YOLO 🚀, AGPL-3.0 license

path: E:/DataSets/KITTI/Object/data # 修改为包含图片和标签的父文件夹
train: images/train # train images (relative to 'path')
val: images/val # val images (relative to 'path')

nc: 6      # 修改为类别数量
# Classes
names:
  0: car
  1: van
  2: truck
  3: tram
  4: person
  5: dontcare

```

