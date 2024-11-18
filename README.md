原始KITTI数据集


链接：https://pan.baidu.com/s/165m3yFxvHowZSyxAgZMjqQ?pwd=h1ar 
提取码：h1ar

```
data_object_image_2/
├── training/
│   └── image_2/
│   │   ├── 000000.png
│   │   ├── 000001.png
│   │   └── ...
└── testing/
│   └── image_2/
│   │   ├── 000000.png
│   │   ├── 000001.png
│   │   └── ...
```

```
training/
├── label/
│   ├── 0000000.txt
│   ├── 0000001.txt
│   └── ...
```

```
conda upgrade conda
```

```
conda create -n CarDetection python=3.8
```

```
conda activate CarDetection
```

```
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```

```
python
```

```
import torch
print(torch.cuda.is_available())
print(torch.backends.cudnn.is_available())
print(torch.version.cuda)
print(torch.backends.cudnn.version())
```

**关闭VPN**

```
pip install -i https://mirrors.aliyun.com/pypi/simple/ tqdm
```

```
pip install -i https://mirrors.aliyun.com/pypi/simple/ opencv-python
```

```
pip install -i https://mirrors.aliyun.com/pypi/simple/ matplotlib
```

```
pip install -i https://mirrors.aliyun.com/pypi/simple/ pandas
```


### 将KITTI格式的标签转为YOLO格式的标签

```python
import os
import cv2
import glob
import shutil
from tqdm import tqdm


# 类别映射
dic = {'Car': 0, 'Van': 0, 'Truck': 0,
       'Tram': 1, 'Pedestrian': 1, 'Person_sitting': 1, 'Cyclist': 1, 'Misc': 1, 'DontCare': 1}


def change_format():
    # 路径配置
    images_path = r"E:/DataSets/KITTI/Object/data_object_image_2/training/image_2/*"  # kitti图像数据
    labels_path = r"E:/DataSets/KITTI/Object/training/label_2/"  # kitti标签数据
    filename_list = glob.glob(images_path)
    images_save_path = r"E:/DataSets/KITTI/Object/data/images/"  # 图像文件保存路径
    labels_save_path = r"E:/DataSets/KITTI/Object/data/labels/"  # yolo格式标签文件保存路径

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

```

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
        "name": 'DontCare',
    }
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




