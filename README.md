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






