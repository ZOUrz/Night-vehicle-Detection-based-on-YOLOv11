


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

  - 执行 [1_TransferLabels_kitti2yolo.py](DataPreprocess/1_TransferLabels_kitti2yolo.py), 将 KITTI 格式的标签转为 YOLO 格式的标签
 
  - 转换标签后的数据集文件存放格式如下:
 
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
    │   ├── data_object_label_2/
    │   │   └── training/
    │   │   │   └── label_2/
    │   │   │   │   ├── 000000.txt
    │   │   │   │   ├── 000001.txt
    │   │   │   │   └── ...  # 总共 7481 个文件
    │   └── data/
    │   │   ├── images/
    │   │   │   ├── 000000.png
    │   │   │   ├── 000001.png
    │   │   │   │   └── ...  # 总共7841个文件
    │   │   └── labels/
    │   │   │   ├── 000000.txt
    │   │   │   ├── 000001.txt
    │   │   │   └── ...  # 总共7841个文件
    ```


- ### 3.2 划分训练集和测试集

  - 执行 [2_SplitDataset.py](DataPreprocess/2_SplitDataset.py), 将数据集划分为训练集和验证集
 
  - 划分后的数据集文件存放格式如下:
 
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
    │   ├── data_object_label_2/
    │   │   └── training/
    │   │   │   └── label_2/
    │   │   │   │   ├── 000000.txt
    │   │   │   │   ├── 000001.txt
    │   │   │   │   └── ...  # 总共 7481 个文件
    │   └── data/
    │   │   ├── images/
    │   │   │   ├── train/
    │   │   │   │   ├── 000002.png
    │   │   │   │   ├── 000003.png
    │   │   │   │   └── ...  # 总共5985个文件
    │   │   │   └── val/
    │   │   │   │   ├── 000000.png
    │   │   │   │   ├── 000001.png
    │   │   │   │   └── ...  # 总共1496个文件
    │   │   └── labels/
    │   │   │   ├── train/
    │   │   │   │   ├── 000002.txt
    │   │   │   │   ├── 000003.txt
    │   │   │   │   └── ...  # 总共5985个文件
    │   │   │   └── val/
    │   │   │   │   ├── 000000.txt
    │   │   │   │   ├── 000001.txt
    │   │   │   │   └── ...  # 总共1496个文件
    ```


- ### 3.3 标签格式转换 YOLO -> COCO

  - 执行 [3_TransferLabels_yolo2coco.py](DataPreprocess/3_TransferLabels_yolo2coco.py), 将 YOLO 格式的标签转换成 COCO 格式的 .json 文件
 
  - 转换标签后的数据集文件存放格式如下:

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
    │   ├── data_object_label_2/
    │   │   └── training/
    │   │   │   └── label_2/
    │   │   │   │   ├── 000000.txt
    │   │   │   │   ├── 000001.txt
    │   │   │   │   └── ...  # 总共 7481 个文件
    │   └── data/
    │   │   ├── images/
    │   │   │   ├── train/
    │   │   │   │   ├── 000002.png
    │   │   │   │   ├── 000003.png
    │   │   │   │   └── ...  # 总共5985个文件
    │   │   │   └── val/
    │   │   │   │   ├── 000000.png
    │   │   │   │   ├── 000001.png
    │   │   │   │   └── ...  # 总共1496个文件
    │   │   ├── labels/
    │   │   │   ├── train/
    │   │   │   │   ├── 000002.txt
    │   │   │   │   ├── 000003.txt
    │   │   │   │   └── ...  # 总共5985个文件
    │   │   │   └── val/
    │   │   │   │   ├── 000000.txt
    │   │   │   │   ├── 000001.txt
    │   │   │   │   └── ...  # 总共1496个文件
    │   │   └── annotations/
    │   │   │   ├── instances_train.json
    │   │   │   └── instances_val.json
    ```


## 4. 使用各个版本的 YOLO 进行训练和评估


- ### 4.1 安装 Ultralytics

  - 在终端输入如下代码:

    ```
    https://github.com/ultralytics/ultralytics.git
    cd ultralytics-main
    pip install -e .
    ```


- ### 4.2 编写 kitti.yaml 文件

  ```yaml
  # Ultralytics YOLO 🚀, AGPL-3.0 license
  
  path: /home/zourz/work/Dataset/CarDetection/KITTI/Object/data # 修改为包含图片和标签的父文件夹
  train: images/train # train images (relative to 'path')
  val: images/val # val images (relative to 'path')
  
  nc: 5      # 修改为类别数量
  # Classes
  names:
    0: vehicle
    1: tram
    2: pedestrian
    3: cyclist
    4: misc
  
  ```


- ### 4.3 修改模型的种类数量

  - 以 yolov8.yaml 为例, 主要修改其种类的数量 nc: 5 # number of classes, 完整的代码如下所示:

    ```yaml
    # Ultralytics YOLO 🚀, AGPL-3.0 license
    # YOLOv8 object detection model with P3-P5 outputs. For Usage examples see https://docs.ultralytics.com/tasks/detect
    
    # Parameters
    nc: 5 # number of classes
    scales: # model compound scaling constants, i.e. 'model=yolov8n.yaml' will call yolov8.yaml with scale 'n'
      # [depth, width, max_channels]
      n: [0.33, 0.25, 1024] # YOLOv8n summary: 225 layers,  3157200 parameters,  3157184 gradients,   8.9 GFLOPs
      s: [0.33, 0.50, 1024] # YOLOv8s summary: 225 layers, 11166560 parameters, 11166544 gradients,  28.8 GFLOPs
      m: [0.67, 0.75, 768] # YOLOv8m summary: 295 layers, 25902640 parameters, 25902624 gradients,  79.3 GFLOPs
      l: [1.00, 1.00, 512] # YOLOv8l summary: 365 layers, 43691520 parameters, 43691504 gradients, 165.7 GFLOPs
      x: [1.00, 1.25, 512] # YOLOv8x summary: 365 layers, 68229648 parameters, 68229632 gradients, 258.5 GFLOPs
    
    # YOLOv8.0n backbone
    backbone:
      # [from, repeats, module, args]
      - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
      - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
      - [-1, 3, C2f, [128, True]]
      - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
      - [-1, 6, C2f, [256, True]]
      - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
      - [-1, 6, C2f, [512, True]]
      - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
      - [-1, 3, C2f, [1024, True]]
      - [-1, 1, SPPF, [1024, 5]] # 9
    
    # YOLOv8.0n head
    head:
      - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
      - [[-1, 6], 1, Concat, [1]] # cat backbone P4
      - [-1, 3, C2f, [512]] # 12
    
      - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
      - [[-1, 4], 1, Concat, [1]] # cat backbone P3
      - [-1, 3, C2f, [256]] # 15 (P3/8-small)
    
      - [-1, 1, Conv, [256, 3, 2]]
      - [[-1, 12], 1, Concat, [1]] # cat head P4
      - [-1, 3, C2f, [512]] # 18 (P4/16-medium)
    
      - [-1, 1, Conv, [512, 3, 2]]
      - [[-1, 9], 1, Concat, [1]] # cat head P5
      - [-1, 3, C2f, [1024]] # 21 (P5/32-large)
    
      - [[15, 18, 21], 1, Detect, [nc]] # Detect(P3, P4, P5)
    
    ```


- ### 4.4 开始训练

  ```
  yolo task=detect mode=train data=kitti.yaml model=yolov8s.yaml epochs=200 imgsz=640 device=[0,1] project=/home/ai/student/ZOUrz/CarDetection/ultralytics-main/runs/yolov8s batch=128 patience=20 name=default_hyp_200epochs
  ```

