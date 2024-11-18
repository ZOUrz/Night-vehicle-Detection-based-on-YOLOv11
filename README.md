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
print(torch.cuda_version)
print(torch.backends.cudnn.version())
```


