import os
import cv2
import json
from tqdm import tqdm


# info, license, categories 结构初始化
# 在train.json, val.json, 里面信息是一致的

# info, license 暂时用不到
info = {
    "year": 2024,
    "version": '1.0',
    "date_created": 2024 - 11 - 22
}

licenses = {
    "id": 1,
    "name": "null",
    "url": "null",
}

# 自己的标签类别, 跟 yolo 的数据集类别要对应好
categories = [
    {
        "id": 0,
        "name": 'Vehicle',
    },
    {
        "id": 1,
        "name": 'Tram',
    },
    {
        "id": 2,
        "name": 'Pedestrian',
    },
    {
        "id": 3,
        "name": 'Cyclist',
    },
    {
        "id": 4,
        "name": 'Misc',
    },
]


# 初始化 train, val 数据字典
# info licenses categories 在 train 和 val 里面都是一致的
train_data = {'info': info, 'licenses': licenses, 'categories': categories, 'images': [], 'annotations': []}
val_data = {'info': info, 'licenses': licenses, 'categories': categories, 'images': [], 'annotations': []}


# image_path 对应的图像路径，比如 images/train
# label_path 对应的 label 路径，比如 labels/train 跟 images 要对应
def yolo_covert_coco_format(image_path, label_path):
    images = []
    annotations = []
    for index, img_file in enumerate(tqdm(os.listdir(image_path), desc="Processing")):
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

                info_annotation["category_id"] = class_id  # 类别的 id
                info_annotation['bbox'] = [xmin, ymin, bbox_w, bbox_h]  # bbox 的坐标
                info_annotation['area'] = bbox_h * bbox_w  # area
                info_annotation['image_id'] = index  # bbox 的 id
                info_annotation['id'] = index * 100 + idx  # bbox 的 id
                info_annotation['segmentation'] = [[xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]]  # 四个点的坐标
                info_annotation['iscrowd'] = 0  # 单例
                annotations.append(info_annotation)
    return images, annotations


# key == train, val
# 对应要生成的 json 文件，比如 instances_train.json, instances_val.json
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
    data_path = "/home/zourz/work/Dataset/CarDetection/KITTI/Object/data"
    coco_format_path = "/home/zourz/work/Dataset/CarDetection/KITTI/Object/data"
    gen_json_file(data_path, coco_format_path, key='train')
    gen_json_file(data_path, coco_format_path, key='val')
