import os

data_root = r'F:\DataSet'

coco_path = os.path.join(data_root, 'COCO')

coco_train_img = os.path.join(coco_path, 'train2017')
coco_val_img = os.path.join(coco_path, 'val2017')
coco_train_annotation =  os.path.join(coco_path, 'annotations', 'instances_train2017.json')
coco_val_annotation = os.path.join(coco_path, 'annotations', 'instances_val2017.json')