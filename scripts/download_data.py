import wget
import config
import os
from zipfile import ZipFile

print('Beginning file download with wget module')

coco_train_url = 'http://images.cocodataset.org/zips/train2017.zip'
coco_val_url = 'http://images.cocodataset.org/zips/val2017.zip'
coco_annotation_url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
config.coco_path = r'F:\DataSet\COCO2'
coco_train_zip_path = os.path.join(config.coco_path, 'train2017.zip')
coco_val_zip_path = os.path.join(config.coco_path, 'val2017.zip')
coco_annotation_zip_path = os.path.join(config.coco_path, 'annotations_trainval2017.zip')


wget.download(coco_annotation_url, coco_annotation_zip_path)
# wget.download(coco_train_url, coco_train_zip_path)
# wget.download(coco_val_url, coco_val_zip_path)

# with ZipFile(coco_train_zip_path, 'r') as f:
#     f.extractall(config.coco_path)
#
# with ZipFile(coco_val_zip_path, 'r') as f:
#     f.extractall(config.coco_path)

with ZipFile(coco_annotation_zip_path, 'r') as f:
   f.extractall(config.coco_path)
