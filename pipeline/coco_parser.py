import os
import json
import tensorflow as tf
import numpy as np
import cv2
from utils.bbox_helper import *
from utils.pipeline_helper import *
import time
from multiprocessing import Pool


def read_annotation(file_path):
    """
    Parsing COCO annotation
    :param file_path: train/val annotation
    :return: parsed annotation, a dictionary as,
            (image_id,   {
                            'file_name': file_name of image,
                            'size': (height, width),
                            'objects':  [
                                            {   'segmentation' : segmentation_info
                                                'area': segmentation_area,
                                                'iscrowd': boolean,
                                                'image_id': image_id,
                                                'bbox': image bounding box,
                                                'category_id': category id,
                                                'id': id
                                            }
                                        ]
                            }
            )   and
            categories, a list as：
            [[super_category_1, super_category_2, ... , super_category_n],
             [id_1, id_2, ... , id_n],
             [name_1, name_2, ... , name_n]]

    """
    with open(file_path, 'r') as f:
        annotation_file = json.load(f)

    image_annotation = {}
    categories = [[category['supercategory'] for category in annotation_file['categories']],
                  [category['id'] for category in annotation_file['categories']],
                  [category['name'] for category in annotation_file['categories']]]
    for image in annotation_file['images']:
        image_id = image['id']
        image_annotation[image_id] = {
            'file_name': image['file_name'],
            'size': (image['width'], image['height']),
            'objects': [],
        }

    for annotation in annotation_file['annotations']:
        image_id = annotation['image_id']
        image_annotation[image_id]['objects'].append(annotation)

    return sorted(image_annotation.items()), categories


class CocoYoloV1:
    """
    CoCo data generator for Yolo_V1 model
    """

    def __init__(self, categories, resize, batch_size, box_num=2, downscale=5):
        """
        Constructor
        :param categories: categories, from read_annotation()
        :param resize: image size for the model, [width, height, channels]
        :param batch_size: generator batch size
        :param box_num: number of prediction boxes for each grid
        :param downscale: number of max pooling layers
        """
        self.batch_size = batch_size
        self.categories = categories
        self.resize = resize
        self.cls_num = len(categories[0])
        self.box_num = box_num
        feature_x, feature_y = self.resize
        for scale in range(downscale):
            feature_x = feature_x // 2
            feature_y = feature_y // 2
        self.feature_size = (feature_x, feature_y)

    def generator(self, annotations, data_path, endless=False):
        """
        Generator
        :return: None
        """
        annotations = [annotations[idx:idx + self.batch_size] for idx in range(0, len(annotations), self.batch_size)]
        while True:
            for annotation in annotations:
                images = []
                labels = []

                for key, val in annotation:
                    file_name = val['file_name']
                    image_path = os.path.join(data_path, file_name)
                    if not os.path.exists(image_path):
                        continue
                    # Load and resize image
                    cur_image = cv2.imread(image_path)
                    cur_image = cv2.resize(cur_image, self.resize)

                    images.append(cur_image)
                    # parse image label
                    cur_label = self.yolo_v1_gt(val)
                    labels.append(cur_label)
                images = np.array(images)
                images = normalize_m01(images)
                yield [images, labels]
            if not endless:
                break

    def yolo_v1_gt(self, image_annotation):
        """
        Parse image annotation according to model structure
        :param image_annotation: annotation of an image
        :return: ground truth label
        """
        # width and height of resized image
        w, h = image_annotation['size']
        w_new, h_new = self.resize
        # width and height of feature map
        feature_w = self.feature_size[0]
        feature_h = self.feature_size[1]
        # initialize label array, since opencv load image as (height, width, channels), we init the label as same
        label = np.zeros((feature_h, feature_w, self.cls_num + 5 * self.box_num))
        # for each object, calculate the corresponding grid, bounding box offset, size, and category
        for obj in image_annotation['objects']:
            # Resize bounding box
            bbox = obj['bbox']
            bbox = [bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h]
            # bounding box center
            x_center = bbox[0] + bbox[2] / 2
            y_center = bbox[1] + bbox[3] / 2
            # grid index
            x_index = np.int(x_center * feature_w)
            y_index = np.int(y_center * feature_h)
            # offset between ground truth center and grid center
            x_pre = x_center - x_index / feature_w
            y_pre = y_center - y_index / feature_h

            # Width and height of ground truth box
            bbox_w = bbox[2]
            bbox_h = bbox[3]

            # Retrieve bounding class id
            category_id = obj['category_id']
            category_label = self.categories[1].index(category_id)

            # if the current grid is empty, assign values to it,
            # since open cv load image as (height, width, channels),
            # the feature map is set as (height, width, channels)
            if label[y_index, x_index, 0] != 1:
                # all the bounding box in one grid are used to predict the same object
                # [box_1_obj, x_pre, y_pre, w, h, ... , class_1, class_2, ..., class_n]
                for box_id in range(self.box_num):
                    label[y_index, x_index, 5 * box_id] = 1
                    label[y_index, x_index, 5 * box_id + 1: 5 * box_id + 5] = [x_pre, y_pre, bbox_w, bbox_h]
                label[y_index, x_index, self.box_num * 5 + category_label] = 1
        return label

    def parse_result(self, label, obj_threshold=0.5, overlap_threshold=0.5):
        """
        Retrive predict result from label
        :param overlap_threshold:
        :param obj_threshold:
        :param label: model result, or ground truth label
        :return: all predicted boxes:   [
                                            [[x_11, y_11, x_12, y_12], cls_1]
                                            [[x_21, y_21, x_22, y_22], cls_2]
                                            ...
                                        ]
        """
        all_boxes = {}  # result
        all_probs = {}
        # set up center location for all bounding boxes,
        # i.e. x_center[y_id, x_id] is the x center of bounding box [y_id, x_id]
        x_center, y_center = np.meshgrid(range(self.feature_size[0]), range(self.feature_size[1]))
        x_center = x_center / self.feature_size[0]
        y_center = y_center / self.feature_size[1]

        # for each bounding box
        idxs = np.where(label[:, :, [5 * i for i in range(self.box_num)]] > 0.5)
        for y_id, x_id, box_id in zip(idxs[0], idxs[1], idxs[2]):
            # predicted center = corresponding box center + center offset
            pre_center_x = x_center[y_id, x_id] + label[y_id, x_id, 5 * box_id + 1]
            pre_center_y = y_center[y_id, x_id] + label[y_id, x_id, 5 * box_id + 2]
            # predicted box width and height
            pre_w, pre_h = label[y_id, x_id, 5 * box_id + 3: 5 * box_id + 5]
            pre_box = [(pre_center_x - pre_w / 2) * self.resize[0], (pre_center_y - pre_h / 2) * self.resize[1],
                       (pre_center_x + pre_w / 2) * self.resize[0], (pre_center_y + pre_h / 2) * self.resize[1]]
            # highest class
            pre_cls = np.argmax(label[y_id, x_id, 5 * self.box_num:])
            pre_prob = label[y_id, x_id, 5 * self.box_num + pre_cls]
            if pre_cls not in all_boxes.keys():
                all_boxes[pre_cls] = []
                all_probs[pre_cls] = []
            all_boxes[pre_cls].append(pre_box)
            all_probs[pre_cls].append(pre_prob)

        for key in list(all_boxes.keys()):
            # Apply non maximum suppression for each class
            boxes, probs = non_max_supp(np.array(all_boxes[key]),
                                        np.array(all_probs[key]),
                                        overlap_threshold=0.7)
            all_boxes[key] = boxes
            all_probs[key] = probs
        return all_boxes, all_probs


# val_ann_path = r"F:\DataSet\COCO\annotations_trainval2017\annotations\instances_val2017.json"
# val_img_path = r"F:\DataSet\COCO\val2017\val2017"
# annotation, categories = read_annotation(val_ann_path)
#
# a = CocoYoloV1(categories, (600, 400), 32)
# val_gen = a.generator(annotation, val_img_path)
#
# for i in range(10):
#     d = next(val_gen)
#     img = d[0][0]
#     img = img * 255.0
#     label = d[1][0]
#     bbox = a.parse_result(label)[0]
#     for key, boxes in bbox.items():
#         for box in boxes:
#             cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0))
#     # for box, cls in bbox:
#     #     cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0))
#     cv2.imwrite(r"C:\Users\jzp0306\Desktop\1" + str(i) + ".jpg", img)
#     print(a)
