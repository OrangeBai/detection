from random import choice
import os
import cv2
import numpy as np
from utils.pipeline_helper import *
from utils.bbox_helper import *
import json


class DataGenerator:
    def __init__(self, path, resize, batch_size):
        """
        Parse the data set
        categories = [name_1, name_2, ...]
        category_indexer = [ name_1:  [image_id_1, iamge_id_2, ..., image_id_n],
                        name_1:  [image_id_1, iamge_id_2, ..., image_id_n],...
                    ]
        train_labels = {
                            image_id_1: {'path': path, 'size':(width, height),
                                        'objects': [
                                                        {'category': category_name, 'bbox': [x1, y1, x2, y2]},
                                                        {'category': category_name, 'bbox': [x1, y1, x2, y2]},...
                                                   ]
                                        },...
                        }
        val_labels has the same structure with train_labels
        category_counter = {'category_name': number of trained_image, ...}
        :param path: Dataset path
        :param resize: image size for the model, [width, height, channels]
        :param batch_size: generator batch size
        """
        self.path = path
        self.resize = resize
        self.batch_size = batch_size

        self.categories = []
        self.category_indexer = {}

        self.train_labels = {}
        self.val_labels = {}

        if not self.load_dataset():
            self.parse_dataset()

        self.category_counter = {category: 0 for category in self.categories}

    def parse_dataset(self, *args, **kwargs):
        """
        Parse dataset
        :return:
        """
        pass

    def balanced_gen(self, model='yolo_v1', *args, **kwargs):
        """
        Sample balanced data generator, only used for training pipeline
        :param model: model name
        :param args: arguments passed to label parser
        :param kwargs: keyword arguments passed to label parser
        :return: generator
        """

        # parser is the function used to convert formatted data label into model ground truth matrix
        parser = self.__set_parser__(model)

        def generator():
            while True:
                indexes = self.__next_annotation_batch__()          # generate a batch of image ids
                images, labels = self.parse_batch_indexes(indexes, parser, 'train', *args, **kwargs)

                yield images, labels

        return generator()

    def sequential_gen(self, dataset='val', model='yolo_v1', endless=False, *args, **kwargs):
        parser = self.__set_parser__(model)

        def generator():
            annotations = self.__set_annotations__(dataset)
            while True:
                for indexes in annotations:
                    images, labels = self.parse_batch_indexes(indexes, parser, dataset, *args, **kwargs)
                    yield [images, labels]
                if not endless:
                    break
                else:
                    annotations = self.__set_annotations__(dataset)
        return generator()

    def parse_batch_indexes(self, indexes, parser, dataset, *args, **kwargs):
        """
        Parse a batch of image ids
        :param indexes: a list of image index, [image_id_1, image_id_2, ..., image_id_n]
        :param parser: parser function, receive *args, and **kwargs as input
        :param args:
        :param kwargs:
        :return:
        """
        images = []
        labels = []

        for index in indexes:
            if dataset == 'train':
                image_annotation = self.train_labels[index]
            else:
                image_annotation = self.val_labels[index]
            image_path = image_annotation['path']
            # Load and resize image
            cur_image = cv2.imread(image_path)
            cur_image = cv2.resize(cur_image, self.resize)

            images.append(cur_image)
            # parse image label
            cur_label = parser(image_annotation, *args, **kwargs)
            labels.append(cur_label)

        images = np.array(images)
        images = normalize_m01(images)
        labels = np.array(labels)
        return images, labels

    def __set_parser__(self, model):
        if model == 'yolo_v1':
            parser = self.yolo_v1_label_parser
        else:
            parser = self.yolo_v1_label_parser
        return parser

    def __set_annotations__(self, dataset):
        """
        split annotation batch
        :param dataset: if val, split val dataset, otherwise split training dataset
        :return:
        """
        if dataset == 'val':
            annotations = sorted(self.val_labels.keys())
        else:
            annotations = sorted(self.train_labels.keys())
        return [annotations[idx:idx+self.batch_size] for idx in range(0, len(annotations), self.batch_size)]

    def __next_annotation_batch__(self):
        """
        Generate a batch of image indexes in accordance with data balance,
        i.e. if category_i has minimum trained images, then next image should be choiced from this class.
        :return: [image_id_1, image_id_2, ..., image_id_k]
        """
        annotation_batch = []
        for i in range(self.batch_size):
            # check minimum category, if minimum < maximum/3, then next category is the minimum category
            # otherwise, randomly choose one
            # from category_indexer pick a image_id
            next_category = self.__check_min__()
            current_indexer = choice(self.category_indexer[next_category])
            annotation_batch.append(current_indexer)
            
            for object in self.train_labels[current_indexer]['objects']:
                # check objects contained in the current image, and update the category_counter
                category = object['category']
                self.category_counter[category] += 1

        return annotation_batch

    def __check_min__(self):
        """
        Check the least trained action,
        if the least is half or less than the most, then generate the least trained action
        otherwise randomly choice an action

        Return:
            action name --  action name of the next action
        """
        maximum_category = max(self.category_counter.keys(), key=(lambda k: self.category_counter[k]))
        minimum_category = min(self.category_counter.keys(), key=(lambda k: self.category_counter[k]))
        if self.category_counter[minimum_category] < self.category_counter[maximum_category] * 0.33:
            return minimum_category
        else:
            return choice(list(self.category_counter.keys()))

    def yolo_v1_label_parser(self, image_annotation, feature_size, cls_num, box_num):
        """
        Parse image annotation according to model structure
        :param image_annotation: annotation of an image
        :param box_num: number of boxes
        :param cls_num: number of classes
        :param feature_size: size of feature map, (width, height)
        :return: ground truth label
        """
        # width and height of resized image
        w, h = image_annotation['size']
        w_new, h_new = self.resize
        # width and height of feature map
        feature_w = feature_size[0]
        feature_h = feature_size[1]
        # initialize label array, since opencv load image as (height, width, channels), we init the label as same
        label = np.zeros((feature_h, feature_w, cls_num + 5 * box_num))
        # for each object, calculate the corresponding grid, bounding box offset, size, and category
        for obj in image_annotation['objects']:
            # Resize bounding box
            bbox = obj['bbox']
            bbox = [bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h]
            # bounding box center
            x_center = (bbox[0] + bbox[2]) / 2
            y_center = (bbox[1] + bbox[3]) / 2
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
            category = obj['category']
            category_label = self.categories.index(category)

            # if the current grid is empty, assign values to it,
            # since open cv load image as (height, width, channels),
            # the feature map is set as (height, width, channels)
            if label[y_index, x_index, 0] != 1:
                # all the bounding box in one grid are used to predict the same object
                # [box_1_obj, x_pre, y_pre, w, h, ... , class_1, class_2, ..., class_n]
                for box_id in range(box_num):
                    label[y_index, x_index, 5 * box_id] = 1
                    label[y_index, x_index, 5 * box_id + 1: 5 * box_id + 5] = [x_pre, y_pre, bbox_w, bbox_h]
                label[y_index, x_index, box_num * 5 + category_label] = 1
        return label

    def yolo_v1_result_parser(self, result, feature_size, box_num, obj_threshold=0.5, overlap_threshold=0.7):
        """
        retrieve predict result from label
        :param box_num:
        :param feature_size:
        :param overlap_threshold:
        :param obj_threshold:
        :param result: model result, or ground truth label
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
        x_center, y_center = np.meshgrid(range(feature_size[0]), range(feature_size[1]))
        x_center = x_center / feature_size[0]
        y_center = y_center / feature_size[1]

        # for each bounding box
        idxs = np.where(result[:, :, [5 * i for i in range(box_num)]] > obj_threshold)
        for y_id, x_id, box_id in zip(idxs[0], idxs[1], idxs[2]):
            # predicted center = corresponding box center + center offset
            pre_center_x = x_center[y_id, x_id] + result[y_id, x_id, 5 * box_id + 1]
            pre_center_y = y_center[y_id, x_id] + result[y_id, x_id, 5 * box_id + 2]
            # predicted box width and height
            pre_w, pre_h = result[y_id, x_id, 5 * box_id + 3: 5 * box_id + 5]
            pre_box = [(pre_center_x - pre_w / 2) * self.resize[0], (pre_center_y - pre_h / 2) * self.resize[1],
                       (pre_center_x + pre_w / 2) * self.resize[0], (pre_center_y + pre_h / 2) * self.resize[1]]
            # highest class
            pre_cls = int(np.argmax(result[y_id, x_id, 5 * box_num:]))
            category = self.categories[pre_cls]
            pre_prob = result[y_id, x_id, 5 * box_num + pre_cls]
            if category not in all_boxes.keys():
                all_boxes[category] = []
                all_probs[category] = []
            all_boxes[category].append(pre_box)
            all_probs[category].append(pre_prob)

        for key in list(all_boxes.keys()):
            # Apply non maximum suppression for each class
            boxes, probs = non_max_supp(np.array(all_boxes[key]),
                                        np.array(all_probs[key]),
                                        overlap_threshold=overlap_threshold)
            all_boxes[key] = boxes
            all_probs[key] = probs
        return all_boxes, all_probs

    def save_dataset(self):
        """
        Save the dataset to base path, so that it does not need to parse all the annotation each time
        :return: None
        """
        dataset_path = os.path.join(self.path, 'dataset.json')
        dataset = {
            'categories': self.categories,
            'category_indexer': self.category_indexer,
            'train_labels': self.train_labels,
            'val_labels': self.val_labels
        }
        with open(dataset_path, 'w') as f:
            json.dump(dataset, f)
        return

    def load_dataset(self):
        """
        Load th
        :return:
        """
        dataset_path = os.path.join(self.path, 'dataset.json')
        try:
            with open(dataset_path, 'r') as f:
                dataset = json.load(f)
            self.categories = dataset['categories']
            self.category_indexer = dataset['category_indexer']
            self.train_labels = dataset['train_labels']
            self.val_labels = dataset['val_labels']
            return True
        except (KeyError, FileNotFoundError, IOError):
            return False
