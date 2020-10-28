from pipeline.base_parser import *
from utils.bbox_helper import *


class COCOParser(DataGenerator):
    """
    CoCo data generator for Yolo_V1 model
    """

    def __init__(self, path, resize, batch_size):
        """
        Constructor
        :param resize: image size for the model, [width, height, channels]
        :param batch_size: generator batch size
        """
        super().__init__(path, resize, batch_size)

    def parse_dataset(self, *args, **kwargs):
        annotation_path = os.path.join(self.path, 'annotations')
        train_path = os.path.join(annotation_path, 'instances_train2017.json')
        val_path = os.path.join(annotation_path, 'instances_val2017.json')
        train_image_dir = os.path.join(self.path, 'train2017')
        val_img_dir = os.path.join(self.path, 'val2017')

        self.train_labels = self.read_annotation(train_path, train_image_dir, 'train')
        self.val_labels = self.read_annotation(val_path, val_img_dir, 'val')
        self.save_dataset()

    def read_annotation(self, annotation_path, image_dir, dataset='train'):
        with open(annotation_path, 'r') as f:
            annotation_file = json.load(f)

        if dataset == 'train':
            self.categories = [category['name'] for category in annotation_file['categories']]
            self.category_indexer = {category: [] for category in self.categories}
            self.category_counter = {category: 0 for category in self.categories}

        image_annotation = {}
        category_id_dict = {category['id']: category['name'] for category in annotation_file['categories']}
        for image in annotation_file['images']:
            image_path = os.path.join(image_dir, image['file_name'])
            if not os.path.exists(image_path):
                continue
            image_size = (image['width'], image['height'])
            image_id = image['id']
            image_annotation[image_id] = {
                'path': image_path,
                'size': image_size,
                'objects': [],
            }
        for annotation in annotation_file['annotations']:
            image_id = annotation['image_id']
            x1, y1, w, h = annotation['bbox']
            bbox = [x1, y1, x1 + w, y1 + h]
            category_id = annotation['category_id']
            category = category_id_dict[category_id]

            if dataset == 'train' and image_id in image_annotation.keys():
                self.category_indexer[category].append(str(image_id))

            if image_id in image_annotation.keys():
                image_annotation[image_id]['objects'].append(
                    {'category': category,
                     'bbox': bbox}
                )

        return image_annotation


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


# a = COCOParser(r'F:\DataSet\COCO', resize=(224, 224), batch_size=32)
