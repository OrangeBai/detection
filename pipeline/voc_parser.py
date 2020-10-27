from pipeline.base_parser import *
from bs4 import BeautifulSoup


class VOCParser(DataGenerator):
    def __init__(self, path, resize, batch_size):
        super().__init__(path, resize, batch_size)

    def parse_dataset(self, subset='2012', *args, **kwargs):
        annotation_dir = os.path.join(self.path, 'VOC2012', 'Annotations')
        train_path = os.path.join(self.path, 'VOC2012', 'ImageSets', 'Main', 'train.txt')
        val_path = os.path.join(self.path, 'VOC2012', 'ImageSets', 'Main', 'val.txt')
        image_dir = os.path.join(self.path, 'VOC2012', 'JPEGImages')

        train_labels = {}
        with open(train_path, 'r') as f:
            train_names = [line.rstrip() for line in f]

        for file_name in train_names:
            file_annotation_path = os.path.join(annotation_dir, file_name + '.xml')
            image_path = os.path.join(image_dir, file_name + '.jpg')
            if not os.path.exists(image_path):
                continue
            with open(file_annotation_path) as f:
                soup = BeautifulSoup(f, 'xml')
            image_size = self.__load_image_size__(soup)

            label = []
            objs = soup.find_all('object')
            for obj in objs:
                category, coordinates = self.__load_obj_annotation(obj)
                label.append({
                    'category': category,
                    'bbox': coordinates
                })
                if category not in self.categories:
                    self.categories.append(category)
                    self.category_indexer[category] = []
                self.category_indexer[category].append(file_name)

            train_labels[file_name] = {
                'path': image_path,
                'size': image_size,
                'objects': label
            }

        val_labels = {}
        with open(val_path, 'r') as f:
            val_names = [line.rstrip() for line in f]

        for file_name in val_names:
            file_annotation_path = os.path.join(annotation_dir, file_name + '.xml')
            image_path = os.path.join(image_dir, file_name + '.jpg')
            if not os.path.exists(image_path):
                continue
            with open(file_annotation_path) as f:
                soup = BeautifulSoup(f, 'xml')
            image_size = self.__load_image_size__(soup)

            label = []
            objs = soup.find_all('object')
            for obj in objs:
                category, coordinates = self.__load_obj_annotation(obj)
                label.append({
                    'category': category,
                    'coordinates': coordinates
                })

            val_labels[file_name] = {
                'image_path': image_path,
                'image_size': image_size,
                'label': label
            }
        self.train_labels = train_labels
        self.val_labels = val_labels
        return

    @staticmethod
    def __load_image_size__(soup):
        size = soup.find('size')
        img_size = (int(size.find('width').text), int(size.find('height').text))
        return img_size

    @staticmethod
    def __load_obj_annotation(obj):
        category = obj.find('name', recursive=False).text
        bbox = obj.find('bndbox', recursive=False)
        coordinates = [int(bbox.xmin.text), int(bbox.ymin.text),
                       int(bbox.xmax.text), int(bbox.ymax.text)]
        return category, coordinates

    def __read_labels__(self, batch_label):
        pass


voc_parser = VOCParser(r'F:\DataSet\VOC\VOCtrainval_11-May-2012\VOCdevkit', resize=(224, 224), batch_size=32)
voc_parser.parse_dataset()
gen = voc_parser.balanced_gen(feature_size=(7, 7), cls_num=80, box_num=2)
for i in range(10):
    print(next(gen))
