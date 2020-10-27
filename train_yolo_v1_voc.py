from models.yolo_v1 import *
from pipeline.coco_parser import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mse
from utils.losses import *
from pipeline.voc_parser import *
from config import *

tf.config.experimental_run_functions_eagerly(True)
input_shape = (224, 224)

voc_parser = VOCParser(r'F:\DataSet\VOC\VOCtrainval_11-May-2012\VOCdevkit', resize=(224, 224), batch_size=32)
voc_parser.parse_dataset()
train_gen = voc_parser.balanced_gen(feature_size=(7, 7), cls_num=20, box_num=2)

yolo_v1 = YoloV1()
yolo_v1.build_model(input_shape=input_shape + (3,), cls_num=20)
yolo_v1.compile(Adam(), yolo_v1_loss(2, np.ones((20, 1))), metrics=None, lr_schedule=static_learning_rate,
                init_rate=0.001)

for i in range(20):
    aps = []
    train_gen = voc_parser.balanced_gen(feature_size=(7, 7), cls_num=20, box_num=2)
    # val_gen = voc.generator(val_annotation[:1000], coco_val_img)
    yolo_v1.train_epoch(1000, train_gen)
    yolo_v1.update_lr(80, i)

    positive = []
    gt_num = []
    current_ap = yolo_v1.evaluate(val_gen, parser)
    mean_ap = 0
    for key, item in current_ap.items():
        mean_ap += item.mean()
    print('Epoch: {0}'.format(i))
    print(mean_ap / len(current_ap))
