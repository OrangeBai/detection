from models.yolo_v1 import *
from pipeline.coco_parser import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mse
from utils.losses import *
from config import *

tf.config.experimental_run_functions_eagerly(True)
input_shape = (224, 224)

val_annotation, categories = read_annotation(coco_val_annotation)
train_annotation, _ = read_annotation(coco_train_annotation)

parser = CocoYoloV1(categories, input_shape, 16, downscale=5)
nums = parser.check_cls(train_annotation)
sqrt_weights = 1 / np.sqrt(nums)
cls_weights = sqrt_weights * 1 / np.mean(sqrt_weights)
cls_weights = np.expand_dims(cls_weights, axis=-1)

yolo_v1 = YoloV1()
yolo_v1.build_model(input_shape=input_shape + (3,), cls_num=80)
yolo_v1.compile(Adam(), yolo_v1_loss(2, cls_weights), metrics=None, lr_schedule=static_learning_rate, init_rate=0.001)

for i in range(20):
    aps = []
    train_gen = parser.generator(train_annotation, coco_train_img)
    val_gen = parser.generator(val_annotation[:1000], coco_val_img)
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
