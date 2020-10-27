from models.yolo_v1 import *
from pipeline.coco_parser import *
from pipeline.voc_parser import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mse
from utils.losses import *
from config import *

tf.config.experimental_run_functions_eagerly(True)
input_shape = (224, 224)

coco_parser = COCOParser(r'F:\DataSet\COCO', resize=(224, 224), batch_size=32)
val_gen = voc_parser.sequential_gen(feature_size=(7, 7), cls_num=20, box_num=2)
train_gen = voc_parser.balanced_gen(feature_size=(7, 7), cls_num=20, box_num=2)

yolo_v1 = YoloV1()
yolo_v1.build_model(input_shape=input_shape + (3,), cls_num=20)
yolo_v1.compile(Adam(), yolo_v1_loss(2, np.ones((20, 1))), metrics=None, lr_schedule=static_learning_rate,
                init_rate=0.001)

for i in range(5):
    aps = []
    yolo_v1.train_epoch(500, train_gen)
    yolo_v1.update_lr(80, i)

    positive = []
    gt_num = []
    predict_res, gt_res = yolo_v1.evaluate(val_gen, coco_parser.yolo_v1_result_parser, 1000, feature_size=(7, 7),
                                           box_num=2)

    val_results = {category: {'positive': [], 'number': 0} for category in coco_parser.categories}
    for single_pre, single_gt in zip(predict_res, gt_res):
        calculate_tp(val_results, single_gt, single_pre)

    current_ap = {}
    for category, val in val_results.items():
        current_ap[category] = calculate_ap(val['positive'], val['number'])

    mean_ap = 0
    for key, item in current_ap.items():
        mean_ap += item.mean()
    print('Epoch: {0}'.format(i))
    print(mean_ap / len(current_ap))
