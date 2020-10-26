from models.base_model import *
from tensorflow.keras.models import Model
from models.backbone import *
from utils.bbox_helper import *


class YoloV1(BaseModel):
    def __init__(self):
        super().__init__()

    def build_model(self, input_shape, cls_num, box_num=2, *args, **kwargs):
        super().build_model(input_shape, cls_num, box_num=2, *args, **kwargs)
        input_tensor = Input(input_shape)
        feature_map = darknet_v1(input_tensor, activation=relu, **kwargs)
        x = Flatten()(feature_map)
        x = dense_layer(x, units=2048, activation=relu, **kwargs)
        x = Dropout(0.5)(x)
        dense_units = feature_map.shape[1] * feature_map.shape[2] * (5 * box_num + cls_num)
        output_shape = (feature_map.shape[1], feature_map.shape[2], 5 * box_num + cls_num)

        x = dense_layer(x, units=dense_units, activation=linear, batch_norm=False)
        x = Reshape(output_shape)(x)
        model = Model(input_tensor, x)
        model.summary()
        self.model = model
        return

    def evaluate(self, val_gen, parser):
        predict_result = []
        ground_truth = []
        pre_res, gt_res = super().evaluate_on_generator(val_gen)
        st = time.time()
        for single_pre, single_gt in zip(pre_res, gt_res):
            single_pre = single_pre.numpy()  # eager mode
            # single_pre = single_pre.eval()             # non-eager
            predict_result.append(parser.parse_result(single_pre))
            ground_truth.append(parser.parse_result(single_gt))

        val_result = {key: {'positive': [], 'number': 0} for key in range(len(parser.categories[0]))}
        for single_pre, single_gt in zip(predict_result, ground_truth):
            calculate_tp(val_result, single_gt, single_pre)
        print(time.time() - st)
        aps = {}
        for key in val_result.keys():
            aps[key] = calculate_ap(val_result[key]['positive'], val_result[key]['number'])
        return aps
