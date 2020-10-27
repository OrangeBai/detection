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
        while True:
            try:
                x_val, y_val = next(val_gen)
                res = self.model.predict_on_batch(np.array(x_val))
                for cur_res, cur_gt in zip(res, y_val):
                    cur_res = cur_res.numpy()  # eager mode
                    # single_pre = single_pre.eval()             # non-eager
                    predict_result.append(parser(cur_res))
                    ground_truth.append(parser(cur_gt))
            except StopIteration:
                break

        return predict_result, ground_truth
