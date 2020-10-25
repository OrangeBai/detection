from models.base_model import *
from tensorflow.keras.models import Model
from models.backbone import *
from utils.bbox_helper import *


class YoloV3(BaseModel):
    def __init__(self):
        super().__init__()

    def build_model(self, input_shape, cls_num, box_num=2, *args, **kwargs):
        super().build_model(input_shape, cls_num, box_num=2, *args, **kwargs)
        input_tensor = Input(input_shape)
        feature_map = darknet_v3(input_tensor, activation=relu, **kwargs)
        x = Flatten()(feature_map)
        x = dense_layer(x, units=2048, activation=relu, **kwargs)
