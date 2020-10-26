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
        scale_1, scale_2, x = darknet_v3(input_tensor, activation=relu, **kwargs)

        x = darknet_block(x, )

