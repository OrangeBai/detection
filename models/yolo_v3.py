from models.base_model import *
from tensorflow.keras.models import Model
from models.backbone import *
from utils.bbox_helper import *

yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])


class YoloV3(BaseModel):
    def __init__(self):
        super().__init__()

    def build_model(self, input_shape, cls_num, box_num=2, *args, **kwargs):
        super().build_model(input_shape, cls_num, box_num=2, *args, **kwargs)
        input_tensor = Input(input_shape)
        scale_1, scale_2, x = darknet_v3(input_tensor, activation=relu, **kwargs)

        x = darknet_conv(x, 512, activation=relu)
        output_1 = yolo_v3_output(x, 512, len(yolo_anchor_masks[0]), cls_num, relu)

        x = UpSampling2D()(x)
        x = Concatenate()([scale_2, x])
        x = darknet_conv(x, 256, activation=relu)
        output_2 = yolo_v3_output(x, 512, len(yolo_anchor_masks[0]), cls_num, relu)

        x = UpSampling2D()(x)
        x = Concatenate()([scale_1, x])
        x = darknet_conv(x, 128, activation=relu)
        output_3 = yolo_v3_output(x, 512, len(yolo_anchor_masks[0]), cls_num, relu)

        model = Model(input_tensor, [output_3, output_2, output_1])
        model.summary(160)


a = YoloV3()
a.build_model([256, 256, 3], 80)
