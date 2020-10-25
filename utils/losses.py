from tensorflow.keras.losses import mse
import tensorflow.keras.backend as bk


def yolo_v1_loss(box_num, lambda_co=5, lambda_no=0.5):
    def yolo_v1_loss_fun(y_gt, y_pre):
        loss_coord = 0
        loss_none = 0
        pos = y_gt[:, :, :, 0]
        neg = 1 - pos
        for box_id in range(box_num):
            box_st = 5 * box_id
            loss_coord += bk.sum(pos * bk.sum(bk.square(y_gt[..., box_st: box_st + 3] -
                                                        y_pre[..., box_st: box_st + 3]), axis=-1))
            loss_coord += bk.sum(pos * bk.sum(bk.square(y_gt[:, :, :, box_st + 3: box_st + 5] -
                                                        y_pre[:, :, :, box_st + 3: box_st + 5]), axis=-1))
            loss_none += bk.sum(neg * bk.sum(bk.square(y_gt[:, :, :, box_st: box_st + 1] -
                                                       y_pre[:, :, :, box_st: box_st + 1]), axis=-1))
        loss_class = bk.sum(pos * bk.sum(bk.square(y_gt[..., 5 * box_num:] - y_pre[..., 5 * box_num:]), axis=-1))

        return loss_coord * lambda_co + loss_none * lambda_no + loss_class

    return yolo_v1_loss_fun
