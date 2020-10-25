import numpy as np


def intersection(a, b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[2], b[2]) - x
    h = min(a[3], b[3]) - y
    if w < 0 or h < 0:
        return 0
    return w * h


def union(a, b, area_intersection):
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_union = area_a + area_b - area_intersection
    return area_union


def non_max_supp(boxes, prob, overlap_threshold=0.7, max_boxes=300, min_prob=0):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # calculate the areas
    area = (x2 - x1) * (y2 - y1)

    # sort the bounding boxes
    idxs = np.argsort(prob)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        if prob[i] < min_prob:
            break
        pick.append(i)

        # find the intersection

        xx1_int = np.maximum(x1[i], x1[idxs[:last]])
        yy1_int = np.maximum(y1[i], y1[idxs[:last]])
        xx2_int = np.minimum(x2[i], x2[idxs[:last]])
        yy2_int = np.minimum(y2[i], y2[idxs[:last]])

        ww_int = np.maximum(0, xx2_int - xx1_int)
        hh_int = np.maximum(0, yy2_int - yy1_int)

        area_int = ww_int * hh_int

        # find the union
        area_union = area[i] + area[idxs[:last]] - area_int

        # compute the ratio of overlap
        overlap = area_int / (area_union + 1e-6)

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_threshold)[0])))

        if len(pick) >= max_boxes:
            break

    # return only the bounding boxes that were picked using the integer data type
    boxes = boxes[pick].astype("int")
    prob = prob[pick]

    return boxes, prob


def calculate_iou(a, b):
    """

    @param a: ground truth box coordinates: ((x1, y1), (x2, y2))
    @param b: anchor box coordinates: ((x1, y1), (x2, y2))
    @return: IoU
    """
    for box, i in [(box, i) for box in [a, b] for i in [0, 1]]:
        if box[i] >= box[i + 2]:
            return 0

    area_i = intersection(a, b)
    area_u = union(a, b, area_i)

    return float(area_i) / float(area_u + 1e-6)


def calculate_tp(val_result, gt_label, predict_result):
    """
    Calculate true positive and False positive
    gt_label - A list of ground truth labels [(class_idx, bounding box),...]
    pre_box - Dictionary: {class_index1: [bounding box1], [bounding box2]}
    pre_pro:
    :return:
    """
    pre_box, pre_pro = predict_result
    gt_label_list = []
    for key, val in gt_label[0].items():
        for box in val:
            gt_label_list.append([box, key])

    for label in gt_label_list:
        gt_loc = label[0]
        gt_cls = label[1]
        # for each ground truth class, check if exist in predicted classes
        if gt_cls in pre_box.keys():
            # all the predict boxes, and IoU with the gt box
            pre_boxes = pre_box[gt_cls]
            if len(pre_boxes) == 0:
                continue

            IoUs = [calculate_iou(box, gt_loc) for box in pre_boxes]
            # find the box with highest IoU, and probability
            best_prediction_idx = int(np.argmax(IoUs))
            best_prediction_pro = pre_pro[gt_cls][best_prediction_idx]
            # If class not recorded in positive labels

            # if IoU > threshold, record as true positive, else record as false positive
            if IoUs[best_prediction_idx] > 0.4:
                val_result[gt_cls]['positive'].append([1, best_prediction_pro])
            else:
                val_result[gt_cls]['positive'].append([0, best_prediction_pro])

            # delete the highest positive box
            pre_box[gt_cls] = np.delete(pre_box[gt_cls], best_prediction_idx, axis=0)
            pre_pro[gt_cls] = np.delete(pre_pro[gt_cls], best_prediction_idx)

        val_result[gt_cls]['number'] += 1

    # for all the remaining predicted boxes, record as false positive
    for key, val in pre_box.items():
        for i in range(len(val)):
            val_result[key]['positive'].append([0, pre_pro[key][i]])
    return


def calculate_ap(positive_tags, num_of_positive):
    """

    positive_tags: [[0/1, probability], ...]
    num_of_positive: int
    :return:
    """
    points = np.arange(0, 1.1, 0.1)
    precision = np.zeros((len(points)))
    if num_of_positive == 0 or len(positive_tags) == 0:
        return precision
    array = np.array(positive_tags)
    index = np.argsort(-1 * np.array(array)[:, 1])
    array_sorted = array[index]
    tp = 0
    fp = 0
    for i in range(array.shape[0]):
        if array_sorted[i, 0] == 1:
            tp += 1
        else:
            fp += 1
        cur_precision = tp / (tp + fp)

        recall = tp / num_of_positive
        for j in range(len(precision)):
            if recall >= points[j] and cur_precision >= precision[j]:
                precision[j] = cur_precision
    return precision