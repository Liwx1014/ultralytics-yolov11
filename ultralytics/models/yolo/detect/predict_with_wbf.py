# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
import numpy as np
import warnings
# ⭐️ 可视化导入
import cv2
import os
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops
import warnings
import numpy as np

def prefilter_boxes(boxes, scores, labels, weights, thr):
    # Create dict with boxes stored by its label
    new_boxes = dict()

    for t in range(len(boxes)):

        if len(boxes[t]) != len(scores[t]):
            print('Error. Length of boxes arrays not equal to length of scores array: {} != {}'.format(len(boxes[t]), len(scores[t])))
            exit()

        if len(boxes[t]) != len(labels[t]):
            print('Error. Length of boxes arrays not equal to length of labels array: {} != {}'.format(len(boxes[t]), len(labels[t])))
            exit()

        for j in range(len(boxes[t])):
            score = scores[t][j]
            if score < thr:
                continue
            label = int(labels[t][j])
            box_part = boxes[t][j]
            x1 = float(box_part[0])
            y1 = float(box_part[1])
            x2 = float(box_part[2])
            y2 = float(box_part[3])

            # Box data checks
            if x2 < x1:
                warnings.warn('X2 < X1 value in box. Swap them.')
                x1, x2 = x2, x1
            if y2 < y1:
                warnings.warn('Y2 < Y1 value in box. Swap them.')
                y1, y2 = y2, y1
            if x1 < 0:
                warnings.warn('X1 < 0 in box. Set it to 0.')
                x1 = 0
            if x1 > 1:
                warnings.warn('X1 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range.')
                x1 = 1
            if x2 < 0:
                warnings.warn('X2 < 0 in box. Set it to 0.')
                x2 = 0
            if x2 > 1:
                warnings.warn('X2 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range.')
                x2 = 1
            if y1 < 0:
                warnings.warn('Y1 < 0 in box. Set it to 0.')
                y1 = 0
            if y1 > 1:
                warnings.warn('Y1 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range.')
                y1 = 1
            if y2 < 0:
                warnings.warn('Y2 < 0 in box. Set it to 0.')
                y2 = 0
            if y2 > 1:
                warnings.warn('Y2 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range.')
                y2 = 1
            if (x2 - x1) * (y2 - y1) == 0.0:
                warnings.warn("Zero area box skipped: {}.".format(box_part))
                continue

            # [label, score, weight, model index, x1, y1, x2, y2]
            b = [int(label), float(score) * weights[t], weights[t], t, x1, y1, x2, y2]
            if label not in new_boxes:
                new_boxes[label] = []
            new_boxes[label].append(b)

    # Sort each list in dict by score and transform it to numpy array
    for k in new_boxes:
        current_boxes = np.array(new_boxes[k])
        new_boxes[k] = current_boxes[current_boxes[:, 1].argsort()[::-1]]

    return new_boxes


def get_weighted_box(boxes, conf_type='avg'):
    """
    Create weighted box for set of boxes
    :param boxes: set of boxes to fuse
    :param conf_type: type of confidence one of 'avg' or 'max'
    :return: weighted box (label, score, weight, model index, x1, y1, x2, y2)
    """

    box = np.zeros(8, dtype=np.float32)
    conf = 0
    conf_list = []
    w = 0
    for b in boxes:
        box[4:] += (b[1] * b[4:])
        conf += b[1]
        conf_list.append(b[1])
        w += b[2]
    box[0] = boxes[0][0]
    if conf_type in ('avg', 'box_and_model_avg', 'absent_model_aware_avg'):
        box[1] = conf / len(boxes)
    elif conf_type == 'max':
        box[1] = np.array(conf_list).max()
    box[2] = w
    box[3] = -1 # model index field is retained for consistency but is not used.
    box[4:] /= conf
    return box


def find_matching_box_fast(boxes_list, new_box, match_iou):
    """
        Reimplementation of find_matching_box with numpy instead of loops. Gives significant speed up for larger arrays
        (~100x). This was previously the bottleneck since the function is called for every entry in the array.
    """
    def bb_iou_array(boxes, new_box):
        # bb interesection over union
        xA = np.maximum(boxes[:, 0], new_box[0])
        yA = np.maximum(boxes[:, 1], new_box[1])
        xB = np.minimum(boxes[:, 2], new_box[2])
        yB = np.minimum(boxes[:, 3], new_box[3])

        interArea = np.maximum(xB - xA, 0) * np.maximum(yB - yA, 0)

        # compute the area of both the prediction and ground-truth rectangles
        boxAArea = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        boxBArea = (new_box[2] - new_box[0]) * (new_box[3] - new_box[1])

        iou = interArea / (boxAArea + boxBArea - interArea)

        return iou

    if boxes_list.shape[0] == 0:
        return -1, match_iou

    # boxes = np.array(boxes_list)
    boxes = boxes_list

    ious = bb_iou_array(boxes[:, 4:], new_box[4:])

    ious[boxes[:, 0] != new_box[0]] = -1

    best_idx = np.argmax(ious)
    best_iou = ious[best_idx]

    if best_iou <= match_iou:
        best_iou = match_iou
        best_idx = -1

    return best_idx, best_iou


def weighted_boxes_fusion(
        boxes_list,
        scores_list,
        labels_list,
        weights=None,
        iou_thr=0.55,
        skip_box_thr=0.0,
        conf_type='avg',
        allows_overflow=False
):
    '''
    :param boxes_list: list of boxes predictions from each model, each box is 4 numbers.
    It has 3 dimensions (models_number, model_preds, 4)
    Order of boxes: x1, y1, x2, y2. We expect float normalized coordinates [0; 1]
    :param scores_list: list of scores for each model
    :param labels_list: list of labels for each model
    :param weights: list of weights for each model. Default: None, which means weight == 1 for each model
    :param iou_thr: IoU value for boxes to be a match
    :param skip_box_thr: exclude boxes with score lower than this variable
    :param conf_type: how to calculate confidence in weighted boxes.
        'avg': average value,
        'max': maximum value,
        'box_and_model_avg': box and model wise hybrid weighted average,
        'absent_model_aware_avg': weighted average that takes into account the absent model.
    :param allows_overflow: false if we want confidence score not exceed 1.0

    :return: boxes: boxes coordinates (Order of boxes: x1, y1, x2, y2).
    :return: scores: confidence scores
    :return: labels: boxes labels
    '''

    if weights is None:
        weights = np.ones(len(boxes_list))
    if len(weights) != len(boxes_list):
        print('Warning: incorrect number of weights {}. Must be: {}. Set weights equal to 1.'.format(len(weights), len(boxes_list)))
        weights = np.ones(len(boxes_list))
    weights = np.array(weights)

    if conf_type not in ['avg', 'max', 'box_and_model_avg', 'absent_model_aware_avg']:
        print('Unknown conf_type: {}. Must be "avg", "max" or "box_and_model_avg", or "absent_model_aware_avg"'.format(conf_type))
        exit()

    filtered_boxes = prefilter_boxes(boxes_list, scores_list, labels_list, weights, skip_box_thr)
    if len(filtered_boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))

    overall_boxes = []
    for label in filtered_boxes:
        boxes = filtered_boxes[label]
        new_boxes = []
        weighted_boxes = np.empty((0, 8))

        # Clusterize boxes
        for j in range(0, len(boxes)):
            index, best_iou = find_matching_box_fast(weighted_boxes, boxes[j], iou_thr)

            if index != -1:
                new_boxes[index].append(boxes[j])
                weighted_boxes[index] = get_weighted_box(new_boxes[index], conf_type)
            else:
                new_boxes.append([boxes[j].copy()])
                weighted_boxes = np.vstack((weighted_boxes, boxes[j].copy()))

        # Rescale confidence based on number of models and boxes
        for i in range(len(new_boxes)):
            clustered_boxes = new_boxes[i]
            if conf_type == 'box_and_model_avg':
                clustered_boxes = np.array(clustered_boxes)
                # weighted average for boxes
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * len(clustered_boxes) / weighted_boxes[i, 2]
                # identify unique model index by model index column
                _, idx = np.unique(clustered_boxes[:, 3], return_index=True)
                # rescale by unique model weights
                weighted_boxes[i, 1] = weighted_boxes[i, 1] *  clustered_boxes[idx, 2].sum() / weights.sum()
            elif conf_type == 'absent_model_aware_avg':
                clustered_boxes = np.array(clustered_boxes)
                # get unique model index in the cluster
                models = np.unique(clustered_boxes[:, 3]).astype(int)
                # create a mask to get unused model weights
                mask = np.ones(len(weights), dtype=bool)
                mask[models] = False
                # absent model aware weighted average
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * len(clustered_boxes) / (weighted_boxes[i, 2] + weights[mask].sum())
            elif conf_type == 'max':
                weighted_boxes[i, 1] = weighted_boxes[i, 1] / weights.max()
            elif not allows_overflow:
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * min(len(weights), len(clustered_boxes)) / weights.sum()
            else:
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * len(clustered_boxes) / weights.sum()
        overall_boxes.append(weighted_boxes)
    overall_boxes = np.concatenate(overall_boxes, axis=0)
    overall_boxes = overall_boxes[overall_boxes[:, 1].argsort()[::-1]]
    boxes = overall_boxes[:, 4:]
    scores = overall_boxes[:, 1]
    labels = overall_boxes[:, 0]
    return boxes, scores, labels

class DetectionPredictor(BasePredictor):
    def postprocess(self, preds, img, orig_imgs, **kwargs):

        '''
        这个过程大致如下：
            算法首先将所有预测框按置信度分数从高到低排序。
            它取出分数最高的那个蓝框，以它为核心，创建第1个聚类。
            然后，它取出下一个蓝框，计算这个框与现有每个聚类的代表框的IoU。
            关键判断：
            如果这个新框与第1个聚类的代表框的IoU大于您设置的iou_thr（例如0.5），那么这个新框就被认为是“第1个聚类的朋友”，并被加入到这个聚类中。
            如果这个新框与所有现有聚类的IoU都小于iou_thr，算法就会认为它不属于任何一个现有的“朋友圈”，于是就为它创建一个新的聚类（例如，第2个聚类）。
            这个过程会一直持续，直到所有的蓝框都被分配到某个聚类中。
        
        
        '''
        print("Using WBF Method Fusion-----")

        # WBF 参数
        # 根据重叠程度（IoU），将这些蓝框分成不同的“群组”或“簇”（Cluster）
        wbf_iou_thr = 0.01
        wbf_skip_box_thr = 0.3
        conf_type = 'avg'
        
        # ⭐️ FIX 3: 使用 self.args 动态控制可视化，而不是硬编码
        visualize = False
        self.img_shape = img.shape[2:]
        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        preds = preds.permute(0, 2, 1)
        preds[..., :4] = ops.xywh2xyxy(preds[..., :4])
        # 每个检测框的最高置信度分数 labels: 每个检测框对应的类别标签(索引
        scores, labels = preds[..., 4:].max(-1)
        
        preds_np = preds.cpu().numpy()
        scores_np = scores.cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        img_h, img_w = self.img_shape
        results_list = []

        for i in range(preds_np.shape[0]):
            pred_i, scores_i, labels_i = preds_np[i], scores_np[i], labels_np[i]

            mask = scores_i >= wbf_skip_box_thr
            boxes_i, scores_i, labels_i = pred_i[mask, :4], scores_i[mask], labels_i[mask]

            pre_wbf_boxes_pixels = boxes_i.copy() if visualize else None

            if not boxes_i.shape[0]:
                results_list.append(torch.empty((0, 6), device=preds.device))
                if visualize:  
                    self.save_visualization(orig_imgs[i], None, None, i)
                continue
            
            # 为 WBF 准备数据 (归一化)
            boxes_i_normalized = boxes_i.copy()
            boxes_i_normalized[:, [0, 2]] /= img_w
            boxes_i_normalized[:, [1, 3]] /= img_h
            
            boxes_list_wbf = [boxes_i_normalized.tolist()]
            scores_list_wbf = [scores_i.tolist()]
            labels_list_wbf = [labels_i.tolist()]
            
            fused_boxes_normalized, fused_scores, fused_labels = weighted_boxes_fusion(
                boxes_list_wbf, scores_list_wbf, labels_list_wbf,
                weights=None, iou_thr=wbf_iou_thr, skip_box_thr=wbf_skip_box_thr, conf_type=conf_type
            )
            
            # ⭐️ FIX 1: 将可视化逻辑移到正确的位置
            if fused_boxes_normalized.shape[0] > 0:
                # 反归一化
                fused_boxes_pixels = fused_boxes_normalized.copy()
                fused_boxes_pixels[:, [0, 2]] *= img_w
                fused_boxes_pixels[:, [1, 3]] *= img_h
                
                # 合并结果
                fused_results = torch.cat([
                    torch.from_numpy(fused_boxes_pixels),
                    torch.from_numpy(fused_scores).unsqueeze(1),
                    torch.from_numpy(fused_labels).unsqueeze(1)
                ], dim=1).to(preds.device)
                results_list.append(fused_results)

                # 在成功融合后调用可视化
                if visualize:
                    self.save_visualization(orig_imgs[i], pre_wbf_boxes_pixels, fused_boxes_pixels, i)
            else:
                results_list.append(torch.empty((0, 6), device=preds.device))
                # ⭐️ FIX 2: 修正可视化调用参数，当融合后没有框时，post_boxes应为None
                if visualize:
                    self.save_visualization(orig_imgs[i], pre_wbf_boxes_pixels, None, i)

        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        return self.construct_results(results_list, img, orig_imgs, **kwargs)

    def save_visualization(self, orig_img, pre_boxes, post_boxes, batch_idx):
        PRE_COLOR = (255, 150, 0)  # 亮蓝色
        POST_COLOR = (0, 255, 0)   # 绿色
        
        vis_img = orig_img.copy()
        img_shape = vis_img.shape
        
        if pre_boxes is not None and len(pre_boxes) > 0:
            scaled_pre_boxes = ops.scale_boxes(self.img_shape, pre_boxes, img_shape)
            for box in scaled_pre_boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), PRE_COLOR, 2) # 线条加粗一点

        if post_boxes is not None and len(post_boxes) > 0:
            scaled_post_boxes = ops.scale_boxes(self.img_shape, post_boxes, img_shape)
            for box in scaled_post_boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), POST_COLOR, 3) # 线条更粗以区分
                cv2.putText(vis_img, 'WBF', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, POST_COLOR, 2)

        p = self.batch[0][batch_idx]
        base, ext = os.path.splitext(os.path.basename(p))
        save_dir = '/home/lwx/work/Code/Experiment/JY-Team/compare_wbf_2'
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{base}_wbf_vis{ext}')
        
        cv2.imwrite(save_path, vis_img)
        # ⭐️ FIX 4: 使用 LOGGER 替代 print
        print(f"WBF visualization saved to {save_path}")

    def get_obj_feats(self, feat_maps, idxs):
        # 此函数无需修改
        import torch
        s = min([x.shape[1] for x in feat_maps]); obj_feats = torch.cat([x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, s, x.shape[1] // s).mean(dim=-1) for x in feat_maps], dim=1)
        return [feats[idx] if len(idx) else [] for feats, idx in zip(obj_feats, idxs)]

    def construct_results(self, preds, img, orig_imgs, **kwargs):
        # 存储模型输入尺寸以备可视化使用
        self.img_shape = img.shape[2:]
        return [self.construct_result(pred, img, orig_img, img_path) for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0])]

    def construct_result(self, pred, img, orig_img, img_path):
        # 此函数无需修改
        if pred.shape[0] > 0:
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6])