# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_loss_n_evalMetrics(loss_or_metric_types=[],
                               loss_or_metric_weights=[],
                               gt_occMap=None,
                               pred_occMap=None,
                               mask=None,
                               exploredPart_mask=None,
                               target_category=None,
                               eval_mode=False,
                               dont_collapse_across_batch=False,
                               is_stitched=False,
                               ):
    """
    compute loss and evaluation metrics for the Chat2Map task
    :param loss_or_metric_types: types of loss or eval metrics
    :param loss_or_metric_weights: weights for computing loss or eval metrics
    :param gt_occMap: ground-truth occupancy map
    :param pred_occMap: predicted occupancy map
    :param mask: query mask saying which entries in the prediction and ground truth are valid and which aren't
    :param exploredPart_mask: mask saying which parts of the predicted and ground-truth maps have been seen in inputs
    :param target_category: target category for computing losses or eval metrics
    :param eval_mode: flag saying if the function is being called in eval mode or not
    :param dont_collapse_across_batch: flag saying if the computed loss or eval metrics should be aggregated (averaged)
                                        over the batch dimension or not
    :param is_stitched: flag saying if the ground truths and the predictions have been stitchec onto a shared map
    :return: computed loss or eval metric values.
    """

    if is_stitched:
        assert eval_mode
        assert pred_occMap.size(-1) == gt_occMap.size(-1) == 1

    if eval_mode:
        loss_or_metric_all_batch_idxs = []
        for i in range(pred_occMap.size(0)):
            if is_stitched:
                pred_validIdxs = torch.where(pred_occMap[i] == 1.)
                if len(pred_validIdxs[0]) == 0:
                    pred_minValidRow = pred_minValidCol = 0
                    pred_maxValidRow = pred_maxValidCol = pred_occMap[i].shape[0] - 1
                else:
                    pred_minValidRow = pred_validIdxs[0][0].item()
                    pred_maxValidRow = pred_validIdxs[0][-1].item()
                    pred_minValidCol = torch.min(pred_validIdxs[1]).item()      # pred_validIdxs[1][0].item()
                    pred_maxValidCol = torch.max(pred_validIdxs[1]).item()      # pred_validIdxs[1][-1].item()

                gt_validIdxs = torch.where(gt_occMap[i] == 1.)
                if len(gt_validIdxs[0]) == 0:
                    gt_minValidRow = gt_minValidCol = 0
                    gt_maxValidRow = gt_maxValidCol = pred_occMap[i].shape[1] - 1
                else:
                    gt_minValidRow = gt_validIdxs[0][0].item()
                    gt_maxValidRow = gt_validIdxs[0][-1].item()
                    gt_minValidCol = torch.min(gt_validIdxs[1]).item()      # gt_validIdxs[1][0].item()
                    gt_maxValidCol = torch.max(gt_validIdxs[1]).item()      # gt_validIdxs[1][-1].item()

                min_validRow = min(pred_minValidRow, gt_minValidRow)
                min_validCol = min(pred_minValidCol, gt_minValidCol)
                max_validRow = max(pred_maxValidRow, gt_maxValidRow)
                max_validCol = max(pred_maxValidCol, gt_maxValidCol)

                assert 0 <= min_validRow < pred_occMap[i].size(0)
                assert 0 <= min_validRow < gt_occMap[i].size(0)

                assert 0 <= max_validRow < pred_occMap[i].size(0)
                assert 0 <= max_validRow < gt_occMap[i].size(0)

                assert 0 <= min_validCol < pred_occMap[i].size(1)
                assert 0 <= min_validCol < gt_occMap[i].size(1)

                assert 0 <= max_validCol < pred_occMap[i].size(1)
                assert 0 <= max_validCol < gt_occMap[i].size(1)

                pred_occMap_thisIdx = pred_occMap[i][min_validRow: max_validRow + 1,
                                                     min_validCol: max_validCol + 1, ...]
                gt_occMap_thisIdx = gt_occMap[i][min_validRow: max_validRow + 1,
                                                 min_validCol: max_validCol + 1, ...]

                if exploredPart_mask is not None:
                    assert 0 <= min_validRow < exploredPart_mask[i].shape[0]
                    assert 0 <= max_validRow < exploredPart_mask[i].shape[0]
                    assert 0 <= min_validCol < exploredPart_mask[i].shape[1]
                    assert 0 <= max_validCol < exploredPart_mask[i].shape[1]

                    exploredPart_mask_thisIdx = exploredPart_mask[i][min_validRow: max_validRow + 1,
                                                             min_validCol: max_validCol + 1, ...]
            else:
                pred_occMap_thisIdx = pred_occMap[i]
                gt_occMap_thisIdx = gt_occMap[i]
                if exploredPart_mask is not None:
                    exploredPart_mask_thisIdx = exploredPart_mask[i]

            loss_or_metric = 0.
            for loss_or_metric_type_idx, loss_or_metric_type in enumerate(loss_or_metric_types):
                loss_or_metric_weight = loss_or_metric_weights[loss_or_metric_type_idx]

                if loss_or_metric_type == "f1_score":
                    assert target_category is not None
                    assert mask is not None

                    if exploredPart_mask is None:
                        exploredPart_mask_thisIdx = mask[i].unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    else:
                        exploredPart_mask_thisIdx = exploredPart_mask_thisIdx.unsqueeze(0) *\
                                                    mask[i].unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

                    tp = ((pred_occMap_thisIdx.unsqueeze(0) == target_category).float() *\
                          (gt_occMap_thisIdx.unsqueeze(0) == target_category).float()) * exploredPart_mask_thisIdx
                    tp = tp.view((tp.size(0), -1))
                    tp = torch.sum(tp, dim=1)

                    tp_plus_fp = (pred_occMap_thisIdx.unsqueeze(0) == target_category).float() * exploredPart_mask_thisIdx
                    tp_plus_fp = tp_plus_fp.view((tp_plus_fp.size(0), -1))
                    tp_plus_fp = torch.sum(tp_plus_fp, dim=1)

                    precision = tp / (tp_plus_fp + 1.0e-13)
                    assert torch.prod((precision >= 0.) * (precision <= 1.)).item() == 1

                    tp_plus_fn = (gt_occMap_thisIdx.unsqueeze(0) == target_category).float() * exploredPart_mask_thisIdx
                    tp_plus_fn = tp_plus_fn.view((tp_plus_fn.size(0), -1))
                    tp_plus_fn = torch.sum(tp_plus_fn, dim=1)

                    recall = tp / (tp_plus_fn + 1.0e-13)
                    assert torch.prod((recall >= 0.) * (recall <= 1.)).item() == 1

                    f1_score = 2 * precision * recall / (precision + recall + 1.0e-13)
                    assert torch.prod((f1_score >= 0.) * (f1_score <= 1.)).item() == 1

                    exploredPart_mask_thisIdx_flattened = exploredPart_mask_thisIdx.view((exploredPart_mask_thisIdx.size(0),
                                                                                          -1))
                    validIdxs_exploredPart_mask_thisIdx_flattened = (torch.sum(exploredPart_mask_thisIdx_flattened,
                                                                               dim=1) != 0.).float()
                    f1_score = torch.sum(f1_score, dim=0) / (torch.sum(validIdxs_exploredPart_mask_thisIdx_flattened,
                                                                       dim=0) + 1.0e-13)

                    loss_or_metric += (f1_score * loss_or_metric_weight)
                elif loss_or_metric_type == "iou":
                    assert target_category is not None
                    assert mask is not None
                    
                    if exploredPart_mask is None:
                        exploredPart_mask_thisIdx = mask[i].unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    else:
                        exploredPart_mask_thisIdx = exploredPart_mask_thisIdx.unsqueeze(0) *\
                                                    mask[i].unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

                    tp = ((pred_occMap_thisIdx.unsqueeze(0) == target_category).float() *\
                          (gt_occMap_thisIdx.unsqueeze(0) == target_category).float()) * exploredPart_mask_thisIdx
                    tp = tp.view((tp.size(0), -1))
                    tp = torch.sum(tp, dim=1)

                    tp_union_fp = ((pred_occMap_thisIdx.unsqueeze(0) == target_category).int() |\
                                   (gt_occMap_thisIdx.unsqueeze(0) == target_category).int()).float() *\
                                  exploredPart_mask_thisIdx
                    tp_union_fp = tp_union_fp.view((tp_union_fp.size(0), -1))
                    tp_union_fp = torch.sum(tp_union_fp, dim=1)

                    iou = tp / (tp_union_fp + 1.0e-13)
                    assert torch.prod((iou >= 0.) * (iou <= 1.)).item() == 1

                    exploredPart_mask_thisIdx_flattened = exploredPart_mask_thisIdx.view((exploredPart_mask_thisIdx.size(0),
                                                                                          -1))
                    validIdxs_exploredPart_mask_thisIdx_flattened = (torch.sum(exploredPart_mask_thisIdx_flattened,
                                                                               dim=1) != 0.).float()
                    iou = torch.sum(iou, dim=0) / (torch.sum(validIdxs_exploredPart_mask_thisIdx_flattened,
                                                             dim=0) + 1.0e-13)

                    loss_or_metric += (iou * loss_or_metric_weight)
                else:
                    raise ValueError

            loss_or_metric_all_batch_idxs.append(loss_or_metric.item())

        loss_or_metric = loss_or_metric_all_batch_idxs
    elif dont_collapse_across_batch:
        assert mask is not None
        loss_or_metric_all_batch_idxs = None
        bs = gt_occMap.size(0)
        for loss_or_metric_type_idx, loss_or_metric_type in enumerate(loss_or_metric_types):
            loss_or_metric_weight = loss_or_metric_weights[loss_or_metric_type_idx]

            if loss_or_metric_type == "bce_loss":
                if exploredPart_mask is None:
                    loss_or_metric_currentType = (torch.sum(
                        (-(gt_occMap * mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)) *\
                         torch.log((pred_occMap * mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)) + 1.0e-13) -\
                        (1 - (gt_occMap * mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))) *\
                         torch.log(1 - (pred_occMap * mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)) +
                                   1.0e-13)).view(bs, -1), dim=-1,
                    ) / (
                            torch.sum(mask.view(bs, -1), dim=-1) * np.prod(list(pred_occMap.size())[2:]) + 1.0e-13
                    )) * loss_or_metric_weight
                else:
                    loss_or_metric_currentType = (torch.sum(
                        (((-gt_occMap * torch.log(pred_occMap + 1.0e-13) -\
                        (1 - gt_occMap) * torch.log(1 - pred_occMap + 1.0e-13))) * exploredPart_mask *
                         mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)).view(bs, -1), dim=-1
                    ) / (
                            torch.sum((exploredPart_mask * mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)).view(bs, -1),
                                      dim=-1) + 1.0e-13
                    )) * loss_or_metric_weight
            elif loss_or_metric_type == "f1_score":
                assert target_category is not None
                if exploredPart_mask is None:
                    exploredPart_mask = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                else:
                    exploredPart_mask = exploredPart_mask * mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                tp = ((pred_occMap == target_category).float() * (gt_occMap == target_category).float()) *\
                     exploredPart_mask
                tp = tp.view((bs * tp.size(1), -1))
                tp = torch.sum(tp, dim=1)
                tp = tp.view((bs, -1))

                tp_plus_fp = (pred_occMap == target_category).float() * exploredPart_mask
                tp_plus_fp = tp_plus_fp.view((bs * tp_plus_fp.size(1), -1))
                tp_plus_fp = torch.sum(tp_plus_fp, dim=1)
                tp_plus_fp = tp_plus_fp.view((bs, -1))

                precision = tp / (tp_plus_fp + 1.0e-13)
                assert torch.prod((precision >= 0.) * (precision <= 1.)).item() == 1

                tp_plus_fn = (gt_occMap == target_category).float() * exploredPart_mask
                tp_plus_fn = tp_plus_fn.view((bs * tp_plus_fn.size(1), -1))
                tp_plus_fn = torch.sum(tp_plus_fn, dim=1)
                tp_plus_fn = tp_plus_fn.view((bs, -1))

                recall = tp / (tp_plus_fn + 1.0e-13)
                assert torch.prod((recall >= 0.) * (recall <= 1.)).item() == 1

                f1_score = 2 * precision * recall / (precision + recall + 1.0e-13)
                assert torch.prod((f1_score >= 0.) * (f1_score <= 1.)).item() == 1

                exploredPart_mask_flattened = exploredPart_mask.view((bs * exploredPart_mask.size(1), -1))
                validIdxs_exploredPart_mask_flattened = (torch.sum(exploredPart_mask_flattened, dim=1) != 0.).float()
                validIdxs_exploredPart_mask_flattened = validIdxs_exploredPart_mask_flattened.view((bs, -1,))
                f1_score = torch.sum(f1_score, dim=1) / (torch.sum(validIdxs_exploredPart_mask_flattened, dim=1) + 1.0e-13)

                loss_or_metric_currentType = f1_score * loss_or_metric_weight
            elif loss_or_metric_type == "iou":
                assert target_category is not None

                if exploredPart_mask is None:
                    exploredPart_mask = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                else:
                    exploredPart_mask = exploredPart_mask * mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                tp = ((pred_occMap == target_category).float() * (gt_occMap == target_category).float()) *\
                     exploredPart_mask
                tp = tp.view((bs * tp.size(1), -1))
                tp = torch.sum(tp, dim=1)
                tp = tp.view((bs, -1))

                tp_union_fp = ((pred_occMap == target_category).int() | (gt_occMap == target_category).int()).float() *\
                              exploredPart_mask
                tp_union_fp = tp_union_fp.view((bs * tp_union_fp.size(1), -1))
                tp_union_fp = torch.sum(tp_union_fp, dim=1)
                tp_union_fp = tp_union_fp.view((bs, -1))

                iou = tp / (tp_union_fp + 1.0e-13)
                assert torch.prod((iou >= 0.) * (iou <= 1.)).item() == 1

                exploredPart_mask_flattened = exploredPart_mask.view((bs * exploredPart_mask.size(1), -1))
                validIdxs_exploredPart_mask_flattened = (torch.sum(exploredPart_mask_flattened, dim=1) != 0.).float()
                validIdxs_exploredPart_mask_flattened = validIdxs_exploredPart_mask_flattened.view((bs, -1))
                iou = torch.sum(iou, dim=1) / (torch.sum(validIdxs_exploredPart_mask_flattened, dim=1) + 1.0e-13)

                loss_or_metric_currentType = iou * loss_or_metric_weight
            else:
                raise NotImplementedError

            if loss_or_metric_all_batch_idxs is None:
                loss_or_metric_all_batch_idxs = loss_or_metric_currentType
            else:
                loss_or_metric_all_batch_idxs += loss_or_metric_currentType
        loss_or_metric = loss_or_metric_all_batch_idxs
    else:
        loss_or_metric = 0.
        for loss_or_metric_type_idx, loss_or_metric_type in enumerate(loss_or_metric_types):
            loss_or_metric_weight = loss_or_metric_weights[loss_or_metric_type_idx]

            if loss_or_metric_type == "bce_loss":
                if mask is None:
                    if exploredPart_mask is None:
                        loss_or_metric += (F.binary_cross_entropy(pred_occMap, gt_occMap) * loss_or_metric_weight)
                    else:
                        loss_or_metric += (F.binary_cross_entropy(pred_occMap * exploredPart_mask, gt_occMap *
                                                                  exploredPart_mask) * loss_or_metric_weight)
                else:
                    if exploredPart_mask is None:
                        loss_or_metric += ((torch.sum(
                            -(gt_occMap * mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)) *
                            torch.log((pred_occMap * mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)) + 1.0e-13) -\
                            (1 - (gt_occMap * mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))) *
                            torch.log(1 - (pred_occMap * mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)) + 1.0e-13)
                        ) / (
                                torch.sum(mask) * np.prod(list(pred_occMap.size())[1:]) + 1.0e-13
                        )) * loss_or_metric_weight)
                    else:
                        loss_or_metric += ((torch.sum(
                            (-gt_occMap * torch.log(pred_occMap + 1.0e-13) -\
                            (1 - gt_occMap) * torch.log(1 - pred_occMap + 1.0e-13)) *
                            exploredPart_mask * mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                        ) / (torch.sum(exploredPart_mask * mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)) +
                             1.0e-13)) * loss_or_metric_weight)
            elif loss_or_metric_type == "f1_score":
                assert target_category is not None
                if mask is None:
                    raise NotImplementedError
                else:
                    if exploredPart_mask is None:
                        exploredPart_mask = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    else:
                        exploredPart_mask = exploredPart_mask * mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    tp = ((pred_occMap == target_category).float() * (gt_occMap == target_category).float()) *\
                         exploredPart_mask
                    tp = tp.view((tp.size(0), -1))
                    tp = torch.sum(tp, dim=1)

                    tp_plus_fp = (pred_occMap == target_category).float() * exploredPart_mask
                    tp_plus_fp = tp_plus_fp.view((tp_plus_fp.size(0), -1))
                    tp_plus_fp = torch.sum(tp_plus_fp, dim=1)

                    precision = tp / (tp_plus_fp + 1.0e-13)
                    assert torch.prod((precision >= 0.) * (precision <= 1.)).item() == 1

                    tp_plus_fn = (gt_occMap == target_category).float() * exploredPart_mask
                    tp_plus_fn = tp_plus_fn.view((tp_plus_fn.size(0), -1))
                    tp_plus_fn = torch.sum(tp_plus_fn, dim=1)

                    recall = tp / (tp_plus_fn + 1.0e-13)
                    assert torch.prod((recall >= 0.) * (recall <= 1.)).item() == 1

                    f1_score = 2 * precision * recall / (precision + recall + 1.0e-13)
                    assert torch.prod((f1_score >= 0.) * (f1_score <= 1.)).item() == 1

                    exploredPart_mask_flattened = exploredPart_mask.view((exploredPart_mask.size(0), -1))
                    validIdxs_exploredPart_mask_flattened = (torch.sum(exploredPart_mask_flattened, dim=1) != 0.).float()
                    f1_score = torch.sum(f1_score, dim=0) / (torch.sum(validIdxs_exploredPart_mask_flattened, dim=0)
                                                             + 1.0e-13)

                    loss_or_metric += (f1_score * loss_or_metric_weight)
            elif loss_or_metric_type == "iou":
                assert target_category is not None
                if mask is None:
                    raise NotImplementedError
                else:
                    if exploredPart_mask is None:
                        exploredPart_mask = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    else:
                        exploredPart_mask = exploredPart_mask * mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    tp = ((pred_occMap == target_category).float() * (gt_occMap == target_category).float()) *\
                         exploredPart_mask
                    tp = tp.view((tp.size(0), -1))
                    tp = torch.sum(tp, dim=1)

                    tp_union_fp = ((pred_occMap == target_category).int() | (gt_occMap == target_category).int()).float()\
                                  * exploredPart_mask
                    tp_union_fp = tp_union_fp.view((tp_union_fp.size(0), -1))
                    tp_union_fp = torch.sum(tp_union_fp, dim=1)

                    iou = tp / (tp_union_fp + 1.0e-13)
                    assert torch.prod((iou >= 0.) * (iou <= 1.)).item() == 1

                    exploredPart_mask_flattened = exploredPart_mask.view((exploredPart_mask.size(0), -1))
                    validIdxs_exploredPart_mask_flattened = (torch.sum(exploredPart_mask_flattened, dim=1) != 0.).float()
                    iou = torch.sum(iou, dim=0) / (torch.sum(validIdxs_exploredPart_mask_flattened, dim=0) + 1.0e-13)

                    loss_or_metric += (iou * loss_or_metric_weight)
            else:
                raise NotImplementedError

    return loss_or_metric
