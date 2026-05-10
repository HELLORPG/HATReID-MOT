# Copyright (c) Ruopeng Gao. All Rights Reserved.
# About:
import torch
from collections import defaultdict, deque
import torch.nn.functional as functional

from .base_tracker import BaseTracker
from scipy.optimize import linear_sum_assignment
from ..models.queue import ScoreQueue
from ..models.queue.fast_fifo_queue import FIFOQueue
from ..models.lda.lda import LDA

import numpy as np


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x, y, w, h = boxes.unbind(-1)
    boxes = [
        x,
        y,
        x + w,
        y + h
    ]
    return torch.stack(boxes, dim=-1)


import torch


def xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    """
        convert boxes: Tensor of shape (N, 4) with [x1, y1, x2, y2] to
        Tensor of shape (N, 4) with [x, y, w, h]
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    x = x1
    y = y1
    w = x2 - x1
    h = y2 - y1

    return torch.stack([x, y, w, h], dim=1)



class EMAReIDV0Tracker(BaseTracker):
    def __init__(
            self,
            args,
            det_thresh: float,
            newborn_thresh: float,
            ema_alpha: float,
            cos_sim_thresh: float,
            disappear_tolerance: int,
            # For Transfer Tracker:
            use_all_zero_filter: bool,
            use_transfer: bool,
            transferred_similarity_alpha: float,
            transfer_hist_len: int,
            transfer_factor_thresh: float,
            use_standard_scaler: str,
            use_sample_average: bool,
            # History strategies:
            history_queue_type: str,
            history_weight_decay_ratio: float,
            history_weight_min: float,
    ):
        super(EMAReIDV0Tracker, self).__init__()
        self.args = args
        self.det_thresh = det_thresh
        if newborn_thresh > 0.0:
            self.newborn_thresh = newborn_thresh
        else:
            self.newborn_thresh = det_thresh
        self.ema_alpha = ema_alpha
        self.cos_sim_thresh = cos_sim_thresh
        self.disappear_tolerance = disappear_tolerance
        # Transfer strategies:
        self.use_all_zero_filter = use_all_zero_filter
        self.use_transfer = use_transfer
        self.transferred_similarity_alpha = transferred_similarity_alpha
        self.transfer_hist_len = transfer_hist_len
        self.transfer_factor_thresh = transfer_factor_thresh
        self.use_standard_scaler = use_standard_scaler
        self.use_sample_average = use_sample_average
        # History strategies:
        self.history_queue_type = history_queue_type
        self.history_weight_decay_ratio = history_weight_decay_ratio
        self.history_weight_min = history_weight_min
        #
        self.ema_object_features = {}
        self.disappear_frames = {}
        self.transfer_histories = {}
        self.next_id = 0

        self.lda_model = LDA(
            use_shrinkage=True,
            use_standard_scaler=self.use_standard_scaler,
            use_sample_average=self.use_sample_average,
            weight_min=self.history_weight_min,
        )
        return


    def transfer_feature(self, curr_features, hist_features, transfer_hist_features, transfer_hist_ids, transfer_hist_scores):
        """
           Perform feature transfer using LDA if conditions are met.
           Returns transformed curr_features, hist_features.
           """
        # Concatenate transfer data
        transfer_hist_features = torch.cat(transfer_hist_features, dim=0)
        transfer_hist_ids = torch.tensor(transfer_hist_ids, dtype=torch.int64)
        transfer_hist_scores = torch.cat(transfer_hist_scores, dim=0)
        all_features = torch.cat([transfer_hist_features, curr_features])

        # There are too many dimensions, so we need to reduce the dimensionality.
        dim_filter = torch.zeros(all_features.shape[1], dtype=torch.bool)  # init the dim filter
        if self.use_all_zero_filter:
            dim_filter = dim_filter | (all_features == 0).all(dim=0)
        transfer_hist_features = transfer_hist_features[:, ~dim_filter]

        # Fit LDA model
        self.lda_model.fit(X=transfer_hist_features, y=transfer_hist_ids, score=transfer_hist_scores)

        # Transform current and historical features
        untransferred_hist_features = hist_features.clone()
        untransferred_curr_features = curr_features.clone()
        hist_features = hist_features[:, ~dim_filter]
        curr_features = curr_features[:, ~dim_filter]
        hist_features = self.lda_model.transform(hist_features)
        curr_features = self.lda_model.transform(curr_features)

        return curr_features, hist_features


    def compute_cosine_similarity(self, curr_features, hist_features,
                                  untransferred_curr_features=None, untransferred_hist_features=None, do_transfer=False):
        """
        Compute cosine similarity matrix between current and history features.
        If transfer is applied and alpha < 1, mix with untransferred similarity.
        """
        # Normalize and compute cosine similarity
        cos_sim_matrix = functional.normalize(curr_features, p=2, dim=-1) @ functional.normalize(hist_features, p=2,
                                                                                                 dim=-1).T
        if do_transfer and self.transferred_similarity_alpha < 1.0:
            untransferred_cos_sim_matrix = functional.normalize(untransferred_curr_features, p=2,
                                                                dim=-1) @ functional.normalize(
                untransferred_hist_features, p=2, dim=-1).T
            untransferred_cos_sim_matrix = untransferred_cos_sim_matrix.to(device=cos_sim_matrix.device,
                                                                           dtype=cos_sim_matrix.dtype)
            cos_sim_matrix = self.transferred_similarity_alpha * cos_sim_matrix + (
                    1 - self.transferred_similarity_alpha) * untransferred_cos_sim_matrix
        cos_sim_matrix = (cos_sim_matrix + 1) / 2  # normalize to [0, 1]
        cos_sim_matrix = cos_sim_matrix.cpu()

        return cos_sim_matrix


    def update(self, output_results, img_info, img_size, frame_idx, id_feature=None):

        if output_results is None:
            return np.empty((0, 5))

        # post_process detections
        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale

        dets = np.concatenate((bboxes, np.expand_dims(scores, axis=-1)), axis=1)
        remain_inds = scores > self.det_thresh
        dets = dets[remain_inds]
        id_feature_keep = id_feature[remain_inds]  # ID feature of 1st stage matching

        boxes = xyxy_to_xywh(torch.from_numpy(dets[:, :4]))  # boxes in xywh format
        scores = torch.from_numpy(dets[:, 4])
        features = torch.from_numpy(id_feature_keep)
        pass

        # 1. calculate similarity matrix and assign IDs for detections:
        hist_ids = list(self.ema_object_features.keys())
        if len(hist_ids) == 0:  # no history trajectories
            curr_ids = [self.next_id + i for i in range(len(boxes))]
            self.next_id += len(boxes)
        elif len(boxes) == 0:   # no current detections
            curr_ids = []
        else:                   # normal case
            hist_features = torch.stack([self.ema_object_features[hist_id].clone() for hist_id in hist_ids])
            curr_features = features.clone()

            # Transfer: project the features into a new space:
            transfer_hist_ids, transfer_hist_features, transfer_hist_scores = [], [], []
            for hist_id in hist_ids:
                _features, _scores = self.transfer_histories[hist_id].get()
                _ids = [hist_id] * len(_features)
                transfer_hist_ids += _ids
                transfer_hist_features.append(_features)
                transfer_hist_scores.append(_scores)

            # Decide whether to do transfer:
            do_transfer = (self.use_transfer
                           and (len(transfer_hist_ids) > self.transfer_factor_thresh * len(hist_ids)))

            untransferred_curr_features = curr_features.clone()
            untransferred_hist_features = hist_features.clone()

            if do_transfer:
                curr_features, hist_features = self.transfer_feature(curr_features=curr_features, hist_features=hist_features,
                                          transfer_hist_features=transfer_hist_features,
                                          transfer_hist_ids=transfer_hist_ids,
                                          transfer_hist_scores=transfer_hist_scores)

            # calculate cosine similarity matrix:
            cos_sim_matrix = self.compute_cosine_similarity(curr_features=curr_features, hist_features=hist_features,
                                                            untransferred_curr_features=untransferred_curr_features,
                                                            untransferred_hist_features=untransferred_hist_features,
                                                            do_transfer=do_transfer)

            # Hungarian matching:
            row_idxs, col_idxs = linear_sum_assignment(-cos_sim_matrix)
            curr_ids = [-1] * len(boxes)
            for curr_idx, hist_idx in zip(row_idxs, col_idxs):
                if cos_sim_matrix[curr_idx, hist_idx] > self.cos_sim_thresh:
                    curr_ids[curr_idx] = hist_ids[hist_idx]
                elif scores[curr_idx] > self.newborn_thresh:
                    curr_ids[curr_idx] = -1
                else:
                    curr_ids[curr_idx] = -2
            pass

        # 2. update the states in the tracker:
        for i, curr_id in enumerate(curr_ids):
            if curr_id == -2:       # not assigned as newborn, because of low confidence
                continue
            # if curr_id == -1:
            #     pass
            if curr_id not in hist_ids:     # add new trajectories
                curr_id = self.next_id
                self.next_id += 1
                curr_ids[i] = curr_id
                self.ema_object_features[curr_id] = features[i]
                self.disappear_frames[curr_id] = 0
                match self.history_queue_type:
                    case "FIFOQueue": self.transfer_histories[curr_id] = FIFOQueue(
                        max_len=self.transfer_hist_len,
                        weight_decay_ratio=self.history_weight_decay_ratio,
                    )
                    case "ScoreQueue": self.transfer_histories[curr_id] = ScoreQueue(
                        max_len=self.transfer_hist_len,
                    )
                    case _: raise ValueError(f"Unknown history queue type: {self.history_queue_type}")
                self.transfer_histories[curr_id].add(feature=features[i], score=scores[i])
            else:                           # update existing trajectories
                self.ema_object_features[curr_id] = (self.ema_alpha * self.ema_object_features[curr_id] +
                                                     (1 - self.ema_alpha) * features[i])
                self.disappear_frames[curr_id] = 0
                # Transfer: update the history feature queue:
                self.transfer_histories[curr_id].add(feature=features[i], score=scores[i])
        for hist_id in hist_ids:
            if hist_id not in curr_ids:
                self.disappear_frames[hist_id] += 1

        # 3. remove trajectories that have disappeared for too long:
        for hist_id in hist_ids:
            if self.disappear_frames[hist_id] > self.disappear_tolerance:
                del self.ema_object_features[hist_id]
                del self.disappear_frames[hist_id]
                del self.transfer_histories[hist_id]

        # 4. return the results:
        curr_ids_np = np.array(curr_ids)
        matched_mask = curr_ids_np >= 0
        curr_matched_ids = curr_ids_np[matched_mask]
        curr_matched_boxes = dets[matched_mask, :4]

        ret = []
        for box, id in zip(curr_matched_boxes, curr_matched_ids):
            ret.append(np.concatenate((box, [id])).reshape(1, -1))
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))