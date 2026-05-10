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


class ReIDTracker:
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(
            self,
            feature,
            score,
            frame_idx,
            ema_alpha,
            box,
            history_queue_type,
            transfer_hist_len,
            history_weight_decay_ratio
    ):
        # ID Assign
        self.id = ReIDTracker.count
        ReIDTracker.count += 1

        self.disappear_frame = 0
        self.last_update_time = frame_idx
        self.last_observation = box

        self.ema_alpha=ema_alpha
        self.ema_object_feature = feature

        match history_queue_type:
            case "FIFOQueue":
                self.transfer_history= FIFOQueue(
                    max_len=transfer_hist_len,
                    weight_decay_ratio=history_weight_decay_ratio,
                )
            case "ScoreQueue":
                self.transfer_history = ScoreQueue(
                    max_len=transfer_hist_len,
                )
            case _:
                raise ValueError(f"Unknown history queue type: {history_queue_type}")
        self.transfer_history.add(feature=feature, score=score)

    def update(self, feature=None, score=None, box=None, frame_idx=None):
        if feature is None:
            self.disappear_frame += 1
        else:
            self.ema_object_feature = (self.ema_alpha * self.ema_object_feature + (1 - self.ema_alpha) * feature)
            self.disappear_frame = 0
            self.last_update_time = frame_idx
            self.last_observation = box
            # update the history feature queue:
            self.transfer_history.add(feature=feature, score=score)


class EmaHatReID(BaseTracker):
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
        super(EmaHatReID, self).__init__()
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

        self.trackers = []

        self.lda_model = LDA(
            use_shrinkage=True,
            use_standard_scaler=self.use_standard_scaler,
            use_sample_average=self.use_sample_average,
            weight_min=self.history_weight_min,
        )
        ReIDTracker.count = 0


    def transfer_feature(
            self,
            curr_features,
            hist_features,
            transfer_hist_features,
            transfer_hist_ids,
            transfer_hist_scores
    ):
        """
           Perform feature transfer using LDA if conditions are met.
           Returns transformed curr_features, hist_features.
           """
        # Concatenate transfer data
        if curr_features is None or hist_features is None or len(curr_features)==0 or len(hist_features)==0:
            return curr_features, hist_features

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
        hist_features = hist_features[:, ~dim_filter]
        curr_features = curr_features[:, ~dim_filter]
        hist_features = self.lda_model.transform(hist_features)
        curr_features = self.lda_model.transform(curr_features)

        return curr_features, hist_features

    def associate(
            self,
            curr_features,
            hist_features,
            untransferred_curr_features=None,
            untransferred_hist_features=None,
            do_transfer=False
    ):
        if curr_features is None or hist_features is None or len(curr_features)==0 or len(hist_features)==0:
            return np.empty((0, 2), dtype=int), np.arange(len(curr_features)), np.empty((0, 5), dtype=int)

        # Normalize and Compute cosine similarity matrix between current and history features.
        cos_sim_matrix = (
                functional.normalize(curr_features, p=2, dim=-1) @ functional.normalize(hist_features, p=2,dim=-1).T)

        # If transfer is applied and alpha < 1, mix with untransferred similarity.
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

        # Hungarian matching:
        matched_indices = linear_sum_assignment(-cos_sim_matrix)
        matched_indices = np.stack(matched_indices, axis=1)

        unmatched_detections = []
        for d, det in enumerate(curr_features):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
        unmatched_trackers = []
        for t, trk in enumerate(hist_features):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)

        # filter out matched with low IOU
        matches = []
        for m in matched_indices:
            if cos_sim_matrix[m[0], m[1]] < self.cos_sim_thresh:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


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

        scores = torch.from_numpy(dets[:, 4]).float()
        features = torch.from_numpy(id_feature_keep).float()

        hist_features = [trk.ema_object_feature.clone() for trk in self.trackers if trk.ema_object_feature is not None]
        if hist_features:
            hist_features = torch.stack(hist_features)
        else:
            hist_features = None
        curr_features = features.clone()

        ret = []

        """
        1. calculate similarity matrix and assign IDs for detections:
        """
        # Transfer: project the features into a new space:
        transfer_hist_ids, transfer_hist_features, transfer_hist_scores = [], [], []
        for trk in self.trackers:
            _features, _scores = trk.transfer_history.get()
            _ids = [trk.id] * len(_features)
            transfer_hist_ids += _ids
            transfer_hist_features.append(_features)
            transfer_hist_scores.append(_scores)

        # Decide whether to do transfer:
        do_transfer = (self.use_transfer
                       and (len(transfer_hist_ids) > self.transfer_factor_thresh * len(self.trackers)))

        untransferred_curr_features = curr_features.clone()
        untransferred_hist_features = hist_features.clone() if hist_features is not None else None

        if do_transfer:
            curr_features, hist_features = (
                self.transfer_feature(curr_features=curr_features, hist_features=hist_features,
                                      transfer_hist_features=transfer_hist_features,
                                      transfer_hist_ids=transfer_hist_ids,
                                      transfer_hist_scores=transfer_hist_scores))

        # Hungarian matching:
        matched, unmatched_dets, unmatched_trks = (
            self.associate(curr_features=curr_features, hist_features=hist_features,
                           untransferred_curr_features=untransferred_curr_features,
                           untransferred_hist_features=untransferred_hist_features,do_transfer=do_transfer))

        for m in matched:
            self.trackers[m[1]].update(
                feature=features[m[0]], score=scores[m[0]], frame_idx=frame_idx, box=dets[m[0], :4])

        """
        2. update the states in the tracker:
        """
        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            if scores[i] > self.newborn_thresh:
                trk = ReIDTracker(feature=features[i], score=scores[i], frame_idx=frame_idx, box=dets[i, :4],
                                  ema_alpha=self.ema_alpha,
                                  history_queue_type=self.history_queue_type,
                                  transfer_hist_len=self.transfer_hist_len,
                                  history_weight_decay_ratio=self.history_weight_decay_ratio)
                self.trackers.append(trk)

        # update disappear_frame for unmatched_trks
        for t in unmatched_trks:
            self.trackers[t].update()

        """
        3. get return value and remove trajectories that have disappeared for too long
        """
        for t, trk in enumerate(self.trackers):
            if trk.last_update_time == frame_idx:
                ret.append(np.concatenate((trk.last_observation[:4], [trk.id+1])).reshape(1, -1))
            if trk.disappear_frame > self.disappear_tolerance:
                self.trackers.pop(t)

        """
        4. return the results
        """
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))