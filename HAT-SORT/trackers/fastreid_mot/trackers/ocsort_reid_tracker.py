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

from .ocsort_utils.association import *


def k_previous_obs(observations, cur_age, k):
    if len(observations) == 0:
        return [-1, -1, -1, -1, -1]
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age-dt]
    max_age = max(observations.keys())
    return observations[max_age]


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h  # scale is just area
    r = w / float(h+1e-6)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if(score == None):
      return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))
    else:
      return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1, 5))


def speed_direction(bbox1, bbox2):
    cx1, cy1 = (bbox1[0]+bbox1[2]) / 2.0, (bbox1[1]+bbox1[3])/2.0
    cx2, cy2 = (bbox2[0]+bbox2[2]) / 2.0, (bbox2[1]+bbox2[3])/2.0
    speed = np.array([cy2-cy1, cx2-cx1])
    norm = np.sqrt((cy2-cy1)**2 + (cx2-cx1)**2) + 1e-6
    return speed / norm


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
    将 [x1, y1, x2, y2] 格式的框转换为 [x, y, w, h] 格式。
    参数:
        boxes: Tensor of shape (N, 4) with [x1, y1, x2, y2]
    返回:
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


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox, delta_t=3, orig=False):
        """
        Initialises a tracker using initial bounding box.

        """
        # define constant velocity model
        if not orig:
          from .ocsort_utils.kalmanfilter import KalmanFilterNew as KalmanFilter
          self.kf = KalmanFilter(dim_x=7, dim_z=4)
        else:
          from filterpy.kalman import KalmanFilter
          self.kf = KalmanFilter(dim_x=7, dim_z=4)
        # F:状态转移矩阵 1的位置表示状态如何传播
        # (x, y, s, r, vx, vy, vs)：
        # x, y: 目标框的中心坐标
        # s: 目标框的面积
        # r: 目标框的长宽比
        # vx, vy, vs: 对应的速度
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [
                            0, 0, 0, 1, 0, 0, 0],  [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        # 观测向量 dim_z=4，即 bbox = (x, y, s, r)，不包含速度 (vx, vy, vs)。
        # H 确保卡尔曼滤波 仅更新 x, y, s, r，而不直接更新速度。
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.    # 测量噪声协方差 R：s 和 r 维度的噪声较大（乘 10），因为这些参数的测量可能不稳定。
        self.kf.P[4:, 4:] *= 1000.  # 速度 (vx, vy, vs) 的初始不确定性设为 1000，表示初始速度未知。
        self.kf.P *= 10.            # 整体误差矩阵 P 乘 10，提高初始不确定性。
        self.kf.Q[-1, -1] *= 0.01   # vx, vy, vs 方向的过程噪声较小（乘 0.01），表明目标运动相对稳定。
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox) # 初始化 bbox 作为卡尔曼滤波器的初始状态。
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        """
        NOTE: [-1,-1,-1,-1,-1] is a compromising placeholder for non-observation status, the same for the return of 
        function k_previous_obs. It is ugly and I do not like it. But to support generate observation array in a 
        fast and unified way, which you would see below k_observations = np.array([k_previous_obs(...]]), let's bear it for now.
        """
        self.last_observation = np.array([-1, -1, -1, -1, -1])  # placeholder
        self.observations = dict()
        self.history_observations = []
        self.velocity = None
        self.delta_t = delta_t

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        if bbox is not None:
            if self.last_observation.sum() >= 0:  # no previous observation
                previous_box = None
                # 尽量找到 delta_t 帧前的 bbox 作为参考，以计算速度，但如果找不到，则使用更近的时间点的 bbox 作为回退方案
                for i in range(self.delta_t):
                    dt = self.delta_t - i
                    if self.age - dt in self.observations:
                        previous_box = self.observations[self.age-dt]
                        break
                if previous_box is None:
                    previous_box = self.last_observation
                """
                  Estimate the track speed direction with observations \Delta t steps away
                """
                self.velocity = speed_direction(previous_box, bbox)

            """
              Insert new observations. This is a ugly way to maintain both self.observations
              and self.history_observations. Bear it for the moment.
            """
            self.last_observation = bbox
            self.observations[self.age] = bbox
            self.history_observations.append(bbox)

            self.time_since_update = 0
            self.history = []
            self.hits += 1
            self.hit_streak += 1
            self.kf.update(convert_bbox_to_z(bbox))
        else:
            self.kf.update(bbox)

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.x[6]+self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0

        self.kf.predict()
        self.age += 1
        if(self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


"""
    We support multiple ways for association cost calculation, by default
    we use IoU. GIoU may have better performance in some situations. We note 
    that we hardly normalize the cost by all methods to (0,1) which may not be 
    the best practice.
"""
ASSO_FUNCS = {  "iou": iou_batch,
                "giou": giou_batch,
                "ciou": ciou_batch,
                "diou": diou_batch,
                "ct_dist": ct_dist}


class OCSortReIDTracker(BaseTracker):
    def __init__(
            self,
            det_thresh: float,
            newborn_thresh: float,
            ema_alpha: float,
            cos_sim_thresh: float,
            disappear_tolerance: int,
            # For Transfer Tracker:
            use_all_zero_filter: bool,
            var_filter_thresh: float,
            use_transfer: bool,
            transferred_similarity_alpha: float,
            transfer_hist_len: int,
            transfer_factor_thresh: float,
            use_normalize: bool,
            use_standard_scaler: str,
            use_pca: str,
            direct_inter_class_diff: bool,
            use_related_inter_class_diff: bool,
            use_weighted_class_mean: bool,
            use_weighted_inter_class_diff: bool,
            weighted_class_mean_alpha: float,
            use_weighted_cov: bool,
            use_sample_average: bool,
            # History strategies:
            history_queue_type: str,
            history_score_decay_ratio: float,
            history_weight_decay_ratio: float,
            history_use_decay_as_weight: bool,
            history_weight_min: float,
            history_use_score_as_weight: bool,
            #
            fit_interval: int,
            #
            iou_match_thresh: float,
            max_age=30,
            min_hits=3,
            iou_threshold=0.3,
            delta_t=3,
            asso_func="iou",
            inertia=0.2,
            use_byte=False,
            reid_matrix_alpha=0.9,
    ):
        super(OCSortReIDTracker, self).__init__()
        """
        Sets key parameters for SORT
        """
        self.iou_match_thresh = iou_match_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = {}
        self.frame_count = 0
        self.det_thresh = det_thresh
        self.delta_t = delta_t
        self.asso_func = ASSO_FUNCS[asso_func]
        self.inertia = inertia
        self.use_byte = use_byte
        self.reid_matrix_alpha = reid_matrix_alpha
        KalmanBoxTracker.count = 0

        if newborn_thresh > 0.0:
            self.newborn_thresh = newborn_thresh
        else:
            self.newborn_thresh = det_thresh
        self.ema_alpha = ema_alpha
        self.cos_sim_thresh = cos_sim_thresh
        self.disappear_tolerance = disappear_tolerance
        # For Transfer Tracker:
        self.use_all_zero_filter = use_all_zero_filter
        self.var_filter_thresh = var_filter_thresh
        self.use_transfer = use_transfer
        self.transferred_similarity_alpha = transferred_similarity_alpha
        self.transfer_hist_len = transfer_hist_len
        self.transfer_factor_thresh = transfer_factor_thresh
        self.use_normalize = use_normalize
        self.use_standard_scaler = use_standard_scaler
        self.use_pca = use_pca
        self.direct_inter_class_diff = direct_inter_class_diff
        self.use_related_inter_class_diff = use_related_inter_class_diff
        self.use_weighted_class_mean = use_weighted_class_mean
        self.use_weighted_inter_class_diff = use_weighted_inter_class_diff
        self.weighted_class_mean_alpha = weighted_class_mean_alpha
        self.use_weighted_cov = use_weighted_cov
        self.use_sample_average = use_sample_average
        # History strategies:
        self.history_queue_type = history_queue_type
        self.history_score_decay_ratio = history_score_decay_ratio
        self.history_weight_decay_ratio = history_weight_decay_ratio
        self.history_use_decay_as_weight = history_use_decay_as_weight
        self.history_weight_min = history_weight_min
        self.history_use_score_as_weight = history_use_score_as_weight
        #
        self.fit_interval = fit_interval

        self.ema_object_features = {}
        self.disappear_frames = {}
        self.transfer_histories = {}
        self.next_id = 0

        self.lda_model = LDA(
            use_shrinkage=True,
            use_normalize=self.use_normalize,
            use_standard_scaler=self.use_standard_scaler,
            use_pca=self.use_pca,
            direct_inter_class_diff=self.direct_inter_class_diff,
            use_weighted_class_mean=self.use_weighted_class_mean,
            weighted_class_mean_alpha=self.weighted_class_mean_alpha,
            use_weighted_cov=self.use_weighted_cov,
            use_sample_average=self.use_sample_average,
            weight_min=self.history_weight_min,
        )
        return

    def update(self, output_results, img_info, img_size, frame_idx):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        if output_results is None:
            return np.empty((0, 5))
        if not isinstance(output_results, np.ndarray):
            output_results = output_results.cpu().numpy()
        self.frame_count += 1
        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2

        # Rescale
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale

        dets = np.concatenate((bboxes, np.expand_dims(scores, axis=-1)), axis=1)
        remain_inds = scores > self.det_thresh
        dets = dets[remain_inds]

        # Generate embeddings
        dets_embs = np.ones((dets.shape[0], 1))
        if dets.shape[0] != 0:
            # Shape = (num detections, 3, 512) if grid
            dets_embs = self.embedder.compute_embedding(img_numpy, dets[:, :4], tag)

        boxes = xyxy_to_xywh(torch.from_numpy(dets[:, :4]))  # boxes in xywh format
        scores = torch.from_numpy(dets[:, 4])
        features = torch.from_numpy(dets_embs)

        # 1. calculate similarity matrix and assign IDs for detections:
        hist_ids = list(self.ema_object_features.keys())
        if len(hist_ids) == 0:  # no history trajectories
            curr_ids = [self.next_id + i for i in range(len(boxes))]
            self.next_id += len(boxes)
        elif len(boxes) == 0:  # no current detections
            curr_ids = []
        else:  # normal case
            # get predicted locations from existing trackers.
            trks = np.zeros((len(self.trackers), 5))
            to_del = []
            ret = []
            for i, hist_id in enumerate(hist_ids):
                pos = self.trackers[hist_id].predict()[0]
                trks[i, :] = [pos[0], pos[1], pos[2], pos[3], 0]
                if np.any(np.isnan(pos)):
                    to_del.append(hist_id)
            trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
            for t in reversed(to_del):
                del self.trackers[t]
                del self.ema_object_features[t]
                del self.disappear_frames[t]
                del self.transfer_histories[t]

            hist_ids = list(self.ema_object_features.keys())
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
            if do_transfer:
                # OK to do LDA
                # transfer_hist_features = torch.stack(transfer_hist_features)
                # transfer_hist_ids = torch.tensor(transfer_hist_ids, dtype=torch.int64)
                # transfer_hist_scores = torch.stack(transfer_hist_scores)
                transfer_hist_features = torch.cat(transfer_hist_features, dim=0)
                transfer_hist_ids = torch.tensor(transfer_hist_ids, dtype=torch.int64)
                transfer_hist_scores = torch.cat(transfer_hist_scores, dim=0)
                all_features = torch.cat([transfer_hist_features, curr_features])
                # There are too many dimensions, so we need to reduce the dimensionality.
                dim_filter = torch.zeros(all_features.shape[1], dtype=torch.bool)  # init the dim filter
                if self.use_all_zero_filter:
                    dim_filter = dim_filter | (all_features == 0).all(dim=0)
                if self.var_filter_thresh > 0.0:
                    feature_vars = torch.var(all_features, dim=0)
                    dim_filter = dim_filter | (
                                (feature_vars / torch.abs(all_features.mean(dim=0))) < self.var_filter_thresh)
                transfer_hist_features = transfer_hist_features[:, ~dim_filter]

                if frame_idx % self.fit_interval == 0 or not self.lda_model.is_fit():
                    self.lda_model.fit(X=transfer_hist_features, y=transfer_hist_ids, score=transfer_hist_scores)

                untransferred_hist_features = hist_features.clone()
                untransferred_curr_features = curr_features.clone()
                hist_features = hist_features[:, ~dim_filter]
                curr_features = curr_features[:, ~dim_filter]
                hist_features = self.lda_model.transform(hist_features)
                curr_features = self.lda_model.transform(curr_features)
                pass
            else:
                pass
            # calculate cosine similarity matrix:
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

            # ==================================== Ocsort - Kalman stage ===================================
            velocities = np.array(
                [trk.velocity if trk.velocity is not None else np.array((0, 0)) for trk in self.trackers.values()])
            last_boxes = np.array([trk.last_observation for trk in self.trackers.values()])
            k_observations = np.array(
                [k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers.values()])

            """
                First round of association
            """
            # matched, unmatched_dets, unmatched_trks,
            iou_matrix, angle_diff_cost = associate_with_score(
                dets, trks, self.iou_threshold, velocities, k_observations, self.inertia)
            # for m in matched:
            #     self.trackers[m[1]].update(dets[m[0], :])

            # Hungarian matching:
            row_idxs, col_idxs = linear_sum_assignment(-(self.reid_matrix_alpha * cos_sim_matrix +
                                                         (1 - self.reid_matrix_alpha) * (iou_matrix + angle_diff_cost)))
            curr_ids = [-1] * len(boxes)

            assert len(curr_ids) == len(dets)
            unmatched_dets = []
            for d, det in enumerate(dets):
                if (d not in row_idxs):
                    unmatched_dets.append(d)
            unmatched_trks = []
            for t, trk in enumerate(trks):
                if (t not in col_idxs):
                    unmatched_trks.append(t)

            for curr_idx, hist_idx in zip(row_idxs, col_idxs):
                if (cos_sim_matrix[curr_idx, hist_idx] > self.cos_sim_thresh or iou_matrix[curr_idx, hist_idx]
                        > self.iou_match_thresh):
                    curr_id = hist_ids[hist_idx]
                    curr_ids[curr_idx] = curr_id

                    self.ema_object_features[curr_id] = (self.ema_alpha * self.ema_object_features[curr_id] +
                                                         (1 - self.ema_alpha) * features[curr_idx])
                    self.disappear_frames[curr_id] = 0
                    # Transfer: update the history feature queue:
                    self.transfer_histories[curr_id].add(feature=features[curr_idx], score=scores[curr_idx])
                    self.trackers[curr_id].update(dets[curr_idx, :])

                elif scores[curr_idx] > self.newborn_thresh:
                    curr_ids[curr_idx] = -1
                    unmatched_dets.append(curr_idx)
                    unmatched_trks.append(hist_idx)
                else:
                    curr_ids[curr_idx] = -2
                    unmatched_dets.append(curr_idx)
                    unmatched_trks.append(hist_idx)
            pass
            unmatched_dets = np.array(unmatched_dets)
            unmatched_trks = np.array(unmatched_trks)

            """
                Second round of associaton by OCR
            """
            if self.use_byte and len(dets_second) > 0 and unmatched_trks.shape[0] > 0:
                u_trks = trks[unmatched_trks]
                iou_left = self.asso_func(dets_second, u_trks)  # iou between low score detections and unmatched tracks
                iou_left = np.array(iou_left)
                if iou_left.max() > self.iou_threshold:
                    """
                        NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                        get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                        uniform here for simplicity
                    """
                    matched_indices = linear_assignment(-iou_left)
                    to_remove_trk_indices = []
                    for m in matched_indices:
                        det_ind, trk_ind = m[0], unmatched_trks[m[1]]
                        if iou_left[m[0], m[1]] < self.iou_threshold:
                            continue
                        self.trackers[hist_ids[trk_ind]].update(dets_second[det_ind, :])
                        curr_ids_second[det_ind] = hist_ids[trk_ind]
                        self.disappear_frames[hist_ids[trk_ind]] = 0
                        # to_remove_trk_indices.append(hist_ids[trk_ind])
                        to_remove_trk_indices.append(unmatched_trks[m[1]])
                    unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

            if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
                left_dets = dets[unmatched_dets]
                left_trks = last_boxes[unmatched_trks]
                iou_left = self.asso_func(left_dets, left_trks)
                iou_left = np.array(iou_left)
                if iou_left.max() > self.iou_threshold:
                    """
                        NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                        get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                        uniform here for simplicity
                    """
                    rematched_indices = linear_assignment(-iou_left)
                    to_remove_det_indices = []
                    to_remove_trk_indices = []
                    for m in rematched_indices:
                        det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                        if iou_left[m[0], m[1]] < self.iou_threshold:
                            continue
                        self.trackers[hist_ids[trk_ind]].update(dets[det_ind, :])
                        curr_ids[det_ind] = hist_ids[trk_ind]
                        self.disappear_frames[hist_ids[trk_ind]] = 0
                        to_remove_det_indices.append(det_ind)
                        to_remove_trk_indices.append(hist_ids[trk_ind])
                    unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                    unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        # 2. update the states in the tracker:
        for i, curr_id in enumerate(curr_ids):
            if curr_id == -2:  # not assigned as newborn, because of low confidence
                continue
            # if curr_id == -1:
            #     pass
            if curr_id not in hist_ids:  # add new trajectories
                curr_id = self.next_id
                self.next_id += 1
                curr_ids[i] = curr_id
                self.ema_object_features[curr_id] = features[i]
                self.disappear_frames[curr_id] = 0
                # Transfer: new the history feature queue:
                # self.transfer_histories[curr_id] = FIFOQueue(max_len=self.transfer_hist_len)
                # self.transfer_histories[curr_id] = ScoreQueue(max_len=self.transfer_hist_len, decay_ratio=0.99)
                match self.history_queue_type:
                    case "FIFOQueue":
                        self.transfer_histories[curr_id] = FIFOQueue(
                            max_len=self.transfer_hist_len,
                            weight_decay_ratio=self.history_weight_decay_ratio,
                            use_decay_as_weight=self.history_use_decay_as_weight,
                            use_score_as_weight=self.history_use_score_as_weight,
                        )
                    case "ScoreQueue":
                        self.transfer_histories[curr_id] = ScoreQueue(
                            max_len=self.transfer_hist_len,
                            score_decay_ratio=self.history_score_decay_ratio,
                        )
                    case _:
                        raise ValueError(f"Unknown history queue type: {self.history_queue_type}")
                self.transfer_histories[curr_id].add(feature=features[i], score=scores[i])
                # add new kalman filter
                trk = KalmanBoxTracker(dets[i, :], delta_t=self.delta_t)
                self.trackers[curr_id] = trk

            # else:  # update existing trajectories
            #     self.ema_object_features[curr_id] = (self.ema_alpha * self.ema_object_features[curr_id] +
            #                                          (1 - self.ema_alpha) * features[i])
            #     self.disappear_frames[curr_id] = 0
            #     # Transfer: update the history feature queue:
            #     self.transfer_histories[curr_id].add(feature=features[i], score=scores[i])
            #     self.trackers[curr_id].update(dets[i, :])

        for hist_id in hist_ids:
            if hist_id not in curr_ids:
                self.disappear_frames[hist_id] += 1
                self.trackers[hist_id].update(None)

        # 3. remove trajectories that have disappeared for too long:
        for hist_id in hist_ids:
            if self.disappear_frames[hist_id] > self.disappear_tolerance:
                del self.ema_object_features[hist_id]
                del self.disappear_frames[hist_id]
                del self.transfer_histories[hist_id]
                del self.trackers[hist_id]

        # 4. return the results:
        ret = []
        for i, curr_id in enumerate(curr_ids):
            if curr_id in hist_ids:
                ret.append(np.concatenate((dets[i, :4], [curr_id])).reshape(1, -1))
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))
