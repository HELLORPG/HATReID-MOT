"""
Author: Ruopeng Gao
Licensed: Apache-2.0 License
"""

from typing import List, Tuple

import torch
import torch.nn.functional as F
from mmdet.models.trackers.base_tracker import BaseTracker
from mmdet.registry import MODELS
from mmdet.structures import TrackDataSample
from mmdet.structures.bbox import bbox_overlaps
from mmengine.structures import InstanceData
from torch import Tensor
from scipy.optimize import linear_sum_assignment

from collections import deque
from transfer.queue import FIFOQueue, ScoreQueue


@MODELS.register_module()
class MasaTaoTransferTracker(BaseTracker):
    """Tracker for MASA on TAO benchmark.

    Args:
        init_score_thr (float): The cls_score threshold to
            initialize a new tracklet. Defaults to 0.8.
        obj_score_thr (float): The cls_score threshold to
            update a tracked tracklet. Defaults to 0.5.
        match_score_thr (float): The match threshold. Defaults to 0.5.
        memo_tracklet_frames (int): The most frames in a tracklet memory.
            Defaults to 10.
        memo_momentum (float): The momentum value for embeds updating.
            Defaults to 0.8.
        distractor_score_thr (float): The score threshold to consider an object as a distractor.
            Defaults to 0.5.
        distractor_nms_thr (float): The NMS threshold for filtering out distractors.
            Defaults to 0.3.
        with_cats (bool): Whether to track with the same category.
            Defaults to True.
        match_metric (str): The match metric. Can be 'bisoftmax', 'softmax', or 'cosine'. Defaults to 'bisoftmax'.
        max_distance (float): Maximum distance for considering matches. Defaults to -1.
        fps (int): Frames per second of the input video. Used for calculating growth factor. Defaults to 1.
    """

    def __init__(
        self,
        init_score_thr: float = 0.8,
        obj_score_thr: float = 0.5,
        match_score_thr: float = 0.5,
        memo_tracklet_frames: int = 10,
        memo_momentum: float = 0.8,
        distractor_score_thr: float = 0.5,
        distractor_nms_thr=0.3,
        with_cats: bool = True,
        max_distance: float = -1,
        fps=1,
        similarity_function: str = "cosine",
        assignment_protocol: str = "hungarian",
        # For Transfer Tracker:
        use_transfer: bool = True,      # default True
        transfer_hist_len: int = 60,
        transfer_factor_thr: float = 4.0,
        transfer_similarity_alpha: float = 1.0,
        use_standard_scaler: bool = False,
        direct_inter_class_diff: bool = True,
        use_related_inter_class_diff: bool = False,
        use_weighted_class_mean: bool = True,
        use_weighted_inter_class_diff: bool = False,
        weighted_class_mean_alpha: float=1.0,
        transfer_dtype=torch.float32,
        # History queue:
        history_queue_type: str = "FIFOQueue",
        history_use_decay_as_weight: bool = True,
        history_weight_decay_ratio: float = 0.9,
        **kwargs
    ):
        super().__init__(**kwargs)
        assert 0 <= memo_momentum <= 1.0
        assert memo_tracklet_frames >= 0
        self.init_score_thr = init_score_thr
        self.obj_score_thr = obj_score_thr
        self.match_score_thr = match_score_thr
        self.memo_tracklet_frames = memo_tracklet_frames
        self.memo_momentum = memo_momentum
        self.distractor_score_thr = distractor_score_thr
        self.distractor_nms_thr = distractor_nms_thr
        self.with_cats = with_cats
        self.similarity_function = similarity_function
        self.assignment_protocol = assignment_protocol
        # For Transfer Tracker:
        self.use_transfer = use_transfer
        self.transfer_hist_len = transfer_hist_len
        self.transfer_factor_thr = transfer_factor_thr
        self.transfer_similarity_alpha = transfer_similarity_alpha
        self.use_standard_scaler = use_standard_scaler
        self.direct_inter_class_diff = direct_inter_class_diff
        self.use_weighted_class_mean = use_weighted_class_mean
        self.use_related_inter_class_diff = use_related_inter_class_diff
        self.use_weighted_inter_class_diff = use_weighted_inter_class_diff
        self.weighted_class_mean_alpha = weighted_class_mean_alpha
        if isinstance(transfer_dtype, str):
            match transfer_dtype:
                case "torch.float32": self.transfer_dtype = torch.float32
                case "torch.float64": self.transfer_dtype = torch.float64
                case _: raise ValueError(f"Unsupported transfer_dtype: {transfer_dtype}")
        else:
            self.transfer_dtype = transfer_dtype
        # History queue:
        self.history_queue_type = history_queue_type
        self.history_use_decay_as_weight = history_use_decay_as_weight
        self.history_weight_decay_ratio = history_weight_decay_ratio

        self.num_tracks = 0
        self.tracks = dict()
        self.backdrops = []                 # Not use in this tracker
        self.max_distance = max_distance    # Maximum distance for considering matches
        self.fps = fps
        self.growth_factor = self.fps / 6  # Growth factor for the distance mask
        self.distance_smoothing_factor = 100 / self.fps

    def reset(self):
        """Reset the buffer of the tracker."""
        self.num_tracks = 0
        self.tracks = dict()
        self.backdrops = []

    def update(
        self,
        ids: Tensor,
        bboxes: Tensor,
        embeds: Tensor,
        labels: Tensor,
        scores: Tensor,
        frame_id: int,
    ) -> None:
        """Tracking forward function.

        Args:
            ids (Tensor): of shape(N, ).
            bboxes (Tensor): of shape (N, 5).
            embeds (Tensor): of shape (N, 256).
            labels (Tensor): of shape (N, ).
            scores (Tensor): of shape (N, ).
            frame_id (int): The id of current frame, 0-index.
        """
        tracklet_inds = ids > -1

        for id, bbox, embed, label, score in zip(
            ids[tracklet_inds],
            bboxes[tracklet_inds],
            embeds[tracklet_inds],
            labels[tracklet_inds],
            scores[tracklet_inds],
        ):
            id = int(id)
            # update the tracked ones and initialize new tracks
            if id in self.tracks.keys():
                self.tracks[id]["bbox"] = bbox
                self.tracks[id]["embed"] = (1 - self.memo_momentum) * self.tracks[id][
                    "embed"
                ] + self.memo_momentum * embed
                self.tracks[id]["last_frame"] = frame_id
                self.tracks[id]["label"] = label
                self.tracks[id]["score"] = score
                # For Transfer Tracker:
                # self.tracks[id]["transfer_hist_embeds"].append(embed)
                # self.tracks[id]["transfer_hist_scores"].append(score)
                self.tracks[id]["transfer_history"].add(feature=embed, score=score)
            else:
                match self.history_queue_type:
                    case "FIFOQueue": queue = FIFOQueue(
                        self.transfer_hist_len,
                        use_decay_as_weight=self.history_use_decay_as_weight,
                        weight_decay_ratio=self.history_weight_decay_ratio,
                    )
                    case _: raise ValueError(f"Unsupported history_queue_type: {self.history_queue_type}")
                self.tracks[id] = dict(
                    bbox=bbox,
                    embed=embed,
                    label=label,
                    score=score,
                    last_frame=frame_id,
                    # For Transfer Tracker:
                    # transfer_hist_embeds=deque(maxlen=self.transfer_hist_len),
                    # transfer_hist_scores=deque(maxlen=self.transfer_hist_len),
                    transfer_history=queue,
                )
                # For Transfer Tracker:
                # self.tracks[id]["transfer_hist_embeds"].append(embed)
                # self.tracks[id]["transfer_hist_scores"].append(score)
                self.tracks[id]["transfer_history"].add(feature=embed, score=score)
                pass

        # pop memo
        invalid_ids = []
        for k, v in self.tracks.items():
            if frame_id - v["last_frame"] >= self.memo_tracklet_frames:
                invalid_ids.append(k)
        for invalid_id in invalid_ids:
            self.tracks.pop(invalid_id)

    @property
    def memo(self) -> Tuple[Tensor, ...]:
        """Get tracks memory."""
        memo_embeds = []
        memo_ids = []
        memo_bboxes = []
        memo_labels = []
        memo_frame_ids = []
        # For Transfer Tracker:
        transfer_hist_embeds = []
        transfer_hist_scores = []
        transfer_hist_ids = []

        # get tracks
        for k, v in self.tracks.items():
            memo_bboxes.append(v["bbox"][None, :])
            memo_embeds.append(v["embed"][None, :])
            memo_ids.append(k)
            memo_labels.append(v["label"].view(1, 1))
            memo_frame_ids.append(v["last_frame"])
            # For Transfer Tracker:
            _embeds, _scores = v["transfer_history"].get()
            transfer_hist_embeds += _embeds
            transfer_hist_scores += _scores
            transfer_hist_ids += [torch.tensor([k]) for _ in range(len(_embeds))]

        memo_ids = torch.tensor(memo_ids, dtype=torch.long).view(1, -1)
        memo_bboxes = torch.cat(memo_bboxes, dim=0)
        memo_embeds = torch.cat(memo_embeds, dim=0)
        memo_labels = torch.cat(memo_labels, dim=0).squeeze(1)
        memo_frame_ids = torch.tensor(memo_frame_ids, dtype=torch.long).view(1, -1)
        # For Transfer Tracker:
        transfer_hist_embeds = torch.stack(transfer_hist_embeds, dim=0)
        transfer_hist_scores = torch.stack(transfer_hist_scores, dim=0)
        transfer_hist_ids = torch.cat(transfer_hist_ids, dim=0)
        assert len(transfer_hist_embeds) == len(transfer_hist_ids), f"{len(transfer_hist_embeds)} != {len(transfer_hist_ids)}"

        return (
            memo_bboxes,
            memo_labels,
            memo_embeds,
            memo_ids.squeeze(0),
            memo_frame_ids.squeeze(0),
            # For Transfer Tracker:
            transfer_hist_embeds,
            transfer_hist_scores,
            transfer_hist_ids,
        )

    def compute_distance_mask(self, bboxes1, bboxes2, frame_ids1, frame_ids2):
        """Compute a mask based on the pairwise center distances and frame IDs with piecewise soft-weighting."""
        centers1 = (bboxes1[:, :2] + bboxes1[:, 2:]) / 2.0
        centers2 = (bboxes2[:, :2] + bboxes2[:, 2:]) / 2.0
        distances = torch.cdist(centers1, centers2)

        frame_id_diff = torch.abs(frame_ids1[:, None] - frame_ids2[None, :]).to(
            distances.device
        )

        # Define a scaling factor for the distance based on frame difference (exponential growth)
        scaling_factor = torch.exp(frame_id_diff.float() / self.growth_factor)

        # Apply the scaling factor to max_distance
        adaptive_max_distance = self.max_distance * scaling_factor

        # Create a piecewise function for soft gating
        soft_distance_mask = torch.where(
            distances <= adaptive_max_distance,
            torch.ones_like(distances),
            torch.exp(
                -(distances - adaptive_max_distance) / self.distance_smoothing_factor
            ),
        )

        return soft_distance_mask

    def track(
        self,
        model: torch.nn.Module,
        img: torch.Tensor,
        feats: List[torch.Tensor],
        data_sample: TrackDataSample,
        rescale=True,
        with_segm=False,
        **kwargs
    ) -> InstanceData:
        """Tracking forward function.

        Args:
            model (nn.Module): MOT model.
            img (Tensor): of shape (T, C, H, W) encoding input image.
                Typically these should be mean centered and std scaled.
                The T denotes the number of key images and usually is 1.
            feats (list[Tensor]): Multi level feature maps of `img`.
            data_sample (:obj:`TrackDataSample`): The data sample.
                It includes information such as `pred_instances`.
            rescale (bool, optional): If True, the bounding boxes should be
                rescaled to fit the original scale of the image. Defaults to
                True.

        Returns:
            :obj:`InstanceData`: Tracking results of the input images.
            Each InstanceData usually contains ``bboxes``, ``labels``,
            ``scores`` and ``instances_id``.
        """
        metainfo = data_sample.metainfo
        bboxes = data_sample.pred_instances.bboxes      # (N, 4), xyxy format
        labels = data_sample.pred_instances.labels      # (N, )
        scores = data_sample.pred_instances.scores      # (N, )

        frame_id = metainfo.get("frame_id", -1)         # 0-based frame index
        # create pred_track_instances
        pred_track_instances = InstanceData()

        # return zero bboxes if there is no track targets
        if bboxes.shape[0] == 0:
            ids = torch.zeros_like(labels)
            pred_track_instances = data_sample.pred_instances.clone()
            pred_track_instances.instances_id = ids
            pred_track_instances.mask_inds = torch.zeros_like(labels)
            return pred_track_instances

        # get track feats
        rescaled_bboxes = bboxes.clone()
        if rescale:
            scale_factor = rescaled_bboxes.new_tensor(metainfo["scale_factor"]).repeat(
                (1, 2)
            )
            rescaled_bboxes = rescaled_bboxes * scale_factor
        track_feats = model.track_head.predict(feats, [rescaled_bboxes])    # (N, C)
        # sort according to the object_score
        _, inds = scores.sort(descending=True)
        bboxes = bboxes[inds]
        scores = scores[inds]
        labels = labels[inds]
        embeds = track_feats[inds, :]
        if with_segm:
            mask_inds = torch.arange(bboxes.size(0)).to(embeds.device)
            mask_inds = mask_inds[inds]
        else:
            mask_inds = []

        bboxes, labels, scores, embeds, mask_inds = self.remove_distractor(
            bboxes,
            labels,
            scores,
            track_feats=embeds,
            mask_inds=mask_inds,
            nms="inter",
            distractor_score_thr=self.distractor_score_thr,
            distractor_nms_thr=self.distractor_nms_thr,
        )

        # init ids container
        ids = torch.full((bboxes.size(0),), -1, dtype=torch.long)   # (N, ), all -1 (default)

        # match if buffer is not empty
        if bboxes.size(0) > 0 and not self.empty:
            (
                memo_bboxes,
                memo_labels,
                memo_embeds,
                memo_ids,
                memo_frame_ids,
                # For Transfer Tracker:
                transfer_hist_embeds,
                transfer_hist_scores,
                transfer_hist_ids,
            ) = self.memo   # N_tracks = len()

            # For Transfer Tracker, get the transfer features:
            if self.use_transfer and len(transfer_hist_embeds) > self.transfer_factor_thr * len(memo_ids):
                # Is OK to employ History-Aware Transformation for ReID features:
                from transform.lda.lda import LDA
                lda_model = LDA(
                    use_shrinkage=True,
                    dtype=self.transfer_dtype,
                    use_standard_scaler=self.use_standard_scaler,
                    direct_inter_class_diff=self.direct_inter_class_diff,
                    use_weighted_class_mean=self.use_weighted_class_mean,
                    use_related_inter_class_diff=self.use_related_inter_class_diff,
                    use_weighted_inter_class_diff=self.use_weighted_inter_class_diff,
                    weighted_class_mean_alpha=self.weighted_class_mean_alpha,
                )
                lda_model.fit(transfer_hist_embeds, transfer_hist_ids, score=transfer_hist_scores)
                transferred_memo_embeds = lda_model.transform(memo_embeds)
                transferred_embeds = lda_model.transform(embeds)
                pass
            else:
                transferred_memo_embeds = memo_embeds.clone()
                transferred_embeds = embeds.clone()

            # Bi-Softmax matching:
            # feats = torch.mm(embeds, memo_embeds.t())
            feats = torch.mm(transferred_embeds, transferred_memo_embeds.t())
            d2t_scores = feats.softmax(dim=1)
            t2d_scores = feats.softmax(dim=0)
            match_scores_bisoftmax = (d2t_scores + t2d_scores) / 2

            # Cosine Similarity matching:
            # match_scores_cosine = torch.mm(
            #     F.normalize(embeds, p=2, dim=1),
            #     F.normalize(memo_embeds, p=2, dim=1).t(),
            # )
            match_scores_cosine = torch.mm(
                F.normalize(transferred_embeds, p=2, dim=1),
                F.normalize(transferred_memo_embeds, p=2, dim=1).t(),
            )

            if self.similarity_function == "masa":
                match_scores = (match_scores_bisoftmax + match_scores_cosine) / 2
            elif self.similarity_function == "cosine":
                if self.transfer_similarity_alpha < 1.0:
                    untransferred_match_score_cosine = torch.mm(
                        F.normalize(embeds, p=2, dim=1),
                        F.normalize(memo_embeds, p=2, dim=1).t(),
                    )
                    match_scores = self.transfer_similarity_alpha * match_scores_cosine + (1 - self.transfer_similarity_alpha) * untransferred_match_score_cosine
                else:
                    match_scores = match_scores_cosine
            elif self.similarity_function == "bisoftmax":
                match_scores = match_scores_bisoftmax
            else:
                raise ValueError(f"Unsupported similarity function: {self.similarity_function}")

            if self.max_distance != -1:

                # Compute the mask based on spatial proximity
                current_frame_ids = torch.full(
                    (bboxes.size(0),), frame_id, dtype=torch.long
                )
                distance_mask = self.compute_distance_mask(
                    bboxes, memo_bboxes, current_frame_ids, memo_frame_ids
                )

                # Apply the mask to the match scores
                match_scores = match_scores * distance_mask

            # track according to match_scores
            if self.assignment_protocol == "masa":
                for i in range(bboxes.size(0)):
                    conf, memo_ind = torch.max(match_scores[i, :], dim=0)
                    id = memo_ids[memo_ind]
                    if conf > self.match_score_thr:
                        if id > -1:
                            # keep bboxes with high object score
                            # and remove background bboxes
                            if scores[i] > self.obj_score_thr:
                                ids[i] = id
                                match_scores[:i, memo_ind] = 0
                                match_scores[i + 1 :, memo_ind] = 0
            elif self.assignment_protocol == "object-max":
                id_max_confs, id_idxs = torch.max(match_scores, dim=0)
                for i in range(bboxes.size(0)):
                    conf, memo_ind = torch.max(match_scores[i, :], dim=0)
                    id = memo_ids[memo_ind]
                    if conf > self.match_score_thr and conf == id_max_confs[memo_ind]:
                        if id > -1:
                            # keep bboxes with high object score
                            # and remove background bboxes
                            if scores[i] > self.obj_score_thr:
                                ids[i] = id
                                id_max_confs[memo_ind] += 0.1   # ensure the same id will not be matched again
            elif self.assignment_protocol == "hungarian":
                match_cost = - match_scores
                row_idxs, col_idxs = linear_sum_assignment(match_cost.cpu().numpy())
                for i in range(len(row_idxs)):
                    obj_idx, id_idx = row_idxs[i], col_idxs[i]
                    id = memo_ids[id_idx]
                    if match_scores[obj_idx, id_idx] > self.match_score_thr:
                        if id > -1:
                            # keep bboxes with high object score
                            # and remove background bboxes
                            if scores[obj_idx] > self.obj_score_thr:
                                ids[obj_idx] = id
            else:
                raise ValueError(f"Unsupported assignment protocol: {self.assignment_protocol}")


        # initialize new tracks
        new_inds = (ids == -1) & (scores > self.init_score_thr).cpu()
        num_news = new_inds.sum()
        ids[new_inds] = torch.arange(
            self.num_tracks, self.num_tracks + num_news, dtype=torch.long
        )
        self.num_tracks += num_news

        self.update(ids, bboxes, embeds, labels, scores, frame_id)
        tracklet_inds = ids > -1
        # update pred_track_instances
        pred_track_instances.bboxes = bboxes[tracklet_inds]
        pred_track_instances.labels = labels[tracklet_inds]
        pred_track_instances.scores = scores[tracklet_inds]
        pred_track_instances.instances_id = ids[tracklet_inds]
        if with_segm:
            pred_track_instances.mask_inds = mask_inds[tracklet_inds]

        return pred_track_instances

    def remove_distractor(
        self,
        bboxes,
        labels,
        scores,
        track_feats,
        mask_inds=[],
        distractor_score_thr=0.5,
        distractor_nms_thr=0.3,
        nms="inter",
    ):
        # all objects is valid here
        valid_inds = labels > -1
        # nms
        low_inds = torch.nonzero(scores < distractor_score_thr, as_tuple=False).squeeze(
            1
        )
        if nms == "inter":
            ious = bbox_overlaps(bboxes[low_inds, :], bboxes[:, :])
        elif nms == "intra":
            cat_same = labels[low_inds].view(-1, 1) == labels.view(1, -1)
            ious = bbox_overlaps(bboxes[low_inds, :], bboxes)
            ious *= cat_same.to(ious.device)
        else:
            raise NotImplementedError

        for i, ind in enumerate(low_inds):
            if (ious[i, :ind] > distractor_nms_thr).any():
                valid_inds[ind] = False

        bboxes = bboxes[valid_inds]
        labels = labels[valid_inds]
        scores = scores[valid_inds]
        if track_feats is not None:
            track_feats = track_feats[valid_inds]

        if len(mask_inds) > 0:
            mask_inds = mask_inds[valid_inds]

        return bboxes, labels, scores, track_feats, mask_inds
