# Copyright (c) Ruopeng Gao. All Rights Reserved.
# About:
import torch
from collections import defaultdict, deque


class ScoreQueue:
    def __init__(self, max_len: int):
        self.max_len = max_len
        self.features = None
        self.scores = None

    def __len__(self):
        assert self.features is not None, "ScoreQueue is not initialized."
        return len(self.features)

    def add(self, feature, score):
        feature, score = feature[None, ...], score[None, ...]
        if self.features is None:
            self.features = feature
            self.scores = score
        else:
            self.features = torch.cat((self.features, feature), dim=0)
            self.scores = torch.cat((self.scores, score), dim=0)
        # Check if the length exceeds the max_len:
        if len(self) > self.max_len:
            top_score_idxs = torch.argsort(self.scores, descending=True)[:self.max_len]
            self.features = self.features[top_score_idxs]
            self.scores = self.scores[top_score_idxs]
        return

    def get(self):
        # Return list of features and scores:
        return [f for f in self.features], [s for s in self.scores]