# Copyright (c) Ruopeng Gao. All Rights Reserved.

import torch
from collections import defaultdict, deque


class FIFOQueue:
    def __init__(self, max_len: int, weight_decay_ratio: float, use_decay_as_weight: bool):
        self.max_len = max_len
        self.weight_decay_ratio = weight_decay_ratio
        self.use_decay_as_weight = use_decay_as_weight
        self.features = deque(maxlen=self.max_len)
        self.scores = deque(maxlen=self.max_len)
        self.weights = deque(maxlen=self.max_len)

    def __len__(self):
        return len(self.features)

    def add(self, feature, score):
        self.features.append(feature)
        self.scores.append(score)
        self.weights.append(torch.tensor(1.0))

        # Decay the weights:
        for _ in range(len(self.weights)):
            self.weights[_] *= self.weight_decay_ratio
        return

    def get(self):
        if self.use_decay_as_weight:
            return list(self.features), list(self.weights)
        else:
            return list(self.features), list(self.scores)