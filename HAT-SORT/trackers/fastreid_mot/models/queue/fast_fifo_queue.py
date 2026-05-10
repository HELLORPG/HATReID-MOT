# Copyright (c) Ruopeng Gao. All Rights Reserved.
# About:
import torch


class FIFOQueue:
    def __init__(
            self, max_len: int,
            weight_decay_ratio: float,
    ):
        self.max_len = max_len
        self.weight_decay_ratio = weight_decay_ratio
        self.features = None
        self.scores = None
        self.weights = None

    def __len__(self):
        return len(self.features)

    def add(self, feature, score):
        if self.features is None:
            self.features = feature[None, ...]
            self.scores = torch.tensor([score], device=feature.device)
            self.weights = torch.tensor([1.0], device=feature.device)
        else:
            self.features = torch.cat((self.features, feature[None, ...]), dim=0)
            self.scores = torch.cat((self.scores, torch.tensor([score], device=feature.device)), dim=0)
            self.weights = torch.cat((self.weights, torch.tensor([1.0], device=feature.device)), dim=0)

        # Decay the weights:
        self.weights = self.weights * self.weight_decay_ratio

        # Simulate the deque behavior:
        if len(self) > self.max_len:
            self.features = self.features[-self.max_len:, ...]
            self.scores = self.scores[-self.max_len:]
            self.weights = self.weights[-self.max_len:]
        return

    def get(self):
        return self.features, self.weights