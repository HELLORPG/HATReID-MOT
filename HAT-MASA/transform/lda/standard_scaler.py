# Copyright (c) Ruopeng Gao. All Rights Reserved.

import torch


class StandardScaler:
    def __init__(self):

        self.mean = None
        self.std = None
        return

    def fit(self, X, weights=None):
        if weights is None:
            weights = torch.ones((X.shape[0], ), dtype=X.dtype, device=X.device)
        else:
            if isinstance(torch.Tensor):
                weights = weights.to(dtype=X.dtype, device=X.device)
            else:
                weights = torch.tensor(weights, dtype=X.dtype, device=X.device)

        weights_sum = torch.sum(weights)

        self.mean = torch.sum(weights[:, None] * X, dim=0) / weights_sum
        self.std = torch.sqrt(torch.sum(weights[:, None] * (X - self.mean) ** 2, dim=0) / weights_sum)


    def transform(self, X):
        assert self.mean is not None, "StandardScaler has not been fitted yet."
        return (X - self.mean) / self.std

    def fit_transform(self, X, weights=None):
        self.fit(X, weights)
        return self.transform(X)


