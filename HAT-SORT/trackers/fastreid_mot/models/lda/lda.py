# Copyright (c) Ruopeng Gao. All Rights Reserved.
# About:
import torch
# from .standard_scaler import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import scipy.linalg
from torch.nn import functional


class LDA:
    def __init__(
            self,
            use_shrinkage=True,
            use_standard_scaler=False,
            use_sample_average=False,
            weight_min=0.0,
            dtype=torch.float32,
            device="cuda",
    ):
        self.use_shrinkage = use_shrinkage
        self.use_standard_scaler = use_standard_scaler
        self.use_sample_average = use_sample_average
        self.weight_min = weight_min
        self.dtype = dtype
        self.device = device

        # Standard Scaler:
        if self.use_standard_scaler:
            self.standard_scaler = StandardScaler()
        else:
            self.standard_scaler = None

        # LDA attributes:
        self.classes = None
        self.class_means = None
        self.project_matrix = None
        pass

    def clear(self):
        self.classes = None
        self.class_means = None
        self.project_matrix = None
        return

    def is_fit(self):
        return self.project_matrix is not None

    def fit(self, X, y, score=None):
        """

        Args:
            X:
            y:
            score:

        Returns:

        """
        # 1. pre-process the input data:
        if isinstance(X, torch.Tensor):
            X = X.to(dtype=self.dtype, device=self.device)
        else:
            X = torch.tensor(X, dtype=self.dtype, device=self.device)
        if isinstance(y, torch.Tensor):
            y = y.to(dtype=torch.long, device=self.device)
        else:
            y = torch.tensor(y, dtype=torch.long, device=self.device)
        if score is not None:
            if isinstance(score, torch.Tensor):
                score = score.to(dtype=self.dtype, device=self.device)
            else:
                score = torch.tensor(score, dtype=self.dtype, device=self.device)
            score = torch.clamp(score, min=self.weight_min)

        # 2. prepare the data:
        self.classes = torch.unique(y).tolist()
        if self.standard_scaler is not None:
            self._standard_scaler_fit(X)
            X = self._standard_scaler_transform(X)

        # 3. calculate the needed statistics:
        S_w = self._get_class_cov(X, y, score=None)
        self.class_means = self._get_class_means(X, y, score=score)  # if self.use_weighted_class_mean else None)
        S_b = self._get_inter_class_diff(X, y, score=score)  # if self.use_weighted_class_mean else None)


        # 4. calculate the projection matrix:
        eig_vals, eig_vecs = scipy.linalg.eigh(S_b.cpu().numpy(), S_w.cpu().numpy())
        # eig_vals, eig_vecs = scipy.linalg.eigh(S_b.cpu().numpy(), S_w.cpu().numpy(), subset_by_index=[S_b.shape[0]-len(self.classes), S_b.shape[0]-1])
        eig_vals = torch.tensor(eig_vals, dtype=self.dtype, device=self.device)
        eig_vecs = torch.tensor(eig_vecs, dtype=self.dtype, device=self.device)
        sorted_indices = torch.argsort(eig_vals, descending=True)
        eig_vecs = eig_vecs[:, sorted_indices]
        self.project_matrix = eig_vecs[:, :len(self.classes) - 1]
        return

    def transform(self, X):
        assert self.project_matrix is not None, "Please fit the model first."
        if isinstance(X, torch.Tensor):
            X = X.to(dtype=self.dtype, device=self.device)
        else:
            X = torch.tensor(X, dtype=self.dtype, device=self.device)

        if self.standard_scaler is not None:
            X = self._standard_scaler_transform(X)

        return X @ self.project_matrix

    def _standard_scaler_fit(self, X):
        X = self._sklearn_preprocess(X)
        self.standard_scaler.fit(X)
        return

    def _standard_scaler_transform(self, X):
        X = self._sklearn_preprocess(X)
        X = self.standard_scaler.transform(X)
        return self._sklearn_postprocess(X)

    def _sklearn_preprocess(self, X):
        if isinstance(X, torch.Tensor):
            return X.cpu().numpy()
        else:
            return X

    def _sklearn_postprocess(self, X):
        return torch.tensor(X, dtype=self.dtype, device=self.device)

    def _get_class_means(self, X, y, score=None):
        class_means = []
        for c in self.classes:
            # class_means.append(X[y == c].mean(dim=0))
            class_means.append(self._get_mean(X[y == c], dim=0, weights=score[y == c] if score is not None else None))
        return torch.stack(class_means, dim=0)

    def _get_inter_class_diff(self, X, y, score=None):
        _N = X.shape[0]
        overall_mean = self._get_mean(X, dim=0, weights=score)
        classes, y_indices = torch.unique(y, sorted=True, return_inverse=True)
        num_classes = len(classes)
        _C = X.shape[1]

        n = torch.zeros(num_classes, device=X.device)
        n = n.index_add_(0, y_indices, score)
        mean_diff = self.class_means - overall_mean             # (N_classes, C)

        # Average weight:
        if self.use_sample_average:
            weights = n
        else:
            weights = (_N / num_classes) * torch.ones(num_classes, device=X.device)

        weights = weights[:, None, None]            # (num_classes, 1, 1)

        # Calculate the inter-class difference:
        mean_diff = mean_diff[:, :, None] @ mean_diff[:, None, :]

        # Calculate the weighted inter-class difference:
        S_b = torch.sum(weights * mean_diff, dim=0)
        return S_b

    def _get_class_cov(self, X, y, score=None):
        _N, _C = X.shape
        class_cov = torch.zeros((_C, _C), dtype=self.dtype, device=self.device)

        for c in self.classes:
            class_X = X[y == c]
            if score is not None:
                score_X = score[y == c]
            else:
                score_X = None
            if self.use_sample_average:
                class_cov += (len(class_X) / _N) * self._get_cov(class_X, score=score_X)
            else:
                class_cov += (1 / len(self.classes)) * self._get_cov(class_X, score=score_X)
            # class_cov += self._get_cov(class_X)
        return class_cov

    def _get_cov(self, X, score):
        if self.use_shrinkage:
            standard_scaler = StandardScaler()
            if score is None:
                X = standard_scaler.fit_transform(X.cpu().numpy())
            else:
                X = standard_scaler.fit_transform(X.cpu().numpy(), sample_weight=score.cpu().numpy())
            X = torch.tensor(X, dtype=self.dtype, device=self.device)
            cov = self._get_shrunk_cov(X, score=score)
            scale_ = torch.tensor(standard_scaler.scale_, dtype=self.dtype, device=self.device)
            cov = scale_[:, None] * cov * scale_[None, :]
            pass
        else:
            raise NotImplementedError
        return cov

    def _get_shrunk_cov(self, X, score=None):
        # In the sklearn version, although X is already centered by StandardScaler, we still center it again.
        # _mean = X.mean(dim=0)
        # shrinkage = self._get_shrinkage(X - _mean)
        # cov = self._get_normal_cov(X - _mean)
        # In our latest version, we do not center X again.
        # On DanceTrack val, the performance is not significantly different. (+ 0.01 HOTA)
        # X = X - X.mean(dim=0)
        shrinkage = self._get_shrinkage(X)
        cov = self._get_normal_cov(X, score=score, assume_centered=True)
        mu = torch.sum(torch.trace(cov)) / cov.shape[0]
        shrunk_cov = (1 - shrinkage) * cov
        diag_indices = torch.arange(cov.shape[0])
        shrunk_cov[diag_indices, diag_indices] += shrinkage * mu
        return shrunk_cov

    def _get_shrinkage(self, X):
        _N, _C = X.shape
        X2 = X ** 2
        emp_cov_trace = torch.sum(X2, dim=0) / _N
        mu = torch.sum(emp_cov_trace) / _C
        beta_, delta_ = 0.0, 0.0

        beta_ += torch.sum(torch.matmul(X2.T, X2))
        delta_ += torch.sum(torch.matmul(X.T, X)**2)

        delta_ /= _N ** 2

        beta = 1.0 / (_C * _N) * (beta_ / _N - delta_)
        delta = delta_ - 2.0 * mu * emp_cov_trace.sum() + _C * mu**2
        delta /= _C

        beta = min(beta, delta)

        shrinkage = 0 if beta == 0 else beta / delta
        return shrinkage

    def _get_normal_cov(self, X, score=None, assume_centered=False):
        if assume_centered:
            if score is None:
                return X.T @ X / X.shape[0]
            else:
                return X.T @ (score[:, None] * X) / torch.sum(score)
        else:
            return torch.cov(X.T, correction=0)

    def _get_mean(self, X, dim, weights=None):
        if weights is None:
            return torch.mean(X, dim=dim)
        else:
            assert len(X.shape) == 2, "Only support 2D tensor."
            assert dim == 0, "Only support dim=0."
            assert len(X) == len(weights), "The length of X and weights must be the same."
            # weights = weights ** self.weighted_class_mean_alpha
            return torch.sum(weights[:, None] * X, dim=dim) / torch.sum(weights)

