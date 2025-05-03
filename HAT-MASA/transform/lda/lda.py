# Copyright (c) Ruopeng Gao. All Rights Reserved.

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
            use_normalize=False,
            use_standard_scaler=False,
            use_pca=False,
            direct_inter_class_diff=True,
            use_related_inter_class_diff=False,
            use_weighted_class_mean=False,
            use_weighted_inter_class_diff=False,
            weighted_class_mean_alpha=1.0,
            use_weighted_cov=False,
            use_sample_average=True,
            weight_min=0.0,
            dtype=torch.float32,
            device="cuda",
    ):
        self.use_shrinkage = use_shrinkage
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
        self.weight_min = weight_min
        self.dtype = dtype
        self.device = device

        # Check for the settings:
        if self.use_weighted_class_mean:
            if self.direct_inter_class_diff is False:
                raise ValueError("When using weighted class mean, direct inter-class difference must be True.")
        if self.use_weighted_inter_class_diff:
            if self.direct_inter_class_diff is False:
                raise ValueError("When using weighted inter-class difference, direct inter-class difference must be True.")

        # Standard Scaler:
        if self.use_standard_scaler:
            self.standard_scaler = StandardScaler()
        else:
            self.standard_scaler = None
        # PCA model:
        if self.use_pca:
            self.pca_model = PCA(n_components=None)
        else:
            self.pca_model = None
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
        if self.use_normalize:
            X = functional.normalize(X, p=2, dim=-1)
        if self.standard_scaler is not None:
            self._standard_scaler_fit(X)
            X = self._standard_scaler_transform(X)
        if self.pca_model is not None:
            self._pca_fit(X)
            X = self._pca_transform(X)

        # 3. calculate the needed statistics:
        S_w = self._get_class_cov(X, y, score=score if self.use_weighted_cov else None)
        if self.direct_inter_class_diff:
            self.class_means = self._get_class_means(X, y, score=score if self.use_weighted_class_mean else None)
            S_b = self._get_inter_class_diff(X, y, score=score if self.use_weighted_class_mean else None)
        else:
            S_t = self._get_cov(X, score=score if self.use_weighted_cov else None)
            S_b = S_t - S_w

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

        if self.use_normalize:
            X = functional.normalize(X, p=2, dim=-1)
        if self.standard_scaler is not None:
            X = self._standard_scaler_transform(X)
        if self.pca_model is not None:
            X = self._pca_transform(X)

        return X @ self.project_matrix

    def _pca_fit(self, X):
        X = self._sklearn_preprocess(X)
        self.pca_model.fit(X)
        return

    def _pca_transform(self, X):
        X = self._sklearn_preprocess(X)
        X = self.pca_model.transform(X)
        return self._sklearn_postprocess(X)

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
        S_b = torch.zeros((X.shape[1], X.shape[1]), dtype=self.dtype, device=self.device)
        # Calculate inter-class diff weights:
        if self.use_weighted_inter_class_diff:
            inter_class_similarity = functional.normalize(self.class_means, p=2, dim=-1) @ functional.normalize(self.class_means, p=2, dim=-1).T
            inter_class_similarity = (inter_class_similarity + 1) / 2       # normalize to [0, 1]
            # inter_class_distance = 1 - inter_class_similarity
            inter_class_weights = inter_class_similarity
            pass
        else:
            inter_class_weights = torch.ones((len(self.classes), len(self.classes)), dtype=self.dtype, device=self.device)      # 对称矩阵
        if self.use_related_inter_class_diff:
            # for i, ci in enumerate(self.classes):
            #     for j, cj in enumerate(self.classes):
            #         if i != j:
            #             mean_diff = self.class_means[i] - self.class_means[j]
            #             S_b += mean_diff[:, None] @ mean_diff[None, :]
            for i in range(len(self.class_means)):
                for j in range(len(self.class_means)):
                    if i != j:
                        mean_diff = self.class_means[i] - self.class_means[j]

                        if self.use_weighted_class_mean:
                            n = torch.sum(score[y == self.classes[i]])
                        else:
                            n = (y == self.classes[i]).sum()

                        if self.use_sample_average:
                            S_b += n * inter_class_weights[i, j] * mean_diff[:, None] @ mean_diff[None, :]
                        else:
                            S_b += inter_class_weights[i, j] * mean_diff[:, None] @ mean_diff[None, :]
            pass
        else:
            overall_mean = self._get_mean(X, dim=0, weights=score)
            for i, c in enumerate(self.classes):
                if self.use_weighted_class_mean:
                    n = torch.sum(score[y == c])
                else:
                    n = (y == c).sum()

                mean_diff = self.class_means[i] - overall_mean
                if self.use_sample_average:
                    S_b += n * mean_diff[:, None] @ mean_diff[None, :]
                else:
                    S_b += (_N / len(self.classes)) * mean_diff[:, None] @ mean_diff[None, :]
            pass
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
        X = X - X.mean(dim=0)
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
            weights = weights ** self.weighted_class_mean_alpha
            return torch.sum(weights[:, None] * X, dim=dim) / torch.sum(weights)

