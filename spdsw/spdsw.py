import torch

import numpy as np
import torch.nn.functional as F

from geoopt import linalg


class SPDSW:
    """
        Class for computing SPDSW distance and embedding

        Parameters
        ----------
        shape_X : int
            dim projections
        num_projections : int
            Number of projections
        num_ts : int
            Number of timestamps for quantiles, default 20
        device : str
            Device for computations, default None
        dtype : type
            Data type, default torch.float
        random_state : int
            Seed, default 123456
        sampling : str
            Sampling type
                - "spdsw": symetric matrices + geodesic projection
                - "logsw": unit norm matrices + geodesic projection
                - "sw": unit norm matrices + euclidean projection
            Default "spdsw"
        """

    def __init__(
        self,
        shape_X,
        num_projections,
        num_ts=20,
        device=None,
        dtype=torch.float,
        random_state=123456,
        sampling="spdsw",
    ):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if sampling not in ["spdsw", "logsw", "sw"]:
            raise Exception("'sampling' should be in ['spdsw', 'logsw', 'sw']")

        self.generate_projections(
            shape_X, num_projections, num_ts,
            device, dtype, random_state, sampling
        )

        self.sampling = sampling

    def generate_projections(self, shape_X, num_projections, num_ts,
                             device, dtype, random_state, sampling):
        """
        Generate projections for sampling

        Parameters
        ----------
        shape_X : int
            dim projections
        num_projections : int
            Number of projections
        device : str
            Device for computations
        dtype : type
            Data type
        random_state : int
            Seed
        sampling : str
            Sampling type
                - "spdsw": symetric matrices + geodesic projection
                - "logsw": unit norm matrices + geodesic projection
                - "sw": unit norm matrices + euclidean projection
        """

        rng = np.random.default_rng(random_state)

        if sampling == "spdsw":

            # Random projection directions, shape (d-1, num_projections)
            theta = rng.normal(size=(num_projections, shape_X))
            theta = F.normalize(
                torch.from_numpy(theta), p=2, dim=-1
            ).type(dtype).to(device)

            D = theta[:, None] * torch.eye(
                theta.shape[-1],
                device=device,
                dtype=dtype
            )

            # Random orthogonal matrices
            Z = torch.randn(
                (num_projections, shape_X, shape_X),
                device=device,
                dtype=dtype
            )
            Q, R = torch.linalg.qr(Z)
            lambd = torch.diagonal(R, dim1=-2, dim2=-1)
            lambd = lambd / torch.abs(lambd)
            P = lambd[:, None] * Q

            self.A = torch.matmul(
                P,
                torch.matmul(D, torch.transpose(P, -2, -1))
            )

        elif sampling in ["logsw", "sw"]:

            self.A = torch.tensor(
                rng.normal(size=(num_projections, shape_X, shape_X)),
                dtype=dtype,
                device=device
            )

            self.A /= torch.norm(self.A, dim=(1, 2), keepdim=True)

        self.ts = torch.linspace(0, 1, num_ts, dtype=dtype, device=device)

    def emd1D(self, u_values, v_values, u_weights=None, v_weights=None, p=1):
        n = u_values.shape[-1]
        m = v_values.shape[-1]

        device = u_values.device
        dtype = u_values.dtype

        if u_weights is None:
            u_weights = torch.full((n,), 1/n, dtype=dtype, device=device)

        if v_weights is None:
            v_weights = torch.full((m,), 1/m, dtype=dtype, device=device)

        # Sort
        u_values, u_sorter = torch.sort(u_values, -1)
        v_values, v_sorter = torch.sort(v_values, -1)

        u_weights = u_weights[..., u_sorter]
        v_weights = v_weights[..., v_sorter]

        # Compute CDF
        u_cdf = torch.cumsum(u_weights, -1)
        v_cdf = torch.cumsum(v_weights, -1)

        cdf_axis, _ = torch.sort(torch.cat((u_cdf, v_cdf), -1), -1)

        u_index = torch.searchsorted(u_cdf, cdf_axis)
        v_index = torch.searchsorted(v_cdf, cdf_axis)

        u_icdf = torch.gather(u_values, -1, u_index.clip(0, n-1))
        v_icdf = torch.gather(v_values, -1, v_index.clip(0, m-1))

        cdf_axis = torch.nn.functional.pad(cdf_axis, (1, 0))
        delta = cdf_axis[..., 1:] - cdf_axis[..., :-1]

        if p == 1:
            return torch.sum(delta * torch.abs(u_icdf - v_icdf), axis=-1)
        if p == 2:
            return torch.sum(delta * torch.square(u_icdf - v_icdf), axis=-1)

        return torch.sum(
            delta * torch.pow(torch.abs(u_icdf - v_icdf), p),
            axis=-1
        )

    def spdsw(self, Xs, Xt, u_weights=None, v_weights=None, p=2):
        """
            Parameters:
            Xs: ndarray, shape (n_batch, d, d)
                Samples in the source domain
            Xt: ndarray, shape (m_batch, d, d)
                Samples in the target domain
            device: str
            p: float
                Power of SW. Need to be >= 1.
        """
        n, _, _ = Xs.shape
        m, _, _ = Xt.shape

        n_proj, d, _ = self.A.shape

        if self.sampling in ["spdsw", "logsw"]:
            # Busemann Coordinates
            log_Xs = linalg.sym_logm(Xs)
            log_Xt = linalg.sym_logm(Xt)

            prod_Xs = (self.A[None] * log_Xs[:, None]).reshape(n, n_proj, -1)
            prod_Xt = (self.A[None] * log_Xt[:, None]).reshape(m, n_proj, -1)

        elif self.sampling in ["sw"]:
            # Euclidean Coordinates
            prod_Xs = (self.A[None] * Xs[:, None]).reshape(n, n_proj, -1)
            prod_Xt = (self.A[None] * Xt[:, None]).reshape(m, n_proj, -1)

        Xps = prod_Xs.sum(-1)
        Xpt = prod_Xt.sum(-1)

        return torch.mean(
            self.emd1D(Xps.T, Xpt.T, u_weights, v_weights, p)
        )

    def get_quantiles(self, x, ts, weights=None):
        """
            Inputs:
            - x: 1D values, size: n_projs * n_batch
            - ts: points at which to evaluate the quantile
        """
        n_projs, n_batch = x.shape

        if weights is None:
            X_weights = torch.full(
                (n_batch,), 1/n_batch, dtype=x.dtype, device=x.device
            )
            X_values, X_sorter = torch.sort(x, -1)
            X_weights = X_weights[..., X_sorter]

        X_cdf = torch.cumsum(X_weights, -1)

        X_index = torch.searchsorted(X_cdf, ts.repeat(n_projs, 1))
        X_icdf = torch.gather(X_values, -1, X_index.clip(0, n_batch-1))

        return X_icdf

    def get_features(self, x, weights=None, p=2):
        """
            Inputs:
            - x: ndarray, shape (n_batch, d, d)
                Samples of SPD
            - weights: weight of each sample, if None, uniform weights
            - p
        """
        num_unifs = len(self.ts)
        n_proj, d, _ = self.A.shape
        n, _, _ = x.shape

        if self.sampling in ["spdsw", "logsw"]:
            log_x = linalg.sym_logm(x)
            Xp = (self.A[None] * log_x[:, None]).reshape(n, n_proj, -1).sum(-1)
        elif self.sampling == "sw":
            Xp = (self.A[None] * x[:, None]).reshape(n, n_proj, -1).sum(-1)
        q_Xp = self.get_quantiles(Xp.T, self.ts, weights)

        return q_Xp / ((n_proj * num_unifs) ** (1 / p))
