import torch

import numpy as np
import torch.nn.functional as F

from geoopt import linalg
from scipy.stats import ortho_group

# from logm import logm
from utils_spd import busemann_spd

device = "cuda" if torch.cuda.is_available() else "cpu"


def emd1D(u_values, v_values, u_weights=None, v_weights=None,p=1, require_sort=True):
    n = u_values.shape[-1]
    m = v_values.shape[-1]

    device = u_values.device
    dtype = u_values.dtype

    if u_weights is None:
        u_weights = torch.full((n,), 1/n, dtype=dtype, device=device)

    if v_weights is None:
        v_weights = torch.full((m,), 1/m, dtype=dtype, device=device)

    if require_sort:
        u_values, u_sorter = torch.sort(u_values, -1)
        v_values, v_sorter = torch.sort(v_values, -1)

        u_weights = u_weights[..., u_sorter]
        v_weights = v_weights[..., v_sorter]

    zero = torch.zeros(1, dtype=dtype, device=device)
    
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
    return torch.sum(delta * torch.pow(torch.abs(u_icdf - v_icdf), p), axis=-1)



def sliced_cost_spd(Xs, Xt, A, u_weights=None, v_weights=None, p=1):
    n, _, _ = Xs.shape
    m, _, _ = Xt.shape
    
    device = Xs.device

    n_proj, d, _ = A.shape
    
    ## Compute logM in advance since we cannot batch it        
    log_Xs = linalg.sym_logm(Xs)
    log_Xt = linalg.sym_logm(Xt)

    ## Busemann Coordinates
    Xps = busemann_spd(log_Xs, A).reshape(-1, n_proj)
    Xpt = busemann_spd(log_Xt, A).reshape(-1, n_proj)
    
    return torch.mean(emd1D(Xps.T,Xpt.T,
                       u_weights=u_weights,
                       v_weights=v_weights,
                       p=p))



def sliced_wasserstein_spd(Xs, Xt, num_projections, device,
                           u_weights=None, v_weights=None, p=2):
    """
        Ref sampling uniform orthogonal matrix:
        Mezzadri, Francesco. "How to generate random matrices from the classical compact groups." arXiv preprint math-ph/0609050 (2006).
        
        Parameters:
        Xs: ndarray, shape (n_batch, d, d)
            Samples in the source domain
        Xt: ndarray, shape (m_batch, d, d)
            Samples in the target domain
        num_projections: int
            Number of projections
        device: str
        p: float
            Power of SW. Need to be >= 1.
    """
    n, d, _ = Xs.shape

    # Random projection directions, shape (d-1, num_projections)
    theta = np.random.normal(size=(num_projections, d))
    theta = F.normalize(torch.from_numpy(theta), p=2, dim=-1).type(Xs.dtype).to(device)
    
    D = theta[:,None] * torch.eye(theta.shape[-1], device=device)
    
    ## Random orthogonal matrices
    Z = torch.randn((num_projections, d, d), device=device, dtype=torch.float64)
    Q, R = torch.linalg.qr(Z)
    lambd = torch.diagonal(R, dim1=-2, dim2=-1)
    lambd = lambd / torch.abs(lambd)
    P = lambd[:,None]*Q
    #P = torch.tensor(ortho_group.rvs(d, num_projections), device=device, dtype=torch.float64)
    
    A = torch.matmul(P, torch.matmul(D, torch.transpose(P, -2, -1)))
    
    return sliced_cost_spd(Xs, Xt, A, u_weights=u_weights, 
                           v_weights=v_weights, p=p)


def sliced_wasserstein_spd_diagbasis(Xs, Xt, num_projections, device,
                           u_weights=None, v_weights=None, p=2):
    """
        Parameters:
        Xs: ndarray, shape (n_batch, d, d)
            Samples in the source domain
        Xt: ndarray, shape (m_batch, d, d)
            Samples in the target domain
        num_projections: int
            Number of projections
        device: str
        p: float
            Power of SW. Need to be >= 1.
    """
    n, d, _ = Xs.shape

    # Random projection directions, shape (d-1, num_projections)
    theta = np.random.normal(size=(num_projections, d))
    theta = F.normalize(torch.from_numpy(theta), p=2, dim=-1).type(Xs.dtype).to(device)
    
    A = theta[:,None] * torch.eye(theta.shape[-1], device=device)
    
    ## Preprocessing to compute the matrix product using a simple product
    #diagA = torch.diagonal(A, dim1=-2, dim2=-1)
    #dA = diagA.unsqueeze(-1)
    #dA = dA.repeat(1,1,d)
    
    return sliced_cost_spd(Xs, Xt, A, u_weights=u_weights, 
                           v_weights=v_weights, p=p)


def get_quantiles(x, ts, weights=None):
    """
        Inputs:
        - x: 1D values, size: n_projs * n_batch
        - ts: points at which to evaluate the quantile
    """
    n_projs, n_batch = x.shape
    
    if weights is None:
        X_weights = torch.full((n_batch,), 1/n_batch, dtype=x.dtype, device=x.device)
        X_values, X_sorter = torch.sort(x, -1)
        X_weights = X_weights[..., X_sorter]

    X_cdf = torch.cumsum(X_weights, -1) 
        
    X_index = torch.searchsorted(X_cdf, ts.repeat(n_projs, 1))
    X_icdf = torch.gather(X_values, -1, X_index.clip(0, n_batch-1))
        
    return X_icdf


def get_features(x, A, ts, weights=None, p=2):
    """
        Inputs:
        - x: ndarray, shape (n_batch, d, d)
            Samples of SPD
        - A: Symmetric matrices on which to project
        - ts: uniform samples on [0,1]
        - weights: weight of each sample, if None, uniform weights
        - p
    """
    num_projs, _, _ = A.shape
    num_unifs = len(ts)
    
    log_x = linalg.sym_logm(x)
    Xp = busemann_spd(log_x, A).reshape(-1, num_projs)
    q_Xp = get_quantiles(Xp.T, ts, weights)
    
    return q_Xp / (num_projs * num_unifs)**(1/p)
    


def sliced_wasserstein_spd_phi(Xs, Xt, num_projections, num_ts, 
                               u_weights=None, v_weights=None, p=2):
    """
        Parameters:
        Xs: ndarray, shape (n_batch, d, d)
            Samples in the source domain
        Xt: ndarray, shape (m_batch, d, d)
            Samples in the target domain
        num_projections: int
            Number of projections
        num_ts: int
            Number of uniform samples on [0,1] to approximate W in 1D
        device: str
        p: float
            Power of SW. Need to be >= 1.
    """
    n, d, _ = Xs.shape
    device = Xs.device
    
    # Random projection directions, shape (d-1, num_projections)
    theta = np.random.normal(size=(num_projections, d))
    theta = F.normalize(torch.from_numpy(theta), p=2, dim=-1).type(Xs.dtype).to(device)

    D = theta[:,None] * torch.eye(theta.shape[-1], device=device)
    
    ## Random orthogonal matrices
    Z = torch.randn((num_projections, d, d), device=device, dtype=torch.float64)
    Q, R = torch.linalg.qr(Z)
    lambd = torch.diagonal(R, dim1=-2, dim2=-1)
    lambd = lambd / torch.abs(lambd)
    P = lambd[:,None]*Q
    #P = torch.tensor(ortho_group.rvs(d, num_projections), device=device, dtype=torch.float64)
    
    A = torch.matmul(P, torch.matmul(D, torch.transpose(P, -2, -1)))

    ## Preprocessing to compute the matrix product using a simple product
    #diagA = torch.diagonal(A, dim1=-2, dim2=-1)
    #dA = diagA.unsqueeze(-1)
    #dA = dA.repeat(1,1,d)

    ts = torch.rand((num_ts,), device=device)
    

    features_Xs = get_features(Xs, A, ts, u_weights, p=p)
    features_Xt = get_features(Xt, A, ts, v_weights, p=p)
    
    if p==2:
        return torch.sum(torch.square(features_Xs-features_Xt))
    
