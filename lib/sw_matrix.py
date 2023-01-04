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



def sliced_cost_matrix(Xs, Xt, A, u_weights=None, v_weights=None, p=1):
    n, _, _ = Xs.shape
    m, _, _ = Xt.shape
    
    device = Xs.device

    n_proj, d, _ = A.shape
    
    ## Compute logM in advance since we cannot batch it        
#     log_Xs = linalg.sym_logm(Xs)
#     log_Xt = linalg.sym_logm(Xt)

    ## Busemann Coordinates
#     Xps = busemann_spd(Xs, A).reshape(-1, n_proj)
#     Xpt = busemann_spd(Xt, A).reshape(-1, n_proj)
    
    prod_Xs = (A[None]*Xs[:,None]).reshape(n, n_proj,-1)
    Xps = prod_Xs.sum(-1) # busemann_spd(log_Xs, A).reshape(-1, n_proj)
    
    prod_Xt = (A[None]*Xt[:,None]).reshape(m, n_proj,-1)
    Xpt = prod_Xt.sum(-1) # busemann_spd(log_Xt, A).reshape(-1, n_proj)
    
    return torch.mean(emd1D(Xps.T,Xpt.T,
                       u_weights=u_weights,
                       v_weights=v_weights,
                       p=p))



def sliced_wasserstein_matrix(Xs, Xt, num_projections, device,
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
    theta = np.random.normal(size=(num_projections, d*d))
    theta = F.normalize(torch.from_numpy(theta), p=2, dim=-1).type(Xs.dtype).to(device)
    A = theta.reshape(num_projections, d, d)
    
    return sliced_cost_matrix(Xs, Xt, A, u_weights=u_weights, 
                           v_weights=v_weights, p=p)

    

def sliced_cost_logmatrix(Xs, Xt, A, u_weights=None, v_weights=None, p=1):
    n, _, _ = Xs.shape
    m, _, _ = Xt.shape
    
    device = Xs.device

    n_proj, d, _ = A.shape
    
    ## Compute logM in advance since we cannot batch it        
    log_Xs = linalg.sym_logm(Xs)
    log_Xt = linalg.sym_logm(Xt)

    ## Busemann Coordinates    
    prod_Xs = (A[None]*log_Xs[:,None]).reshape(n, n_proj,-1)
    Xps = prod_Xs.sum(-1) # busemann_spd(log_Xs, A).reshape(-1, n_proj)
    
    prod_Xt = (A[None]*log_Xt[:,None]).reshape(m, n_proj,-1)
    Xpt = prod_Xt.sum(-1) # busemann_spd(log_Xt, A).reshape(-1, n_proj)
    
    return torch.mean(emd1D(Xps.T,Xpt.T,
                       u_weights=u_weights,
                       v_weights=v_weights,
                       p=p))



def sliced_wasserstein_logmatrix(Xs, Xt, num_projections, device,
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
    theta = np.random.normal(size=(num_projections, d*d))
    theta = F.normalize(torch.from_numpy(theta), p=2, dim=-1).type(Xs.dtype).to(device)
    A = theta.reshape(num_projections, d, d)
    
    return sliced_cost_logmatrix(Xs, Xt, A, u_weights=u_weights, 
                           v_weights=v_weights, p=p)