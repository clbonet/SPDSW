import torch

from geoopt import linalg


def coral(Xs, Xt, d=22):
    log_Xs = linalg.sym_logm(Xs[:,0,0]).detach().cpu().reshape(-1, d*d)
    log_Xt = linalg.sym_logm(Xt[:,0,0]).detach().cpu().reshape(-1, d*d)
    
    Cs = torch.cov(log_Xs.T) + torch.eye(log_Xs.shape[1])
    Ct = torch.cov(log_Xt.T) + torch.eye(log_Xt.shape[1])

    Cs_ = torch.linalg.inv(Cs)
    Cs_12 = linalg.sym_sqrtm(Cs_)

    Ct12 = linalg.sym_sqrtm(Ct)
    
    Xs_emb = torch.matmul(log_Xs, Cs_12)
    Xs_emb = torch.matmul(Xs_emb, Ct12)
    
    return linalg.sym_expm(Xs_emb.reshape(-1,d,d))
