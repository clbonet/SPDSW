import torch
import geoopt 
import ot

from pyriemann.utils.mean import mean_riemann


def otda(Xs, Xt, metric="ai", loss="emd", reg=1):
    """
        Xs: Source (n_batch, d, d)
        Xt: Target (m_batch, d, d)
        metric: "ai" or "le"
        loss: "emd" or "sinkhorn"
        reg: regularization for Sinkhorn
    """
    d = Xs.shape[-1]
    device = Xs.device 
    
    if metric == "ai":
        manifold_spd = geoopt.SymmetricPositiveDefinite("AIM")
    elif meric == "le":
        manifold_spd = geoopt.SymmetricPositiveDefinite("LEM")        

    a = torch.ones((len(Xs),), device=device, dtype=torch.float64)/len(Xs)
    b = torch.ones((len(Xt),), device=device, dtype=torch.float64)/len(Xt)
    M = manifold_spd.dist(Xs[:,None], Xt[None])**2
    
    if loss == "emd":
        P = ot.emd(a, b, M)
    elif loss == "sinkhorn":
        P = ot.sinkhorn(a, b, M, reg)
    
    if metric == "ai":
        cpt = torch.zeros((len(Xs), d, d))
        for i in range(len(Xs)):
            cpt[i] = torch.tensor(mean_riemann(Xt.cpu().numpy(), sample_weight=P[i].cpu().numpy()))
    elif metric == "le":
        log_Xt = geoopt.linalg.sym_logm(Xt)            
        cpt = torch.matmul(P, log_Xt[None].reshape(-1, d*d)).reshape(-1,d,d)
        cpt = geoopt.linalg.sym_expm(cpt*len(Xt))
            
    return cpt
