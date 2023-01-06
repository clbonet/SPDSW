import sys
import torch
import geoopt
import time
import argparse
import ot

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import trange
from sklearn.manifold import TSNE, SpectralEmbedding
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from geoopt import linalg, ManifoldParameter, Stiefel
from geoopt.optim import RiemannianSGD, RiemannianAdam
from copy import deepcopy

sys.path.append("../lib/data")
from get_data import get_data, get_cov2 ## get_cov2: with paramaters to have multi frequences

sys.path.append("../lib")
from swspd import sliced_wasserstein_spd, sliced_cost_spd
from sw_matrix import sliced_wasserstein_matrix, sliced_wasserstein_logmatrix
from vecswspd import sliced_wasserstein_vecspd
# from transformations import Translation, Rotation, sym_reeig




parser = argparse.ArgumentParser()
parser.add_argument("--loss", type=str, default="spdsw", help="Which loss to use")
parser.add_argument("--epochs", type=int, default=500, help="Number of epochs")
# parser.add_argument("--lr", type=float, default=1e-1, help="Learning rate") ## lr:1e-1 for swspd, 1e-2 for lew
parser.add_argument("--pbar", action="store_true", help="If yes, plot pbar")
args = parser.parse_args()


def get_projs(num_projections, d, device, dtype):
    # Random projection directions, shape (d-1, num_projections)
    theta = np.random.normal(size=(num_projections, d))
    theta = F.normalize(torch.from_numpy(theta), p=2, dim=-1).type(dtype).to(device)
    
    D = theta[:,None] * torch.eye(theta.shape[-1], device=device)
    
    ## Random orthogonal matrices
    Z = torch.randn((num_projections, d, d), device=device, dtype=dtype)
    Q, R = torch.linalg.qr(Z)
    lambd = torch.diagonal(R, dim1=-2, dim2=-1)
    lambd = lambd / torch.abs(lambd)
    P = lambd[:,None]*Q
#     P = torch.tensor(ortho_group.rvs(d, num_projections), device=device, dtype=torch.float64)
#     P = ortho_group_rvs(d, num_projections)
    
    A = torch.matmul(P, torch.matmul(D, torch.transpose(P, -2, -1)))
    return A


def spdsw(Xs, Xt, A, device, u_weights=None, v_weights=None, p=2):
    return sliced_cost_spd2(Xs, Xt, A, u_weights=u_weights, 
                           v_weights=v_weights, p=p)



## NNs

class Translation(nn.Module):
    def __init__(self, d, n_freq, device):
        super().__init__()

        manifold_spdai = geoopt.SymmetricPositiveDefinite("AIM")        
        self._W = ManifoldParameter(torch.eye(d, dtype=torch.double, device=device)[None, :].repeat(n_freq, 1, 1), manifold=manifold_spdai)

        with torch.no_grad():
            self._W.proj_()

    def forward(self, X):
        return torch.matmul(self._W, torch.matmul(X, self._W.transpose(2, 1)))

class Rotation(nn.Module):
    def __init__(self, d, n_freq, device):
        super().__init__()

        manifold = Stiefel()        
        self._W = ManifoldParameter(torch.eye(d, dtype=torch.double, device=device)[None, :].repeat(n_freq, 1, 1), manifold=manifold)

        with torch.no_grad():
            self._W.proj_()

    def forward(self, X):
        return torch.matmul(self._W, torch.matmul(X, self._W.transpose(2, 1)))


class Transformations(nn.Module):
    def __init__(self, d, n_freq, device, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        self.translation = Translation(d, n_freq, device)
        self.rotation = Rotation(d, n_freq,device)

    def forward(self, X):
        Y = self.translation(X)
        Y = self.rotation(Y)
        return Y
    
    
class FeaturesKernel(BaseEstimator, TransformerMixin):
    
    def __init__(self, sigma=1.):
        self.sigma = sigma
    
    def fit(self, X, y=None):
        self.X = X.astype(np.float64)
        self.N =  np.sum(self.X ** 2, axis=(2, 3))
#         print(self.N)
        return self
    
    def transform(self, X, y=None):
        C = 1.
        X_d = X.astype(np.float64)
        
#         print("??", self.N)
        
        N = np.sum(X_d ** 2, axis=(2, 3))
        for i in range(X_d.shape[1]):
            C1 = self.N[None, :, i] + N[:, i, None]
            C2 = X_d[:, i].reshape(X_d.shape[0], -1) @ self.X[:, i].reshape(self.X.shape[0], -1).T
            C_current = np.exp(-(C1 - 2 * C2) / (self.sigma ** 2))
            C += C_current
        
        return C 
    
    def get_params(self, deep=True):
        return {"sigma": self.sigma}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    
def get_svc(Xs, Xt, ys, yt):
    log_Xs = linalg.sym_logm(Xs).detach().cpu().numpy()
    log_Xt = linalg.sym_logm(Xt).detach().cpu().numpy()
    
    svc = make_pipeline(
        FeaturesKernel(7),
        GridSearchCV(SVC(), {"C": np.logspace(-2, 2, 10), "kernel": ["precomputed"]}, n_jobs=10)
    )
    svc.fit(log_Xs, ys.cpu())
    return svc.score(log_Xt, yt.cpu())


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    
    results_noalign = np.zeros((5,5))
    results_align = np.zeros((5,5))

    d = 22
    n_classes = 4
    num_projs = 500
    epochs = args.epochs
    
    if args.loss == "spdsw":
        projs = get_projs(num_projs, d, device, torch.float64)
    
    if args.loss == "lew":
        manifold_spd = geoopt.SymmetricPositiveDefinite("LEM")        

    results0 = np.zeros((5,5))
    results1 = np.zeros((5,5))
    
    L_t = []

    for i, s1 in enumerate([1,3,7,8,9]):
        for j, s2 in enumerate([1,3,7,8,9]):
            if s1 != s2:
                Xs, ys = get_data(s1, True, "../dataset/")
                cov_Xs = torch.tensor(get_cov2(Xs), device=device) #, dtype=torch.float32)
                ys = torch.tensor(ys, device=device, dtype=torch.long)-1

                Xt, yt = get_data(s2, True, "../dataset/")
                cov_Xt = torch.tensor(get_cov2(Xt), device=device) #, dtype=torch.float32)
                yt = torch.tensor(yt, device=device, dtype=torch.long)-1
                
                n_freq = cov_Xs.shape[2]
                
                model = Transformations(d, n_freq, device)
                
                if args.loss == "lew":
                    lr = 1e-2
                else:
                    lr = 1e-1
                    
                optimizer = RiemannianSGD(model.parameters(), lr=lr)
                # optimizer = RiemannianAdam(model.parameters(), lr=1e-1)

                L_loss = []
#                 L_t = []
                
                if args.pbar:
                    pbar = trange(epochs)
                else:
                    pbar = range(epochs)
                
                t = time.time()
                for e in pbar:
                    zs = model(cov_Xs)
                                 
                    loss = 0
                    for l in range(zs.shape[2]):
                        if args.loss == "spdsw":
#                             sw = sliced_wasserstein_spd(zs[:,0,l], cov_Xt[:,0,l], num_projs, device, p=2)
#                             sw = spdsw(zs[:,0,l], cov_Xt[:,0,l], projs, device, p=2)
                            sw = sliced_wasserstein_spd(zs[:,0,l], cov_Xt[:,0,l], num_projs, device, p=2)
                        elif args.loss == "lew":
                            a = torch.ones((len(zs),), device=device, dtype=torch.float64)/len(zs)
                            b = torch.ones((len(cov_Xt),), device=device, dtype=torch.float64)/len(cov_Xt)
                            M = manifold_spd.dist(zs[:,0,l][:,None], cov_Xt[:,0,l][None])**2
                            sw = 0.1 * ot.emd2(a, b, M)
                        elif args.loss == "sw_matrix":
                            sw = sliced_wasserstein_matrix(zs[:,0,l], cov_Xt[:,0,l], num_projs, device, p=2)
                        elif args.loss == "sw_logmatrix":
                            sw = sliced_wasserstein_logmatrix(zs[:,0,l], cov_Xt[:,0,l], num_projs, device, p=2)
                        loss += sw

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    L_loss.append(loss.item())
                    
                    if args.pbar:
                        pbar.set_postfix_str(f"loss = {loss.item():.3f}")
                                 
                L_t.append(time.time()-t)

                s_source = get_svc(cov_Xs[:,0], cov_Xt[:,0], ys, yt)
                s_target = get_svc(model(cov_Xs)[:,0], cov_Xt[:,0], ys, yt)

                results0[i, j] = s_source
                results1[i, j] = s_target
                 
                                 
    np.savetxt("./results_no_align.csv", results0, delimiter=",")
    np.savetxt("./results_align_"+ str(args.loss) +".csv", results1, delimiter=",")
    np.savetxt("./runtime_"+str(args.loss)+".csv", L_t, delimiter=",")
    
    for k, row in enumerate(results0):
        print(k, np.sum(row)/4)
                                 
    for k, row in enumerate(results1):
        print(k, np.sum(row)/4)
                
