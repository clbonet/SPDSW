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
from sklearn.svm import SVC, LinearSVC
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from geoopt import linalg, ManifoldParameter, Stiefel
from geoopt.optim import RiemannianSGD, RiemannianAdam
from copy import deepcopy

sys.path.append("../lib/data")
from get_data import get_data, get_cov

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
    
    
def get_svc(Xs, Xt, ys, yt):

    log_Xs = linalg.sym_logm(Xs).detach().cpu().reshape(-1, d*d)
    log_Xt = linalg.sym_logm(Xt).detach().cpu().reshape(-1, d*d)

    clf = make_pipeline(GridSearchCV(LinearSVC(), {"C": np.logspace(-2, 2, 10)}, n_jobs=10))
    clf.fit(log_Xs, ys.cpu())
    return clf.score(log_Xt, yt.cpu())



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
#     device = "cpu"
    
#     print(device, flush=True)
#     print(torch.cuda.get_arch_list(), flush=True)
    
    results_noalign = np.zeros((5,5))
    results_align = np.zeros((5,5))

    d = 22
    n_classes = 4
    num_projs = 500
    epochs = args.epochs
    
#     if args.loss == "spdsw":
#         projs = get_projs(num_projs, d, device, torch.float64)
    
    if args.loss == "lew" or args.loss == "les":
        manifold_spd = geoopt.SymmetricPositiveDefinite("LEM")        

    
    results_source = []
    results_target = []
    L_t = []

    for i, s1 in enumerate([1,3,7,8,9]):
        Xs, ys = get_data(s1, True, "../dataset/")
        cov_Xs = torch.tensor(get_cov(Xs), device=device) #, dtype=torch.float32)
        ys = torch.tensor(ys, device=device, dtype=torch.long)-1

        Xt, yt = get_data(s1, False, "../dataset/")
        cov_Xt = torch.tensor(get_cov(Xt), device=device) #, dtype=torch.float32)
        yt = torch.tensor(yt, device=device, dtype=torch.long)-1

        n_freq = cov_Xs.shape[2]

        model = Transformations(d, n_freq, device)

        if args.loss == "lew" or args.loss == "les":
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

            if args.loss == "spdsw":
#                             sw = sliced_wasserstein_spd(zs[:,0,l], cov_Xt[:,0,l], num_projs, device, p=2)
#                             sw = spdsw(zs[:,0,l], cov_Xt[:,0,l], projs, device, p=2)
                sw = sliced_wasserstein_spd(zs[:,0,0], cov_Xt[:,0,0], num_projs, device, p=2)
            elif args.loss == "lew":
                a = torch.ones((len(zs),), device=device, dtype=torch.float64)/len(zs)
                b = torch.ones((len(cov_Xt),), device=device, dtype=torch.float64)/len(cov_Xt)
                M = manifold_spd.dist(zs[:,0,0][:,None], cov_Xt[:,0,0][None])**2
                sw = 0.1 * ot.emd2(a, b, M)
            elif args.loss == "les":
                a = torch.ones((len(zs),), device=device, dtype=torch.float64)/len(zs)
                b = torch.ones((len(cov_Xt),), device=device, dtype=torch.float64)/len(cov_Xt)
                M = manifold_spd.dist(zs[:,0,0][:,None], cov_Xt[:,0,0][None])**2
                sw = 0.1 * ot.sinkhorn2(a, b, M, 1)
            elif args.loss == "sw_matrix":
                sw = sliced_wasserstein_matrix(zs[:,0,0], cov_Xt[:,0,0], num_projs, device, p=2)
            elif args.loss == "sw_logmatrix":
                sw = sliced_wasserstein_logmatrix(zs[:,0,0], cov_Xt[:,0,0], num_projs, device, p=2)
            loss = sw

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            L_loss.append(loss.item())

            if args.pbar:
                pbar.set_postfix_str(f"loss = {loss.item():.3f}")

        L_t.append(time.time()-t)

        s_source = get_svc(cov_Xs[:,0], cov_Xt[:,0], ys, yt)
        s_target = get_svc(model(cov_Xs)[:,0], cov_Xt[:,0], ys, yt)

        results_source.append(s_source)
        results_target.append(s_target)
                 
                                 
    np.savetxt("./results_cross_session_no_align.csv", results_source, delimiter=",")
    np.savetxt("./results_cross_session_align_"+ str(args.loss) +".csv", results_target, delimiter=",")
    np.savetxt("./runtime_cross_session_"+str(args.loss)+".csv", L_t, delimiter=",")
                