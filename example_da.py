# %%

import sys
import torch
import geoopt

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import trange
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from geoopt import linalg, ManifoldParameter, Stiefel
from geoopt.optim import RiemannianSGD, RiemannianAdam
from geoopt.manifolds import SymmetricPositiveDefinite
from copy import deepcopy

from lib.data.get_data import get_data, get_cov
from lib.swspd import sliced_wasserstein_spd


# %%

device = "cuda:0"

subject = 1
Xs, ys = get_data(subject, True, "./dataset/")
cov_Xs = torch.tensor(get_cov(Xs), device=device) #, dtype=torch.float32)
ys = torch.tensor(ys, device=device, dtype=torch.long)-1

Xt, yt = get_data(subject, False, "./dataset/")
cov_Xt = torch.tensor(get_cov(Xt), device=device) #, dtype=torch.float32)
yt = torch.tensor(yt, device=device, dtype=torch.long)-1

# %%
d = 22
n_classes = 4
freq = 0

# %%
class Translation(nn.Module):
    def __init__(self, d, device):
        super().__init__()

        manifold_spdai = geoopt.SymmetricPositiveDefinite("AIM")        
        self._W = ManifoldParameter(torch.eye(d, dtype=torch.double, device=device), manifold=manifold_spdai)

        with torch.no_grad():
            self._W.proj_()

    def forward(self, X):
        return torch.matmul(self._W, torch.matmul(X, self._W.T))

class Rotation(nn.Module):
    def __init__(self, d, device):
        super().__init__()

        manifold = Stiefel()        
        self._W = ManifoldParameter(torch.eye(d, dtype=torch.double, device=device), manifold=manifold)

        with torch.no_grad():
            self._W.proj_()

    def forward(self, X):
        return torch.matmul(self._W, torch.matmul(X, self._W.T))


class Transformations(nn.Module):
    def __init__(self, d, device):
        super().__init__()

        self.translation = Translation(d, device)
        self.rotation = Rotation(d, device)

    def forward(self, X):
        Y = self.translation(X)
        Y = self.rotation(Y)
        return Y


class MLP(nn.Module):
    def __init__(self, d, n_c, h, device):
        super().__init__()
        self.linear1 = nn.Linear(d, h, device=device, dtype=torch.double)
        self.linear2 = nn.Linear(h, n_c, device=device, dtype=torch.double)
        self.batch_norm = nn.BatchNorm1d(d, device=device, dtype=torch.double)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        out = self.batch_norm(X)
        out = F.relu(self.linear1(out))
        out = self.linear2(out)
        return self.sigmoid(out)

# %%

epochs = 500
num_projs = 500

model = Transformations(22, device)
mlp = MLP(d * d, 4, 50, device)
optimizer_riemann = RiemannianSGD(model.parameters(), lr=1e-1)
optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-2)

mt = SymmetricPositiveDefinite()
Id = torch.eye(d, device=device, dtype=torch.double)
logm = lambda x : mt.logmap(x, Id)
sw_loss = []
ce_loss = []
L_loss = []
cross_entropy = torch.nn.CrossEntropyLoss()
pbar = trange(epochs)

for e in pbar:
    zt = model(cov_Xs[:,:,freq])
    log_Xs = logm(zt[:,0,]).reshape(-1, d*d)
    sw = sliced_wasserstein_spd(zt[:,0], cov_Xt[:,0,freq], num_projs, device, p=2)
    ce = 0.01 * cross_entropy(mlp(log_Xs), ys)
    loss = sw + ce
    loss.backward()
    optimizer.step()
    optimizer_riemann.step()
    optimizer.zero_grad()
    optimizer_riemann.zero_grad()
    
    L_loss.append(loss.item())
    sw_loss.append(sw.item())
    ce_loss.append(ce.item())
    
    pbar.set_postfix_str(f"loss = {loss.item():.5f}")
# %%
plt.plot(L_loss)
plt.plot(sw_loss)
plt.plot(ce_loss)
plt.show()


# %%
from sklearn.linear_model import LogisticRegressionCV

log_Xs = logm(cov_Xs[:,0,0]).detach().cpu().reshape(-1, d*d)
log_Xt = logm(cov_Xt[:,0,0]).detach().cpu().reshape(-1, d*d)

# for k in range(1,10):
#     neigh = KNeighborsClassifier(n_neighbors=k)
#     neigh.fit(log_Xs, ys)
#     print(k, neigh.score(log_Xt, yt))

lr = LogisticRegressionCV(Cs=10, max_iter=1000, n_jobs=10)
lr.fit(log_Xs, ys.cpu())
lr.score(log_Xt, yt.cpu())


# %%
log_Xs = logm(model(cov_Xs[:,0,0])).detach().cpu().reshape(-1, d*d)
log_Xt = logm(cov_Xt[:,:,0]).detach().cpu().reshape(-1, d*d)

# for k in range(1,10):
#     neigh = KNeighborsClassifier(n_neighbors=k)
#     neigh.fit(log_Xs, ys)
#     print(k, neigh.score(log_Xt, yt))
    

lr = LogisticRegressionCV(Cs=10, max_iter=1000, n_jobs=10)
lr.fit(log_Xs, ys.cpu())
lr.score(log_Xt, yt.cpu())
# %%
from sklearn.metrics import accuracy_score

accuracy_score(mlp(log_Xt.to("cuda:0")).argmax(axis=1).detach().cpu(), yt.cpu())

# %%
accuracy_score(mlp(log_Xs.to("cuda:0")).argmax(axis=1).detach().cpu(), ys.cpu())
# %%
