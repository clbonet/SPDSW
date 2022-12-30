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

device = "cuda:1"

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
n_freq = cov_Xs.shape[2]

# %%
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


class MLP(nn.Module):
    def __init__(self, d, n_c, h, device, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        self.linear1 = nn.Linear(d, h, device=device, dtype=torch.double)
        self.linear2 = nn.Linear(h, n_c, device=device, dtype=torch.double)
        self.batch_norm = nn.BatchNorm1d(d, device=device, dtype=torch.double)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        # out = self.batch_norm(X)
        out = F.relu(self.linear1(X))
        out = self.linear2(out)
        return self.sigmoid(out)

# %%
from sklearn.metrics import accuracy_score

# import ot
# manifold_spdai = geoopt.SymmetricPositiveDefinite("AIM")

epochs = 100
num_projs = 500

seed = 2022

model = Transformations(d, n_freq, device, seed)
# mlp = MLP(d * d, 4, 50, device, seed)
optimizer_riemann = RiemannianAdam(model.parameters(), lr=1e-1)
# optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-2)

sw_loss = []
ce_loss = []
L_loss = []
loss_train = []
loss_test = []
cross_entropy = torch.nn.CrossEntropyLoss()
pbar = trange(epochs)

# log_Xt = linalg.sym_logm(cov_Xt[:,0,]).reshape(-1, d*d)

for e in pbar:
    loss = 0
    zt = model(cov_Xs)
    log_Xs = linalg.sym_logm(zt[:,0,]).reshape(-1, d*d)
    for i in range(zt.shape[2]):
        sw = sliced_wasserstein_spd(
            zt[:,0, i],
            cov_Xt[:,0, i],
            num_projs,
            device,
            p=2
        )
        loss += sw

    # a = torch.ones((len(zt),), device=device, dtype=torch.float64)/len(zt)
    # b = torch.ones((len(cov_Xt),), device=device, dtype=torch.float64)/len(cov_Xt)

    # M = manifold_spdai.dist(zt[:,0][:,None], cov_Xt[:,0,freq][None])**2
    # sw = 0.0001 * ot.emd2(a, b, M)

    # ce = 0.01 * cross_entropy(mlp(log_Xs), ys)
    loss.backward()
    
    # optimizer.step()
    optimizer_riemann.step()
    # optimizer.zero_grad()
    optimizer_riemann.zero_grad()
    
    # loss_train.append(
    #     accuracy_score(mlp(log_Xs).argmax(axis=1).detach().cpu(), ys.cpu())
    # )
    
    # loss_test.append(
    #     accuracy_score(mlp(log_Xt).argmax(axis=1).detach().cpu(), yt.cpu())
    # )
    
    # # zt = model2(cov_Xs[:,:,freq])
    # zt = cov_Xs[:,:,freq]
    # # log_Xs = logm(zt[:,0,]).reshape(-1, d*d)
    # log_Xs = linalg.sym_logm(zt[:,0,]).reshape(-1, d*d)
    # loss2 = 0.01 * cross_entropy(mlp2(log_Xs), ys)
    # loss2.backward()
    # optimizer2.step()
    # optimizer_riemann2.step()
    # optimizer2.zero_grad()
    # optimizer_riemann2.zero_grad()
    
    
    # loss_train_2.append(
    #     accuracy_score(mlp2(log_Xs).argmax(axis=1).detach().cpu(), ys.cpu())
    # )
    
    # loss_test_2.append(
    #     accuracy_score(mlp2(log_Xt).argmax(axis=1).detach().cpu(), yt.cpu())
    # )
    
    # L_loss.append(loss.item())
    sw_loss.append(loss.item())
    # ce_loss.append(ce.item())
    
    pbar.set_postfix_str(
        f"loss = {loss.item():.5f}"
        # f" - accuracy source = {loss_train[-1]:5f}"
        # f" - accuracy target = {loss_test[-1]:.5f}"
    )
# %%
fig, axs = plt.subplots(1, 2)

# axs[0].plot(L_loss, label="loss")
axs[0].plot(sw_loss, label="SPDSW")
# axs[0].plot(ce_loss, label="Cross-entropy")
axs[0].legend()

# axs[1].plot(loss_train, label="source acc. SPDSW")
# axs[1].plot(loss_test, label="target acc. SPDSW")
# axs[1].plot(loss_train_2, label="source acc. baseline ")
# axs[1].plot(loss_test_2, label="target acc. baseline")
# axs[1].legend(loc="lower right")

plt.tight_layout()
# plt.savefig("loss.png")
plt.show()


# %%
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin


class FeaturesKernel(BaseEstimator, TransformerMixin):
    
    def __init__(self, sigma=1.):
        self.sigma = sigma
    
    def fit(self, X, y=None):
        self.X = X.astype(np.float64)
        self.N =  np.sum(self.X ** 2, axis=(2, 3))
        return self
    
    def transform(self, X, y=None):
        C = 1.
        X_d = X.astype(np.float64)
        
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

# %%
log_Xs = linalg.sym_logm(cov_Xs[:, 0]).detach().cpu().numpy()
log_Xt = linalg.sym_logm(cov_Xt[:, 0]).detach().cpu().numpy()

svc = make_pipeline(
    FeaturesKernel(),
    GridSearchCV(SVC(), {"C": np.logspace(-2, 2, 10), "kernel": ["precomputed"]})
)
# svc = LinearSVC(C = 0.1, intercept_scaling=1., max_iter=10000, loss='hinge', multi_class='ovr', penalty='l2', random_state=1, tol=0.00001)
svc_gs = GridSearchCV(svc, {"featureskernel__sigma": np.logspace(0, 2, 20)}, n_jobs=20)

svc_gs.fit(log_Xs, ys.cpu())
print(svc.score(log_Xs, ys.cpu()))
print(svc.score(log_Xt, yt.cpu()))


# %%
log_Xs = linalg.sym_logm(model(cov_Xs)[:, 0]).detach().cpu().numpy()
log_Xt = linalg.sym_logm(cov_Xt[:, 0]).detach().cpu().numpy()


svc = make_pipeline(
    FeaturesKernel(7),
    GridSearchCV(SVC(), {"C": np.logspace(-2, 2, 10), "kernel": ["precomputed"]}, n_jobs=10)
)

# svc = LinearSVC(C = 0.1, intercept_scaling=1., max_iter=10000, loss='hinge', multi_class='ovr', penalty='l2', random_state=1, tol=0.00001)

svc.fit(log_Xs, ys.cpu())
print(svc.score(log_Xs, ys.cpu()))
print(svc.score(log_Xt, yt.cpu()))

# # %%
# from sklearn.metrics import accuracy_score

# accuracy_score(mlp(log_Xt.to("cuda:0")).argmax(axis=1).detach().cpu(), yt.cpu())

# # %%
# accuracy_score(mlp(log_Xs.to("cuda:0")).argmax(axis=1).detach().cpu(), ys.cpu())
# # %%

# %%
