#%%
from vedo import *

settings.default_font = "Ubuntu"

import torch
import ot
import geoopt
# import scipy.linalg
import sys
device = "cuda" if torch.cuda.is_available() else "cpu"

import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.distributions as D

from tqdm.auto import trange
from geoopt import linalg
from pyriemann.datasets import sample_gaussian_spd, generate_random_spd_matrix

#%%
# Needed functions

def busemann_spd(logM, diagA):
    C = diagA[None] * logM[:,None]
    return -C.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)

def proj_geod(theta, M):
    """
        - theta \in S^{d-1}
        - M: batch_size x d x d: SPD matrices
    """
    A = theta[:,None] * torch.eye(theta.shape[-1], device=device)
    
    ## Preprocessing to compute the matrix product using a simple product
    diagA = torch.diagonal(A, dim1=-2, dim2=-1)
    dA = diagA.unsqueeze(-1)
    dA = dA.repeat(1,1,2)
    
    n_proj, d, _ = dA.shape

    log_M = linalg.sym_logm(M)
    Mp = busemann_spd(log_M, dA).reshape(n_proj, -1)
    
    return torch.exp(-Mp[:,:,None] * diagA[:,None]), Mp

#%% Needed data
def mat2point(tab_mat):
    return np.concatenate([[tab_mat[:,0,0]], [tab_mat[:,1,1]], [tab_mat[:,0,1]]], axis=0).T

mean0 = np.eye(2)
sigma0 = 1
n_samples=10


B0 = torch.tensor(sample_gaussian_spd(n_matrices=n_samples, mean=mean0, sigma=sigma0), device=device)
vect_B0 = mat2point(B0.numpy())

theta = np.random.normal(size=(1, 2))
theta = F.normalize(torch.from_numpy(theta), p=2, dim=-1).to(device)

A = theta[:,None] * torch.eye(theta.shape[-1], device=device)

proj_B0, buseman_coord = proj_geod(theta, B0)

ts = torch.linspace(buseman_coord.min()-0.1,buseman_coord.max()+0.1,100)
geod = []

for i in range(len(ts)): 
    geod.append(torch.linalg.matrix_exp(ts[i]*A[0]).cpu().numpy())
geod = np.array(geod)


#%% DISPLAY PART

# create cone
height= 4
res=50
light="off" # off, default, glossy, metallic, etc.
#cone_mesh = Cone(pos=(height/2, 0, 0), r=height*2, height=height, axis=(-1, 0, 0),)
cone_mesh1 = Plane(pos=(height/2, height/2, 0), normal=(0, 0, 1), s=(height, height), alpha=.7, res=(res, res))
cone_points1= cone_mesh1.points()
cone_mesh2 = Plane(pos=(height/2, height/2, 0), normal=(0, 0, 1), s=(height, height), alpha=.7, res=(res, res))
cone_points2= cone_mesh2.points()
for i in range(cone_points1.shape[0]):
    cone_points1[i,2]=np.sqrt(cone_points1[i,0]*cone_points1[i,1])
    cone_points2[i,2]=-np.sqrt(cone_points2[i,0]*cone_points2[i,1])
cone_mesh1.points(cone_points1).compute_normals().lighting(light)
cone_mesh2.points(cone_points2).compute_normals().lighting(light)

#cone_mesh1.c('indigo1').lc('grey9').lw(0.05)
#cone_mesh2.c('indigo1').lc('grey9').lw(0.05)
#cone_mesh1.lc('grey1').lw(0.1)
#cone_mesh2.lc('grey1').lw(0.1)

geodesic = Line(mat2point(geod)).color('black').lw(2)

## Projections (geodesics between points and projected points)
ts = torch.linspace(0, 1, 100)
tab_proj = []
tab_proj_AI = []

for i in range(len(vect_B0)):
    proj_B0_diag = proj_B0[0,i][:,None] * torch.eye(2)
    geod_le = linalg.sym_expm((1-ts)[:,None,None] * linalg.sym_logm(B0[i]) + 
                              ts[:,None,None] * linalg.sym_logm(proj_B0_diag))
    tab_proj.append(Line(mat2point(geod_le.numpy())).color('blue').lw(1))

    B12 = linalg.sym_sqrtm(B0[i])
    B12_ = linalg.sym_invm(B12)
    log = linalg.sym_logm(torch.matmul(B12_, torch.matmul(proj_B0_diag, B12_)))
    exp = linalg.sym_expm(ts[:,None,None] * log)
    
    geod_ai = torch.matmul(B12, torch.matmul(exp, B12))
    tab_proj_AI.append(Line(mat2point(geod_ai.numpy())).color('blue').lw(1))


s = Spheres(vect_B0, r=.07).c("red")
proj_mat = np.concatenate([[proj_B0[0,:,0].numpy()], [proj_B0[0,:,1].numpy()], [np.zeros(len(vect_B0))]], axis=0).T
s_proj = Spheres(proj_mat, r=.04).c("green")

plt = Plotter(N=2, bg='blackboard', axes=1)

#plt.at(0).show(cone_mesh1,cone_mesh2, geodesic,tab_proj, s,s_proj, zoom=1.1, camera={'pos':(10,10,10), 'focal_point':(0,0,0)}, interactive=1)
plt.at(0).show(cone_mesh1,cone_mesh2, geodesic,tab_proj, s,s_proj, "Log Euclidean metric", zoom=1.1, interactive=0)
plt.at(1).show(cone_mesh1,cone_mesh2, geodesic,tab_proj_AI, s,s_proj, "Affine Invariant metric", zoom=1.1, interactive=1)


# %%
