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
num_geod = 100

def mat2point(tab_mat):
    return np.concatenate([[tab_mat[:,0,0]], [tab_mat[:,1,1]], [tab_mat[:,0,1]]], axis=0).T

theta = np.random.normal(size=(num_geod, 2))
theta = F.normalize(torch.from_numpy(theta), p=2, dim=-1).to(device)

A = theta[:,None] * torch.eye(theta.shape[-1], device=device)

geod_curves = []
ts = torch.linspace(-2,2,100)
for g in range(num_geod):
    geod =[]
    angle = np.random.random()*2*np.pi
    ROT = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    M_base = np.dot(np.dot(ROT,A[g]),ROT.T) 
    # M_base = A[g]  # <-- juste les matrices diagonales avec diag \in S^d-1
    for i in range(len(ts)): 
        geod.append(torch.linalg.matrix_exp(ts[i]*M_base).cpu().numpy())
    geod_curves.append(Line(mat2point(np.array(geod))).c(g).lw(4))


#%% DISPLAY PART

# create cone
height= 4
res=50
light="off" # off, default, glossy, metallic, etc.
color="grey5"
#cone_mesh = Cone(pos=(height/2, 0, 0), r=height*2, height=height, axis=(-1, 0, 0),)
cone_mesh1 = Plane(pos=(height/2, height/2, 0), normal=(0, 0, 1), s=(height, height), c=color, alpha=.7, res=(res, res))
cone_points1= cone_mesh1.points()
cone_mesh2 = Plane(pos=(height/2, height/2, 0), normal=(0, 0, 1), s=(height, height), c=color, alpha=.7, res=(res, res))
cone_points2= cone_mesh2.points()
for i in range(cone_points1.shape[0]):
    cone_points1[i,2]=np.sqrt(cone_points1[i,0]*cone_points1[i,1])
    cone_points2[i,2]=-np.sqrt(cone_points2[i,0]*cone_points2[i,1])
cone_mesh1.points(cone_points1).smooth().compute_normals().lighting(light).phong()
cone_mesh2.points(cone_points2).smooth().compute_normals().lighting(light).phong()



plt = Plotter(N=1, bg='white', axes=1)

plt.at(0).show(cone_mesh1,cone_mesh2, geod_curves, "Log Euclidean metric", zoom=1.1, interactive=1)

# %%
