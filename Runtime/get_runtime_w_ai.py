import sys
import torch
import argparse
import time
import ot
import geoopt

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from tqdm.auto import trange

sys.path.append("../lib")
#from swspd import sliced_wasserstein_spd
#from sw import sliced_wasserstein


parser = argparse.ArgumentParser()
parser.add_argument("--ntry", type=int, default=20, help="number of restart")
parser.add_argument("--pbar", action="store_true", help="If yes, plot pbar")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"


ntry = args.ntry

ds = [3, 100]
samples = [int(1e2),int(1e3),int(1e4),int(1e5/2),int(1e5)] #,int(1e6/2)]
#projs = [200]

L_w = np.zeros((len(ds), len(samples), ntry))
#L_swspd = np.zeros((len(ds), len(projs), len(samples), ntry))

manifold_le = geoopt.SymmetricPositiveDefinite("AIM")


if __name__ == "__main__":    
    for i, d in enumerate(ds):
        for k, n_samples in enumerate(samples):
            a = torch.ones((n_samples,), device=device, dtype=torch.float64)/n_samples
            b = torch.ones((n_samples,), device=device, dtype=torch.float64)/n_samples
            
            m0 = D.Wishart(torch.tensor([2], dtype=torch.float64).to(device), torch.eye(2, dtype=torch.float64, device=device))
            x0 = m0.sample((n_samples,))[:,0]
            x1 = m0.sample((n_samples,))[:,0]
            
            if args.pbar:
                bar = trange(ntry)
            else:
                bar = range(ntry)

            for j in bar:
                try:
                    t0 = time.time()
                    M = manifold_le.dist(x0[:,None], x1[None])**2
                    w = ot.emd2(a, b, M)
                    L_w[k, i] = w.item()
                    L_w[i,k,j] = time.time()-t0
                except:
                    L_w[i,k,j] = np.inf

                    
    for i, d in enumerate(ds):
        np.savetxt("./Comparison_AIW_d"+str(d), L_w[i])
