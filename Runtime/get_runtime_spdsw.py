import sys
import torch
import argparse
import time
import ot

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from tqdm.auto import trange

sys.path.append("../lib")
from swspd import sliced_wasserstein_spd
#from sw import sliced_wasserstein


parser = argparse.ArgumentParser()
parser.add_argument("--ntry", type=int, default=20, help="number of restart")
parser.add_argument("--pbar", action="store_true", help="If yes, plot pbar")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"


ntry = args.ntry

ds = [3, 100]
samples = [int(1e2),int(1e3),int(1e4),int(1e5/2),int(1e5)] #,int(1e6/2)]
projs = [200]

L_swspd = np.zeros((len(ds), len(projs), len(samples), ntry))

if __name__ == "__main__":    
    for i, d in enumerate(ds):
        for k, n_samples in enumerate(samples):            
            m0 = D.Wishart(torch.tensor([2], dtype=torch.float64).to(device), torch.eye(2, dtype=torch.float64, device=device))
            x0 = m0.sample((n_samples,))[:,0]
            x1 = m0.sample((n_samples,))[:,0]
            
            if args.pbar:
                bar = trange(ntry+1)
            else:
                bar = range(ntry+1)

            for j in bar:
                for l, n_projs in enumerate(projs):
                    # try:
                    t0 = time.time()
                    sw = sliced_wasserstein_spd(x0, x1, n_projs, device, p=2)
                    L_swspd[i,l,k,j] = time.time()-t0
                    # except:
                    #     L_swspd[i,l,k,j] = np.inf

                    
    for i, d in enumerate(ds):
        for l, n_projs in enumerate(projs):
            np.savetxt("./Comparison_SWSPDp_projs_"+str(n_projs)+"_d"+str(d), L_swspd[i, l, :, 1:])
