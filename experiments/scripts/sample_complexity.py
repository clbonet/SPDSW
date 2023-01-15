import torch
import argparse
import time
import ot
import geoopt
import numpy as np
import pandas as pd
import itertools
import os
import torch.distributions as D

from spdsw.spdsw import SPDSW

from pathlib import Path
from joblib import Memory
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--ntry", type=int, default=20, help="number of restart")
args = parser.parse_args()


SEED = 2022
NTRY = args.ntry
EXPERIMENTS = Path(__file__).resolve().parents[1]
RESULTS = os.path.join(EXPERIMENTS, "results/sample_complexity.csv")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"
DTYPE = torch.float64
RNG = np.random.default_rng(SEED)
mem = Memory(
    location=os.path.join(EXPERIMENTS, "scripts/tmp_sample_complexity/"),
    verbose=0
)


# @mem.cache
# def run_test(params):

#     n_samples = params["n_samples"]
#     ds = params["ds"]
#     distance = params["distance"]
#     n_proj = params["n_proj"]
#     seed = params["seed"]

#     a = torch.ones((n_samples,), device=DEVICE, dtype=DTYPE) / n_samples
#     b = torch.ones((n_samples,), device=DEVICE, dtype=DTYPE) / n_samples

#     m0 = D.Wishart(
#         torch.tensor([ds], dtype=DTYPE).to(DEVICE),
#         torch.eye(ds, dtype=DTYPE, device=DEVICE)
#     )
#     x0 = m0.sample((n_samples,))[:, 0]
#     x1 = m0.sample((n_samples,))[:, 0]

#     if distance in ["lew", "sinkhorn"]:
#         manifold = geoopt.SymmetricPositiveDefinite("LEM")
#     elif distance == "aiw":
#         manifold = geoopt.SymmetricPositiveDefinite("AIM")

#     try:

#         if distance == "sinkhorn":
#             M = manifold.dist(x0[:, None], x1[None])**2
#             out = ot.sinkhorn2(
#                 a, b, M, reg=1, numitermax=numitermax, stopThr=stop_thr
#             )

#         elif distance == "aiw":
#             M = manifold.dist(x0[:, None], x1[None])**2
#             out = ot.emd2(a, b, M)

#         elif distance == "lew":
#             M = manifold.dist(x0[:, None], x1[None])**2
#             out = ot.emd2(a, b, M)

#         elif distance in ["spdsw", "logsw", "sw", "aispdsw"]:
#             spdsw = SPDSW(ds, n_proj, device=DEVICE, dtype=DTYPE,
#                               random_state=seed, sampling=distance)
#             out = spdsw.spdsw(x0, x1, p=2)

#         return out.item()

#     except Exception:

#         return np.inf


if __name__ == "__main__":

#     hyperparams = {
#         "n_samples": np.logspace(1, 3, num=5, dtype=int),
#         "ds": [2, 50],
#         "distance": ["spdsw", "lew"], # ["lew", "aiw", "spdsw", "logsw"],
#         "n_proj": [1000],
#         "seed": RNG.choice(10000, NTRY, replace=False)
#     }

#     keys, values = zip(*hyperparams.items())
#     permuts_params = [dict(zip(keys, v)) for v in itertools.product(*values)]
    

    dico_results = {
        "value": [],
        "distance": [],
        "ds": [],
        "n_samples": [],
        "seed": [],
        "n_projs": []
    }
    
    
    ds = [2, 50]
    n_samples = np.logspace(1, 3, num=5, dtype=int)
    distances = ["spdsw", "lew"]
    seeds = RNG.choice(10000, NTRY, replace=False)
    n_proj = 1000

    
    
    for j, d in enumerate(ds):
        for k in range(NTRY):
            print(k, d, flush=True)
            seed = seeds[k]
            for i, n in enumerate(n_samples):
                m0 = D.Wishart(
                    torch.tensor([d], dtype=DTYPE).to(DEVICE),
                    torch.eye(d, dtype=DTYPE, device=DEVICE)
                )
                x0 = m0.sample((n,))[:, 0]
                x1 = m0.sample((n,))[:, 0]
                
                a = torch.ones((n,), device=DEVICE, dtype=DTYPE) / n
                b = torch.ones((n,), device=DEVICE, dtype=DTYPE) / n
                
                for distance in distances:
                    try:
                        if distance in ["lew", "sinkhorn"]:
                            manifold = geoopt.SymmetricPositiveDefinite("LEM")
                        elif distance == "aiw":
                            manifold = geoopt.SymmetricPositiveDefinite("AIM")

                        if distance == "sinkhorn":
                            M = manifold.dist(x0[:, None], x1[None])**2
                            out = ot.sinkhorn2(
                                a, b, M, reg=1, numitermax=numitermax, stopThr=stop_thr
                            ).item()

                        elif distance == "aiw":
                            M = manifold.dist(x0[:, None], x1[None])**2
                            out = ot.emd2(a, b, M).item()

                        elif distance == "lew":
                            M = manifold.dist(x0[:, None], x1[None])**2
                            out = ot.emd2(a, b, M).item()

                        elif distance in ["spdsw", "logsw", "sw", "aispdsw"]:
                            spdsw = SPDSW(d, n_proj, device=DEVICE, dtype=DTYPE,
                                          random_state=seed, sampling=distance)
                            out = spdsw.spdsw(x0, x1, p=2).item()

                    except:
                        out = np.inf

                    dico_results["value"].append(out)
                    dico_results["distance"].append(distance)
                    dico_results["ds"].append(d)
                    dico_results["n_samples"].append(n)
                    dico_results["seed"].append(seed)
                    dico_results["n_projs"].append(n_proj)


    results = pd.DataFrame(dico_results)
    results.to_csv(RESULTS)

