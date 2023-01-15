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
RESULTS = os.path.join(EXPERIMENTS, "results/proj_complexity.csv")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"
DTYPE = torch.float64
RNG = np.random.default_rng(SEED)
mem = Memory(
    location=os.path.join(EXPERIMENTS, "scripts/tmp_proj_complexity/"),
    verbose=0
)


if __name__ == "__main__":

    hyperparams = {
        "n_proj": np.logspace(0, 3, num=10, dtype=int),
#         "ds": [2, 50],
        "distance": ["spdsw", "logsw"],
        "seed": RNG.choice(10000, NTRY, replace=False)
    }

#     keys, values = zip(*hyperparams.items())
#     permuts_params = [dict(zip(keys, v)) for v in itertools.product(*values)]
    

    dico_results = {
        "value": [], 
        "distance": [],
        "seed": [],
        "n_proj": [],
        "d": []
    }
        
    
    n_samples = 500
        
    a = torch.ones((n_samples,), device=DEVICE, dtype=DTYPE) / n_samples
    b = torch.ones((n_samples,), device=DEVICE, dtype=DTYPE) / n_samples
    
    seeds = hyperparams["seed"]
    projs = hyperparams["n_proj"]
    
    for t in range(NTRY):
        for d in [2, 20]:
            m0 = D.Wishart(
                torch.tensor([d], dtype=DTYPE).to(DEVICE),
                torch.eye(d, dtype=DTYPE, device=DEVICE)
            )

            x0 = m0.sample((n_samples,))[:, 0]
            x1 = m0.sample((n_samples,))[:, 0]

            for distance in hyperparams["distance"]:
                spdsw = SPDSW(d, 10000, device=DEVICE, dtype=DTYPE,
                              random_state=seeds[t], sampling=distance)
                sw_star = spdsw.spdsw(x0, x1, p=2)
                
                for proj in projs:
                    try:
                        spdsw = SPDSW(d, proj, device=DEVICE, dtype=DTYPE,
                                      random_state=seeds[t], sampling=distance)
                        sw = spdsw.spdsw(x0, x1, p=2)

                        out = torch.abs(sw_star-sw).item()
                    
                    except Exception:
                        out = np.inf
                    
                    print(seeds[t], distance, d, proj, out, flush=True)
                    
                    dico_results["value"].append(out)
                    dico_results["distance"].append(distance)
                    dico_results["seed"].append(seeds[t])
                    dico_results["n_proj"].append(proj)
                    dico_results["d"].append(d)
                    
    results = pd.DataFrame(dico_results)
    results.to_csv(RESULTS)

