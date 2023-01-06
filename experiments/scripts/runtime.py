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
RESULTS = os.path.join(EXPERIMENTS, "results/runtime.csv")
# DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"
DTYPE = torch.float
RNG = np.random.default_rng(SEED)
mem = Memory(
    location=os.path.join(EXPERIMENTS, "scripts/tmp_runtime/"),
    verbose=0
)


@mem.cache
def run_test(params):

    n_samples = params["n_samples"]
    ds = params["ds"]
    distance = params["distance"]
    n_proj = params["n_proj"]
    numitermax = params["numitermax"]
    stop_thr = params["stop_thr"]
    seed = params["seed"]

    a = torch.ones((n_samples,), device=DEVICE, dtype=DTYPE) / n_samples
    b = torch.ones((n_samples,), device=DEVICE, dtype=DTYPE) / n_samples

    m0 = D.Wishart(
        torch.tensor([ds], dtype=DTYPE).to(DEVICE),
        torch.eye(ds, dtype=DTYPE, device=DEVICE)
    )
    x0 = m0.sample((n_samples,))[:, 0]
    x1 = m0.sample((n_samples,))[:, 0]

    if distance in ["lew", "sinkhorn"]:
        manifold = geoopt.SymmetricPositiveDefinite("LEM")
    elif distance == "aiw":
        manifold = geoopt.SymmetricPositiveDefinite("AIM")

    t0 = time.time()

    try:

        if distance == "sinkhorn":
            M = manifold.dist(x0[:, None], x1[None])**2
            ot.sinkhorn2(
                a, b, M, reg=1, numitermax=numitermax, stopThr=stop_thr
            )

        elif distance == "aiw":
            M = manifold.dist(x0[:, None], x1[None])**2
            ot.emd2(a, b, M)

        elif distance == "lew":
            M = manifold.dist(x0[:, None], x1[None])**2
            ot.emd2(a, b, M)

        elif distance in ["spdsw", "logsw", "sw"]:
            spdsw = SPDSW(ds, n_proj, device=DEVICE, dtype=DTYPE,
                          random_state=seed, sampling=distance)
            spdsw.spdsw(x0, x1, p=2)

        return time.time() - t0

    except Exception:

        return np.inf


if __name__ == "__main__":

    hyperparams = {
        "n_samples": np.logspace(2, 5, num=10, dtype=int),
        "ds": [2, 10],
        "distance": ["lew", "aiw", "sinkhorn", "spdsw"],
        "n_proj": [200],
        "numitermax": [10000],
        "stop_thr": [1e-10],
        "seed": RNG.choice(10000, NTRY, replace=False)
    }

    keys, values = zip(*hyperparams.items())
    permuts_params = [dict(zip(keys, v)) for v in itertools.product(*values)]

    dico_results = {
        "time": []
    }

    for params in tqdm(permuts_params):
        try:
            result = run_test(params)

            # Storing results
            for key in params.keys():
                if key not in dico_results:
                    dico_results[key] = [params[key]]
                else:
                    dico_results[key].append(params[key])

            dico_results["time"].append(result)

        except (KeyboardInterrupt, SystemExit):
            raise

    results = pd.DataFrame(dico_results)
    results.to_csv(RESULTS)
