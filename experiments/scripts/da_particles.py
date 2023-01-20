import warnings
import torch
import argparse
import time
import ot
import geoopt
import numpy as np
import pandas as pd
import itertools
import os

from pathlib import Path
from joblib import Memory
from tqdm import trange
from copy import deepcopy

from geoopt import linalg
from geoopt.optim import RiemannianSGD

from spdsw.spdsw import SPDSW
from utils.download_bci import download_bci
from utils.get_data import get_data, get_cov, get_cov2
from utils.models import Transformations, FeaturesKernel, get_svc


warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
# os.environ["PYTHONWARNINGS"] = "ignore::ConvergenceWarning:sklearn.svm.LinearSVC"

parser = argparse.ArgumentParser()
parser.add_argument("--ntry", type=int, default=1, help="number of restart")
parser.add_argument("--task", type=str, default="session", help="session or subject")
args = parser.parse_args()

N_JOBS = 50
SEED = 2022
NTRY = args.ntry
EXPERIMENTS = Path(__file__).resolve().parents[1]
PATH_DATA = os.path.join(EXPERIMENTS, "data_bci/")
RESULTS = os.path.join(EXPERIMENTS, "results/da_particles.csv")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float64
RNG = np.random.default_rng(SEED)
mem = Memory(
    location=os.path.join(EXPERIMENTS, "scripts/tmp_da_particles/"),
    verbose=0
)

# Set to True to download the data in experiments/data_bci
DOWNLOAD = False

if DOWNLOAD:
    path_data = download_bci(EXPERIMENTS)





@mem.cache
def run_test(params):

    distance = params["distance"]
    n_proj = params["n_proj"]
    n_epochs = params["n_epochs"]
    seed = params["seed"]
    subject = params["subject"]
    multifreq = params["multifreq"]

    cross_subject = params["cross_subject"]
    target_subject = params["target_subject"]

    if multifreq:
        get_cov_function = get_cov2
    else:
        get_cov_function = get_cov

    if cross_subject:
        if target_subject == subject:
            return 1., 1., 0

        Xs, ys = get_data(subject, True, PATH_DATA)
        cov_Xs = torch.tensor(get_cov_function(Xs), device=DEVICE)
        ys = torch.tensor(ys, device=DEVICE, dtype=torch.int) - 1

        Xt, yt = get_data(target_subject, True, PATH_DATA)
        cov_Xt = torch.tensor(get_cov_function(Xt), device=DEVICE)
        yt = torch.tensor(yt, device=DEVICE, dtype=torch.int) - 1

    else:

        Xs, ys = get_data(subject, True, PATH_DATA)
        cov_Xs = torch.tensor(get_cov_function(Xs), device=DEVICE)
        ys = torch.tensor(ys, device=DEVICE, dtype=torch.int) - 1

        Xt, yt = get_data(subject, False, PATH_DATA)
        cov_Xt = torch.tensor(get_cov_function(Xt), device=DEVICE)
        yt = torch.tensor(yt, device=DEVICE, dtype=torch.int) - 1

    d = 22
    n_freq = cov_Xs.shape[2]

    n_samples_s = len(cov_Xs)
    n_samples_t = len(cov_Xt)

    start = time.time()

    if distance in ["lew", "les"]:
        lr = 10
#         a = torch.ones((n_samples_s,), device=DEVICE, dtype=DTYPE) / n_samples_s
#         b = torch.ones((n_samples_t,), device=DEVICE, dtype=DTYPE) / n_samples_t
        a = torch.ones((n_samples_s,), device=DEVICE, dtype=torch.float64) / n_samples_s
        b = torch.ones((n_samples_t,), device=DEVICE, dtype=torch.float64) / n_samples_t

    elif distance in ["spdsw", "logsw", "sw"]:
        lr = 1000
        spdsw = SPDSW(
            d,
            n_proj,
            device=DEVICE,
            dtype=DTYPE,
            random_state=seed,
            sampling=distance
        )

        
    manifold = geoopt.SymmetricPositiveDefinite("LEM")
    
    x = deepcopy(cov_Xs[:,0,0]).requires_grad_(True)

    optimizer = RiemannianSGD([x], lr=lr)
    optimizer._default_manifold = manifold

    pbar = trange(n_epochs)
    

    for e in pbar:
        if distance == "lew":
            M = manifold.dist(x[:, None], cov_Xt[:, 0, 0][None]) ** 2
            loss = 0.1 * ot.emd2(a, b, M)

        elif distance == "les":
            M = manifold.dist(x[:, None], cov_Xt[:, 0, 0][None]) ** 2
            loss = 0.1 * ot.sinkhorn2(a, b, M, 1)

        elif distance in ["spdsw", "logsw", "sw"]:
            loss = spdsw.spdsw(x, cov_Xt[:, 0, 0], p=2)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        pbar.set_postfix_str(f"loss = {loss.item():.3f}")

    stop = time.time()

    s_noalign = get_svc(cov_Xs[:, 0], cov_Xt[:, 0], ys, yt, d)
    s_align = get_svc(x, cov_Xt[:, 0], ys, yt, d)

    return s_noalign, s_align, stop - start


if __name__ == "__main__":
    hyperparams = {
        "distance": ["lew", "les", "spdsw", "logsw"],
        "n_proj": [500],
        "n_epochs": [500],
        "seed": RNG.choice(10000, NTRY, replace=False),
        "subject": [1, 3, 7, 8, 9],
        "target_subject": [1, 3, 7, 8, 9],
#         "cross_subject": [False],
        "multifreq": [False]
    }
    
    if args.task == "session":
        hyperparams["cross_subject"] = [False]
        #hyperparams["n_epochs"] = [50, 500]
        RESULTS = os.path.join(EXPERIMENTS, "results/da_particles_cross_session.csv")
    elif args.task == "subject":
        hyperparams["cross_subject"] = [True]
        #hyperparams["n_epochs"] = [250, 500]
        RESULTS = os.path.join(EXPERIMENTS, "results/da_particles_cross_subject.csv")
        

    keys, values = zip(*hyperparams.items())
    permuts_params = [dict(zip(keys, v)) for v in itertools.product(*values)]

    dico_results = {
        "align": [],
        "no_align": [],
        "time": []
    }

    for params in permuts_params:
        try:
            print(params)
            if not params["cross_subject"]:
                params["target_subject"] = 0
            s_noalign, s_align, runtime = run_test(params)

            # Storing results
            for key in params.keys():
                if key not in dico_results:
                    dico_results[key] = [params[key]]
                else:
                    dico_results[key].append(params[key])

            dico_results["align"].append(s_align)
            dico_results["no_align"].append(s_noalign)
            dico_results["time"].append(runtime)

        except (KeyboardInterrupt, SystemExit):
            raise

    results = pd.DataFrame(dico_results)
    results.to_csv(RESULTS)
