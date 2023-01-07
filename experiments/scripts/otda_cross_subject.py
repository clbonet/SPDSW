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

from geoopt import linalg

from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

from utils.download_bci import download_bci
from utils.get_data import get_data, get_cov, get_cov2
from utils.otda import otda


parser = argparse.ArgumentParser()
parser.add_argument("--ntry", type=int, default=1, help="number of restart")
args = parser.parse_args()

N_JOBS = 10
SEED = 2022
NTRY = args.ntry
EXPERIMENTS = Path(__file__).resolve().parents[1]
PATH_DATA = os.path.join(EXPERIMENTS, "data_bci/")
RESULTS = os.path.join(EXPERIMENTS, "results/otda_subject.csv")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float64
RNG = np.random.default_rng(SEED)
mem = Memory(
    location=os.path.join(EXPERIMENTS, "scripts/tmp_otda_subject/"),
    verbose=0
)

# Set to True to download the data in experiments/data_bci
DOWNLOAD = False #True #False

if DOWNLOAD:
    path_data = download_bci(EXPERIMENTS)


def get_svc(Xs, Xt, ys, yt, d):

    log_Xs = linalg.sym_logm(Xs).detach().cpu().reshape(-1, d * d)
    log_Xt = linalg.sym_logm(Xt).detach().cpu().reshape(-1, d * d)

    clf = GridSearchCV(
        LinearSVC(),
        {"C": np.logspace(-2, 2, 100)},
        n_jobs=N_JOBS
    )
    clf.fit(log_Xs, ys.cpu())
    return clf.score(log_Xt, yt.cpu())


@mem.cache
def run_test(params):

    distance = params["distance"]
    seed = params["seed"]
    subject = params["subject"]
    reg = params["reg"]
#     multifreq = params["multifreq"]

    cross_subject = params["cross_subject"]
    target_subject = params["target_subject"]

#     if multifreq:
#         get_cov_function = get_cov2
#     else:
    get_cov_function = get_cov

    if cross_subject:
        if target_subject == subject:
            return 1., 1.#, 0

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


    if distance == "aiwotda":
        Xs2 = otda(cov_Xs[:,0,0], cov_Xt[:,0,0], metric="ai", loss="emd")
    elif distance == "aisotda":
        Xs2 = otda(cov_Xs[:,0,0], cov_Xt[:,0,0], metric="ai", loss="sinkhorn", reg=reg)
    elif distance == "lewotda":
        Xs2 = otda(cov_Xs[:,0,0], cov_Xt[:,0,0], metric="le", loss="emd")
    elif distance == "lesotda":
        Xs2 = otda(cov_Xs[:,0,0], cov_Xt[:,0,0], metric="le", loss="sinkhorn", reg=reg)       
                
    s_noalign = get_svc(cov_Xs[:, 0], cov_Xt[:, 0], ys, yt, d)
    s_align = get_svc(Xs2, cov_Xt[:, 0], ys, yt, d)

    return s_noalign, s_align


if __name__ == "__main__":

    hyperparams = {
        "distance": ["aiwotda", "aisotda", "lewotda", "lesotda"],
        "seed": RNG.choice(10000, NTRY, replace=False),
        "subject": [1, 3, 7, 8, 9],
        "target_subject": [1, 3, 7, 8, 9],
        "cross_subject": [True],
        "reg": [1],
#         "multifreq": [False]
    }

    keys, values = zip(*hyperparams.items())
    permuts_params = [dict(zip(keys, v)) for v in itertools.product(*values)]

    dico_results = {
        "align": [],
        "no_align": []
    }

    for params in permuts_params:
        try:
            print(params)
            if not params["cross_subject"]:
                params["target_subject"] = 0

            s_noalign, s_align = run_test(params)

            # Storing results
            for key in params.keys():
                if key not in dico_results:
                    dico_results[key] = [params[key]]
                else:
                    dico_results[key].append(params[key])

            dico_results["align"].append(s_align)
            dico_results["no_align"].append(s_noalign)

        except (KeyboardInterrupt, SystemExit):
            raise

    results = pd.DataFrame(dico_results)
    results.to_csv(RESULTS)
