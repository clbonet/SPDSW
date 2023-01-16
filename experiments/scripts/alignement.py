# %%
import pandas as pd
import numpy as np

import ot
import os
import torch
import geoopt
import itertools

from tqdm import trange
from copy import deepcopy

from geoopt import linalg
from geoopt.optim import RiemannianSGD
from pathlib import Path

from sklearn.decomposition import PCA
from spdsw.spdsw import SPDSW
from utils.download_bci import download_bci
from utils.get_data import get_data, get_cov
from utils.models import Transformations


EXPERIMENTS = Path(__file__).resolve().parents[1]
RESULTS = os.path.join(EXPERIMENTS, "results/data_alignment.csv")

SEED = 2022
RNG = np.random.default_rng(SEED)
PATH_DATA = os.path.join(EXPERIMENTS, "data_bci/")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float64

# plt.style.use(
#     os.path.join(EXPERIMENTS, 'figures_scripts/figures_style.mplstyle')
# )


# Set to True to download the data in experiments/data_bci
DOWNLOAD = False

if DOWNLOAD:
    path_data = download_bci(EXPERIMENTS)


def run_test(params):
    distance = params["distance"]
    n_proj = params["n_proj"]
    n_epochs = params["n_epochs"]
    seed = params["seed"]
    subject = params["subject"]
    optim_model = params["model"]

    cross_subject = params["cross_subject"]
    target_subject = params["target_subject"]

    get_cov_function = get_cov

    if cross_subject:
        if target_subject == subject:
            return 1., 1., 0, 0, 0

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

    if distance in ["lew", "les"]:
        a = torch.ones(
            (n_samples_s,),
            device=DEVICE,
            dtype=torch.float64
        ) / n_samples_s
        b = torch.ones(
            (n_samples_t,),
            device=DEVICE,
            dtype=torch.float64
        ) / n_samples_t
        manifold = geoopt.SymmetricPositiveDefinite("LEM")

    elif distance in ["spdsw", "logsw", "sw"]:
        spdsw = SPDSW(
            d,
            n_proj,
            device=DEVICE,
            dtype=DTYPE,
            random_state=seed,
            sampling=distance
        )

    if optim_model == "transformations":
        model = Transformations(d, n_freq, DEVICE, seed=seed)

        optimizer = RiemannianSGD(model.parameters(), lr=1e-1)

        pbar = trange(n_epochs)

        for e in pbar:
            zs = model(cov_Xs)

            if distance == "lew":
                M = manifold.dist(
                    zs[:, 0, 0][:, None], cov_Xt[:, 0, 0][None]
                ) ** 2
                loss = 0.1 * ot.emd2(a, b, M)

            elif distance == "les":
                M = manifold.dist(
                    zs[:, 0, 0][:, None], cov_Xt[:, 0, 0][None]
                ) ** 2
                loss = 0.1 * ot.sinkhorn2(a, b, M, 1)

            elif distance in ["spdsw", "logsw", "sw"]:
                loss = spdsw.spdsw(zs[:, 0, 0], cov_Xt[:, 0, 0], p=2)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            pbar.set_postfix_str(f"loss = {loss.item():.3f}")

        log_X = linalg.sym_logm(
            model(cov_Xs)[:, 0, 0].detach().cpu()
        ).reshape(-1, d*d)

    elif optim_model == "particles":
        manifold = geoopt.SymmetricPositiveDefinite("LEM")

        x = deepcopy(cov_Xs[:, 0, 0]).requires_grad_(True)

        optimizer = RiemannianSGD([x], lr=1000)
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

        log_X = linalg.sym_logm(x.detach().cpu()).reshape(-1, d*d)

    log_Xs = linalg.sym_logm(cov_Xs[:, 0, 0].detach().cpu()).reshape(-1, d*d)
    log_Xt = linalg.sym_logm(cov_Xt[:, 0, 0].detach().cpu()).reshape(-1, d*d)

    return log_Xs, log_Xt, log_X, ys, yt


if __name__ == "__main__":

    hyperparams = {
        "distance": ["spdsw"],
        "n_proj": [1000],
        "n_epochs": [500],
        "seed": RNG.choice(10000, 1, replace=False),
        "subject": [1,3,7,8,9],
        "target_subject": [1,3,7,8,9],
        "cross_subject": [True],
        "model": ["particles"]
    }

    keys, values = zip(*hyperparams.items())
    permuts_params = [dict(zip(keys, v)) for v in itertools.product(*values)]

    dico_results = {
        "x1": [],
        "x2": [],
        "y": [],
        "session": [],
        "subject": [],
        "target_subject": []
    }

    for params in permuts_params:
        try:
            print(params)
            if not params["cross_subject"]:
                params["target_subject"] = 0
                
            print(params["subject"], params["target_subject"])
                
            if params["subject"] != params["target_subject"]:
                log_Xs, log_Xt, log_X, ys, yt = run_test(params)
                log_data = np.concatenate([log_Xs, log_Xt, log_X], axis=0)
                pca = PCA(n_components=2).fit(log_data)

                for points, session, y in zip(
                    [log_Xs, log_Xt, log_X],
                    ["source", "target", "aligned"],
                    [ys, yt, ys]
                ):
                    X_points = pca.transform(points)
                    for i, point in enumerate(X_points):
                        dico_results["x1"].append(point[0])
                        dico_results["x2"].append(point[1])
                        dico_results["y"].append(y.detach().cpu()[i].item())
                        dico_results["session"].append(session)
                        dico_results["subject"].append(params["subject"])
                        dico_results["target_subject"].append(params["target_subject"])                                                     
        except (KeyboardInterrupt, SystemExit):
            raise

    results = pd.DataFrame(dico_results)
    results.to_csv(RESULTS)