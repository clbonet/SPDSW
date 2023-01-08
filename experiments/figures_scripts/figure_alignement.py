# %%
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

from sklearn.decomposition import PCA

from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1]
RESULTS = os.path.join(EXPERIMENTS, "results/data_aligned.csv")
RESULTS_src = os.path.join(EXPERIMENTS, "results/data_src.csv")
RESULTS_tgt = os.path.join(EXPERIMENTS, "results/data_tgt.csv")
FIGURE = os.path.join(EXPERIMENTS, "figures/figure_alignment_particles.pdf")
PATH_DATA = os.path.join(EXPERIMENTS, "data_bci/")

# plt.style.use(
#     os.path.join(EXPERIMENTS, 'figures_scripts/figures_style.mplstyle')
# )

# %%
results = pd.read_csv(RESULTS)
# %%

def run_plot(log_Xs, log_Xt, log_X, d=22):
    fig = plt.figure(figsize=(6, 4))
    
    log_data = np.concatenate([log_Xs, log_Xt, log_X], axis=0)
    X_embedded = PCA(n_components=2).fit_transform(log_data)

    plt.scatter(X_embedded[:len(log_Xs),0], X_embedded[:len(log_Xs),1], c="blue", label="Session 1")
    plt.scatter(X_embedded[len(log_Xs):len(log_Xs)+len(log_Xt),0], X_embedded[len(log_Xs):len(log_Xs)+len(log_Xt),1], c="red", label="Session 2")
    plt.scatter(X_embedded[len(log_Xs)+len(log_Xt):,0], X_embedded[len(log_Xs)+len(log_Xt):,1], c="green", label="Session 1 aligned")

    plt.title("PCA")

    plt.legend()
    plt.xticks([])
    plt.yticks([])

    plt.savefig(FIGURE, bbox_inches="tight")
    
    
if __name__ == "__main__":

    log_Xs = np.loadtxt(RESULTS_src, delimiter=",")
    log_Xt = np.loadtxt(RESULTS_tgt, delimiter=",")
    log_X = np.loadtxt(RESULTS, delimiter=",")
    
    run_plot(log_Xs, log_Xt, log_X)
    
