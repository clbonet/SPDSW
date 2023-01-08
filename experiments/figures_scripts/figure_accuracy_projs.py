# %%
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

from sklearn.decomposition import PCA

from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1]
RESULTS = os.path.join(EXPERIMENTS, "results/da_subject_projs.csv")
FIGURE = os.path.join(EXPERIMENTS, "figures/Evolution_accuracy_projs.pdf")

# plt.style.use(
#     os.path.join(EXPERIMENTS, 'figures_scripts/figures_style.mplstyle')
# )

def get_mean(results_subject):
    distances = results_subject["distance"].unique()
    subjects_src = results_subject["subject"].unique()
    subjects_tgt = results_subject["target_subject"].unique()
    projs = results_subject["n_proj"].unique()
    
    L_subject = {}
    for n_projs in projs:
        L_subject[n_projs] = {}
        for distance in distances:
            L_subject[n_projs][distance] = {}
        L_subject[n_projs]["no_align"] = {}


    for n_projs in projs:
        for distance in distances:
            for s1 in subjects_src:
                L_subject[n_projs][distance][s1] = {}
                for s2 in subjects_tgt:
                    if s1 != s2:
                        L_subject[n_projs][distance][s1][s2] = {}

                        scores = results_subject[
                            (results_subject["distance"] == distance)
                            & (results_subject["subject"] == s1)
                            & (results_subject["target_subject"] == s2)
                            & (results_subject["n_proj"] == n_projs)
                        ]["align"]
                        m_score = scores.mean()
                        std_score = scores.std()

                        times = results_subject[
                            (results_subject["distance"] == distance)
                            & (results_subject["subject"] == s1)
                            & (results_subject["target_subject"] == s2)
                            & (results_subject["n_proj"] == n_projs)
                        ]["time"]

                        m_t = times.mean()
                        std_t = times.std()

                        L_subject[n_projs][distance][s1][s2]["mean_score"] = m_score
                        L_subject[n_projs][distance][s1][s2]["std_score"] = std_score
                        L_subject[n_projs][distance][s1][s2]["mean_time"] = m_t
                        L_subject[n_projs][distance][s1][s2]["std_t"] = std_t


        for s1 in subjects_src:
            L_subject[n_projs]["no_align"][s1] = {}
            for s2 in subjects_tgt:
                if s1 != s2:
                    dico_distances = {"mean_score": 0, "std_score": 0, "mean_time": 0, "std_time":0}

                    scores = results_subject[
                        (results_subject["distance"] == distance)
                        & (results_subject["subject"] == s1)
                        & (results_subject["target_subject"] == s2)
                        & (results_subject["n_proj"] == n_projs)
                    ]["no_align"]
                    m_score = scores.mean()
                    std_score = scores.std()

                    dico_distances["mean_score"] = m_score
                    dico_distances["std_score"] = std_score

                    L_subject[n_projs]["no_align"][s1][s2] = dico_distances
                    
                    
    results_mean = np.zeros((len(projs), len(subjects_src), len(distances)))
    results_std = np.zeros((len(projs), len(subjects_src), len(distances)))
    t_subjects = np.zeros((len(projs), len(subjects_src), len(distances)))

    for k, n_projs in enumerate(projs):
        for l, distance in enumerate(distances):
            results = np.zeros((5,5))
            stds = np.zeros((5,5))
            ts = np.zeros((5,5))
            for i, s1 in enumerate(subjects_src):
                for j, s2 in enumerate(subjects_tgt):
                    if s1 != s2:
                        results[i, j] = L_subject[n_projs][distance][s1][s2]["mean_score"]
                        stds[i, j] = L_subject[n_projs][distance][s1][s2]["std_score"]
                        if distance != "no_align":
                            ts[i, j] = L_subject[n_projs][distance][s1][s2]["mean_time"]


                results_mean[k, i, l] = np.sum(results[i, :])/4
                results_std[k, i, l] = np.sum(stds[i,:])/4

                if distance != "no_align":
                    t_subjects[k, i, l] = np.sum(ts[i, :])/4

    return results_mean, results_std


def run_plot(results_mean, results_std, projs, subjects_src, d=22):
    plt.figure(figsize=(3,3))

    for j, s in enumerate(subjects_src):
        plt.plot(projs, results_mean[:,j,0], label="Subject "+str(s))
        plt.fill_between(projs, results_mean[:,j,0]-results_std[:,j,0], 
                         results_mean[:,j,0]+results_std[:,j,0], alpha=0.5)
    plt.legend(fontsize=13)
    plt.xlabel("Number of projections", fontsize=13)
    plt.ylabel("Accuracy", fontsize=13)
    # plt.xticks(projs)
    plt.grid(True)

    plt.savefig(FIGURE, bbox_inches="tight")
    
    
if __name__ == "__main__":

    results_subject = pd.read_csv(RESULTS)
    
#     distances = results_subject["distance"].unique()
    subjects_src = results_subject["subject"].unique()
#     subjects_tgt = results_subject["target_subject"].unique()
    projs = results_subject["n_proj"].unique()
    
    results_mean, results_std = get_mean(results_subject)
    run_plot(results_mean, results_std, projs, subjects_src)