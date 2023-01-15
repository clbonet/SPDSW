# %%
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

from sklearn.decomposition import PCA

from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1]
RESULTS = os.path.join(EXPERIMENTS, "results/da_subject_projs.csv")
RESULTS_transfs_session = os.path.join(EXPERIMENTS, "results/da_transfs_cross_session_projs.csv")
FIGURE_transfs_session = os.path.join(EXPERIMENTS, "figures/Evolution_accuracy_transfs_session_projs.pdf")

# plt.style.use(
#     os.path.join(EXPERIMENTS, 'figures_scripts/figures_style.mplstyle')
# )

def get_mean_subject(results_subject):
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


def get_mean_session(results_session):
    distances = results_session["distance"].unique()
    subjects = results_session["subject"].unique()
    projs = results_session["n_proj"].unique()

    L_session = {}

    for n_projs in projs:
        L_session[n_projs] = {}
        for distance in distances:
            L_session[n_projs][distance] = {}
        L_session[n_projs]["no_align"] = {}

    for n_projs in projs:
        for distance in distances:
            for s in subjects:
                dico_distances = {"mean_score": 0, "std_score": 0, "mean_time": 0, "std_time":0}

                scores = results_session[
                    (results_session["distance"] == distance)
                    & (results_session["subject"] == s)
                    & (results_session["n_proj"] == n_projs)
                ]["align"]
                m_score = scores.mean()
                std_score = scores.std()

                times = results_session[
                    (results_session["distance"] == distance)
                    & (results_session["subject"] == s)
                    & (results_session["n_proj"] == n_projs)
                ]["time"]
                m_t = times.mean()
                std_t = times.std()

                dico_distances["mean_score"] = m_score
                dico_distances["std_score"] = std_score
                dico_distances["mean_time"] = m_t
                dico_distances["std_time"] = std_t

                L_session[n_projs][distance][s] = dico_distances


        for s in subjects:
            dico_distances = {"mean_score": 0, "std_score": 0, "mean_time": 0, "std_time":0}

            scores = results_session[
                (results_session["distance"] == distance)
                & (results_session["subject"] == s)
                & (results_session["n_proj"] == n_projs)
            ]["no_align"]
            m_score = scores.mean()
            std_score = scores.std()

            dico_distances["mean_score"] = m_score
            dico_distances["std_score"] = std_score

            L_session[n_projs]["no_align"][s] = dico_distances
            
    acc_session = np.zeros((len(projs), len(subjects), len(distances)+1))
    acc_std_session = np.zeros((len(projs),len(subjects), len(distances)))
    t_session = np.zeros((len(projs),len(subjects), len(distances)))
    t_std_session = np.zeros((len(projs),len(subjects), len(distances)))

    for l, n_projs in enumerate(projs):
        for i, distance in enumerate(distances):
            for j, s in enumerate(subjects):
                acc_session[l, j, i] = L_session[n_projs][distance][s]["mean_score"]
                acc_std_session[l, j, i] = L_session[n_projs][distance][s]["std_score"]
                t_session[l, j, i] = L_session[n_projs][distance][s]["mean_time"]
                t_std_session[l, j, i] = L_session[n_projs][distance][s]["std_time"]

        distance = "no_align"        
        for j, s in enumerate(subjects):
            acc_session[l, j, -1] = L_session[n_projs][distance][s]["mean_score"]
            
    return acc_session, acc_std_session


def run_plot(results_mean, results_std, projs, subjects_src, FIGURE, d=22):
    plt.figure(figsize=(3,3))

    for j, s in enumerate(subjects_src):
        plt.plot(projs, results_mean[:,j,0], label="Subject "+str(s))
#         plt.plot(projs, results_mean[:,j,0], label=str(s))
        plt.fill_between(projs, results_mean[:,j,0]-results_std[:,j,0], 
                         results_mean[:,j,0]+results_std[:,j,0], alpha=0.5)
    plt.legend(fontsize=13) #, ncol=3, loc="lower center")
    plt.xlabel("Number of projections", fontsize=13)
    plt.ylabel("Accuracy", fontsize=13)
    # plt.xticks(projs)
    plt.grid(True)

    plt.savefig(FIGURE, bbox_inches="tight")
    
    
if __name__ == "__main__":

    results = pd.read_csv(RESULTS_transfs_session)
    
    subjects_src = results["subject"].unique()
    projs = results["n_proj"].unique()
    
    results_mean, results_std = get_mean_session(results)
    run_plot(results_mean, results_std, projs, subjects_src, FIGURE_transfs_session)