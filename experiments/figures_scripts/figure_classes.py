import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns

from pathlib import Path


EXPERIMENTS = Path(__file__).resolve().parents[1]

RESULTS_session = os.path.join(EXPERIMENTS, "results/data_alignment_session.csv")
RESULTS_subject = os.path.join(EXPERIMENTS, "results/data_alignment_subject.csv")

# FIGURE_alignment = os.path.join(EXPERIMENTS, "figures/figure_alignment_particles.pdf")

FIGURE_classes_session = os.path.join(EXPERIMENTS, "figures/figure_classes_session.pdf")
FIGURE_classes_subject = os.path.join(EXPERIMENTS, "figures/figure_classes_subject.pdf")

PATH_DATA = os.path.join(EXPERIMENTS, "data_bci/")


def plot_session(results):
    subjects = results["subject"].unique()
    colors = ["r", "g", "b", "y"]

    fig, ax = plt.subplots(1, 5, figsize=(16, 4))


    for i, s in enumerate(subjects):
        for k in range(4):
            ax[i].scatter(results[results["subject"]==s][results["y"]==k][results["session"]=="target"]["x1"], 
                        results[results["subject"]==s][results["y"]==k][results["session"]=="target"]["x2"], 
                        label="Class "+str(k+1), c=colors[k], s=10)

            ax[i].scatter(results[results["subject"]==s][results["y"]==k][results["session"]=="aligned"]["x1"], 
                        results[results["subject"]==s][results["y"]==k][results["session"]=="aligned"]["x2"], 
                        c=colors[k], s=10, marker="x")

    #         ax[i].scatter(results[results["subject"]==s][results["y"]==k][results["session"]=="source"]["x1"], 
    #                     results[results["subject"]==s][results["y"]==k][results["session"]=="source"]["x2"], 
    #                     c=colors[k], s=10, marker="+")


    #     plt.scatter(results[results["session"]=="target"]["x1"], results[results["session"]=="target"]["x2"], c="r", label="Target")
    #     plt.scatter(results[results["session"]=="aligned"]["x1"], results[results["session"]=="aligned"]["x2"], c="g", label="Aligned")

        ax[i].set_yticks([])
        ax[i].set_xticks([])
        ax[i].set_ylabel("")
        ax[i].set_xlabel("")
        ax[i].set_ylim([results["x2"].min() - 0.5, results["x2"].max() + 0.5])
        ax[i].set_xlim([results["x1"].min() - 0.5, results["x1"].max() + 0.5])

        if i==0:
            ax[i].legend(title="", fontsize=13)
        ax[i].set_title("Subject " + str(s))

    plt.savefig(FIGURE_classes_session, bbox_inches="tight")        
#     plt.show()


def plot_subjects(results):
    subjects = results["subject"].unique()
    colors = ["r", "g", "b", "y"]
    
    # fig = plt.figure(figsize=(16, 16))
    fig, ax = plt.subplots(5, 4, figsize=(16,16))

    # i = 0
    j = 0

    for i, s1 in enumerate(subjects):
        j = 0
        for s2 in subjects:
            if s1 != s2:
                for k in range(4):
    #                 print(results[results["subject"]==s1])
                    ax[i,j].scatter(results[results["subject"]==s1][results["target_subject"]==s2][results["y"]==k][results["session"]=="target"]["x1"], 
                                results[results["subject"]==s1][results["target_subject"]==s2][results["y"]==k][results["session"]=="target"]["x2"], 
                                label="Class "+str(k+1), c=colors[k], s=10)

                    ax[i,j].scatter(results[results["subject"]==s1][results["target_subject"]==s2][results["y"]==k][results["session"]=="aligned"]["x1"], 
                                results[results["subject"]==s1][results["target_subject"]==s2][results["y"]==k][results["session"]=="aligned"]["x2"], 
                                c=colors[k], s=10, marker="x")

                    ax[i,j].scatter(results[results["subject"]==s1][results["target_subject"]==s2][results["y"]==k][results["session"]=="source"]["x1"], 
                                results[results["subject"]==s1][results["target_subject"]==s2][results["y"]==k][results["session"]=="source"]["x2"], 
                                c=colors[k], s=10, marker="s")

            #     plt.scatter(results[results["session"]=="target"]["x1"], results[results["session"]=="target"]["x2"], c="r", label="Target")
            #     plt.scatter(results[results["session"]=="aligned"]["x1"], results[results["session"]=="aligned"]["x2"], c="g", label="Aligned")

                ax[i,j].set_yticks([])
                ax[i,j].set_xticks([])
                ax[i,j].set_ylabel("")
                ax[i,j].set_xlabel("")
                ax[i,j].set_ylim([results["x2"].min() - 0.5, results["x2"].max() + 0.5])
                ax[i,j].set_xlim([results["x1"].min() - 0.5, results["x1"].max() + 0.5])

                if i==0 and j==0:
                    ax[i,j].legend(title="", fontsize=13)
                ax[i,j].set_title("Subject " + str(s1) + " to " + str(s2))

                j += 1

    plt.savefig(FIGURE_classes_subject, bbox_inches="tight")        
#     plt.show()


if __name__ == "__main__":
    results_session = pd.read_csv(RESULTS_session)
    plot_session(results_session)
    
    results_subject = pd.read_csv(RESULTS_subject)
    plot_subjects(result_subject)