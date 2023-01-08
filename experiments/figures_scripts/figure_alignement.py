# %%
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns

from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1]
RESULTS = os.path.join(EXPERIMENTS, "results/data_alignment.csv")
FIGURE = os.path.join(EXPERIMENTS, "figures/figure_alignment_particles.pdf")
PATH_DATA = os.path.join(EXPERIMENTS, "data_bci/")

# %%


def run_plot(results):

    sns.set_theme(style="ticks")
    sns.jointplot(
        results,
        x="x1",
        y="x2",
        hue="session",
        kind="scatter",
        height=5,
        alpha=0.7
    )

    plt.yticks([])
    plt.xticks([])
    plt.ylabel("")
    plt.xlabel("")
    plt.ylim([results["x2"].min() - 0.5, results["x2"].max() + 0.5])
    plt.xlim([results["x1"].min() - 0.5, results["x1"].max() + 0.5])

    plt.savefig(FIGURE, bbox_inches="tight")


# %%
if __name__ == "__main__":

    results = pd.read_csv(RESULTS)
    run_plot(results[["x1", "x2", "session"]])


# %%
