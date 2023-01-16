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

    # sns.set_theme(style="ticks")
    # sns.relplot(
    #     results,
    #     x="x1",
    #     y="x2",
    #     hue="session",
    #     # kind="scatter",
    #     height=4,
    #     alpha=0.7,
    #     legend="auto"
    # )
    fig = plt.figure(figsize=(4, 4))
    
    for session in pd.unique(results["session"]):
        results_session = results.query(f"session == '{session}'")
        session_text = f"{session[0].capitalize()}{session[1:]}"
        plt.scatter(
            results_session["x1"],
            results_session["x2"],
            label=session_text,
            alpha=0.5,
            s=20
        )

    plt.yticks([])
    plt.xticks([])
    plt.ylabel("")
    plt.xlabel("")
    plt.ylim([results["x2"].min() - 0.3, results["x2"].max() + 0.3])
    plt.xlim([results["x1"].min() - 0.5, results["x1"].max() + 0.5])

    plt.legend(title="", fontsize=14, labelspacing=0.2, handletextpad=0.1, borderaxespad=0.2, borderpad=0.2)
    plt.savefig(FIGURE, bbox_inches="tight")



# %%
if __name__ == "__main__":

    results = pd.read_csv(RESULTS)
    run_plot(results[["x1", "x2", "session"]])


# %%
