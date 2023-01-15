# %%
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns

from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1]
RESULTS = os.path.join(EXPERIMENTS, "results/data_alignment.csv")
FIGURE_alignment = os.path.join(EXPERIMENTS, "figures/figure_alignment_particles.pdf")
FIGURE_classes = os.path.join(EXPERIMENTS, "figures/figure_classes.pdf")
PATH_DATA = os.path.join(EXPERIMENTS, "data_bci/")

# %%


def run_plot(results):

#     sns.set_theme(style="ticks")
#     sns.jointplot(
#         results,
#         x="x1",
#         y="x2",
#         hue="session",
#         kind="scatter",
#         height=4,
#         alpha=0.7
#     )
#     print(results, results["x1"])

    s = 8

    fig = plt.figure(figsize=(4, 4))

    plt.scatter(results[results["session"]=="source"]["x1"], 
                results[results["session"]=="source"]["x2"], 
                c="b", label="Source", s=s)
    plt.scatter(results[results["session"]=="target"]["x1"], 
                results[results["session"]=="target"]["x2"], 
                c="r", label="Target", s=s)
    plt.scatter(results[results["session"]=="aligned"]["x1"], 
                results[results["session"]=="aligned"]["x2"], 
                c="g", label="Aligned", s=s)

    plt.yticks([])
    plt.xticks([])
    plt.ylabel("")
    plt.xlabel("")
    plt.ylim([results["x2"].min() - 0.5, results["x2"].max() + 0.5])
    plt.xlim([results["x1"].min() - 0.5, results["x1"].max() + 0.5])

    plt.legend(title="", fontsize=13)
    plt.savefig(FIGURE_alignment, bbox_inches="tight")
    
    
    fig = plt.figure(figsize=(4, 4))
    
    colors = ["r", "g", "b", "y"]
    
    for k in range(4):
        plt.scatter(results[results["y"]==k][results["session"]=="target"]["x1"], 
                    results[results["y"]==k][results["session"]=="target"]["x2"], 
                    label="Class "+str(k+1), c=colors[k], s=s)
        
        plt.scatter(results[results["y"]==k][results["session"]=="aligned"]["x1"], 
                    results[results["y"]==k][results["session"]=="aligned"]["x2"], 
                    c=colors[k], s=s, marker="x")
        
#     plt.scatter(results[results["session"]=="target"]["x1"], results[results["session"]=="target"]["x2"], c="r", label="Target")
#     plt.scatter(results[results["session"]=="aligned"]["x1"], results[results["session"]=="aligned"]["x2"], c="g", label="Aligned")

    plt.yticks([])
    plt.xticks([])
    plt.ylabel("")
    plt.xlabel("")
    plt.ylim([results["x2"].min() - 0.5, results["x2"].max() + 0.5])
    plt.xlim([results["x1"].min() - 0.5, results["x1"].max() + 0.5])

    plt.legend(title="", fontsize=13)
    plt.savefig(FIGURE_classes, bbox_inches="tight")


# %%
if __name__ == "__main__":

    results = pd.read_csv(RESULTS)
    print(results)
    run_plot(results[["x1", "x2", "y", "session"]])


# %%
