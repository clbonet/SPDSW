# %%
import matplotlib.pyplot as plt
import pandas as pd
import os

from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1]
RESULTS = os.path.join(EXPERIMENTS, "results/runtime.csv")
FIGURE = os.path.join(EXPERIMENTS, "figures/figure_runtime.pdf")

plt.style.use(
    os.path.join(EXPERIMENTS, 'figures_scripts/figures_style.mplstyle')
)

# %%
results = pd.read_csv(RESULTS)
# %%

fig = plt.figure(figsize=(2, 1.5))

n_samples = results["n_samples"].unique()
dim_s = results["ds"].unique()
distances = results["distance"].unique()

dico_distances = {}
for distance in distances:
    dico_distances[distance] = []

d_s = dim_s[0]

for n_s in n_samples:
    for distance in distances:
        dico_distances[distance].append(
            results[
                (results["n_samples"] == n_s)
                & (results["ds"] == d_s)
                & (results["distance"] == distance)
            ]["time"].mean()
        )

for distance in distances:
    plt.plot(n_samples, dico_distances[distance], label=distance)

plt.xlabel(r"Number of samples in each distribution")
plt.ylabel(r"Seconds")

plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", ncol=2)
plt.grid(True)
plt.yscale("log")
plt.xscale("log")

plt.tight_layout()
plt.savefig(FIGURE, bbox_inches="tight")

# %%
