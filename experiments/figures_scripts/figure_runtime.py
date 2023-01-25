# %%
import matplotlib.pyplot as plt
import pandas as pd
import os

from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1]
RESULTS = os.path.join(EXPERIMENTS, "results/runtime.csv")
FIGURE = os.path.join(EXPERIMENTS, "figures/figure_runtime_d20.pdf")

# plt.style.use(
#     os.path.join(EXPERIMENTS, 'figures_scripts/figures_style.mplstyle')
# )

# %%
results = pd.read_csv(RESULTS)
# %%

fig = plt.figure(figsize=(4, 2))

n_samples = results["n_samples"].unique()
dim_s = results["ds"].unique()
distances = results["distance"].unique()

dico_distances = {}
for distance in distances:
    dico_distances[distance] = []

d_s = dim_s[2]


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
    if distance == "spdsw":
        label = r"$\mathrm{SPDSW}$"
    elif distance == "lew":
        label = r"$\mathrm{LEW}$"
    elif distance == "sinkhorn":
        label = r"$\mathrm{LES}$"
    elif distance == "aiw":
        label = r"$\mathrm{AIW}$"
    elif distance == "logsw":
        label = r"$\mathrm{\log SW}$"
#     else:
#         print(distance)
    elif distance == "aispdsw":
        label = r"$\mathrm{HSPDSW}$"
        break
    plt.plot(n_samples, dico_distances[distance], label=label, linewidth=2.)

plt.xlabel(r"Number of samples")
plt.ylabel(r"Time (s)")


plt.legend(
    bbox_to_anchor=(1, 1.02),
#     bbox_to_anchor=(1,1.2),
    ncol=1
)
plt.grid(True)
plt.yscale("log")
plt.xscale("log")

plt.tight_layout()
plt.savefig(FIGURE, bbox_inches="tight")

# %%
