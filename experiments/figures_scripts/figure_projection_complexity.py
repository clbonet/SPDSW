# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1]
RESULTS = os.path.join(EXPERIMENTS, "results/proj_complexity.csv")
FIGURE = os.path.join(EXPERIMENTS, "figures/figure_projection_complexity.pdf")

# plt.style.use(
#     os.path.join(EXPERIMENTS, 'figures_scripts/figures_style.mplstyle')
# )

# %%
results = pd.read_csv(RESULTS)
# %%

fig = plt.figure(figsize=(5.5, 1.5))

n_projs = results["n_proj"].unique()
dim_s = results["d"].unique()
distances = results["distance"].unique()

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

colors = {2:colors[0], 20:colors[3]}

for d in dim_s:
    dico_distances_mean = {}
    for distance in distances:
        dico_distances_mean[distance] = []
        
    dico_distances_std = {}
    for distance in distances:
        dico_distances_std[distance] = []
    
    for n_proj in n_projs:
        for distance in distances:
            dico_distances_mean[distance].append(
                results[
                    (results["n_proj"] == n_proj)
                    & (results["d"] == d)
                    & (results["distance"] == distance)
                ]["value"].mean()
            )
            
            dico_distances_std[distance].append(
                results[
                    (results["n_proj"] == n_proj)
                    & (results["d"] == d)
                    & (results["distance"] == distance)
                ]["value"].std()
            )

    for distance in distances:
        if distance == "spdsw":
            label = r"$\mathrm{SPDSW}$, d="+str(d)
            linestyle = "-"
        elif distance == "lew":
            label = r"$\mathrm{LEW}$, d="+str(d)
            linestyle = "--"
        elif distance == "sinkhorn":
            label = r"$\mathrm{LES}$, d="+str(d)
        elif distance == "aiw":
            label = r"$\mathrm{AIW}$, d="+str(d)
            continue
        elif distance == "logsw":
            label = r"$\mathrm{\log SW}$, d="+str(d)
            linestyle = "--"
    #     else:
    #         print(distance)
        elif distance == "aispdsw":
            label = r"$\mathrm{HSPDSW}$;, d="+str(d)
            break
            
        m = np.array(dico_distances_mean[distance])
        s = np.array(dico_distances_std[distance])
        
        plt.plot(n_projs, m, linestyle=linestyle, label=label, linewidth=1., color=colors[d])
        plt.fill_between(n_projs, m-2*s/10, m+2*s/10, alpha=0.5, color=colors[d])
                         

plt.xlabel(r"Number of projections")
plt.ylabel(r"Monte-Carlo error")


# plt.legend(
#     bbox_to_anchor=(1, 1.02),
# #     bbox_to_anchor=(1,1.2),
#     ncol=1
# )
plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", ncol=2, fontsize=13)
# plt.legend()

plt.grid(True)
plt.yscale("log")
plt.xscale("log")

# plt.tight_layout()
plt.savefig(FIGURE, bbox_inches="tight")

# %%