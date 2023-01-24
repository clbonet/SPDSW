# Sliced Wasserstein on Symmetric Positive Definite Matrices

## Installation

```
pip install -e .
```

## Experiments

The procedures to reproduce the figures and experiments in the paper are described below.

- Figure 1:
  ```
  python experiments/visu/plot_cone_SPD_log.py
  ```
- Figure 2:
  ```
  python experiments/scripts/runtime.py
  python experiments/figures_scripts/figure_runtime.py
  ```
- Table 1:
  ```
  python experiments/scripts/da_particles.py --task session --ntry 5
  python experiments/scripts/da_transfs.py --task session --ntry 5
  ```
  Then, the results can be obtained by running the jupyter notebooks `experiments/results/parse_results_particles.ipynb` and `experiments/results/parse_results_transformations.ipynb`
- Figure 4a:
  ```
  python experiments/scripts/alignement.py
  python experiments/scripts/figure_alignement.py
  ```
- Figure 4b:
  The script is the same as in Table 1, except that the benchmark is run with `"distance": ["spdsw"]`, and `"n_proj": np.logspace(1, 3, 10, dtype=int)`.

  ```
  python experiments/scripts/da_transfs.py --task session --ntry 5
  python experiments/figure_accuracy_projs.py
  ```

- BCI in appendix:
  All other experiments on BCI for cross subject alignement can be obtained by replacing `--task session` by `--task subject`.

- MEG experiments:
  The code is mainly based on https://github.com/meeg-ml-benchmarks/brain-age-benchmark-paper, on the class `SPDSW` in `spdsw/spdsw.py`, and on the class `KernelRidgeRegression` in `scikit-learn`. The full version will be made available later to respect anonymity.
