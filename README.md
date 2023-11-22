# Sliced-Wasserstein on Symmetric Positive Definite Matrices for M/EEG Signals

This repository contains the code to reproduce the experiments of the paper [Sliced-Wasserstein on Symmetric Positive Definite Matrices for M/EEG Signals](https://arxiv.org/abs/2303.05798). We propose in this paper a Sliced-Wasserstein distance on the space of symmetric positive definite matrices endowed with the Log-Euclidean metric.

## Abstract 

When dealing with electro or magnetoencephalography records, many supervised prediction tasks are solved by working with covariance matrices to summarize the signals. Learning with these matrices requires the usage of Riemanian geometry to account for their structure. In this paper, we propose a new method to deal with distributions of covariance matrices, and demonstrate its computational efficiency on M/EEG multivariate time series. More specifically, we define a Sliced-Wasserstein distance between measures of symmetric positive definite matrices that comes with strong theoretical guarantees. Then, we take advantage of its properties and kernel methods to apply this discrepancy to brain-age prediction from MEG data, and compare it to state-of-the-art algorithms based on Riemannian geometry. Finally, we show that it is an efficient surrogate to the Wasserstein distance in domain adaptation for Brain Computer Interface applications.

## Citation

```
@inproceedings{bonet2023sliced,
  title={Sliced-Wasserstein on Symmetric Positive Definite Matrices for M/EEG Signals},
  author={Bonet, Clément and Malézieux, Benoît and Rakotomamonjy, Alain and Drumetz, Lucas and Moreau, Thomas and Kowalski, Matthieu and Courty, Nicolas},
  booktitle={International Conference on Machine Learning},
  pages={2777--2805},
  year={2023},
  organization={PMLR}
}
```

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
  To download the data, put the flag ```DOWNLOAD``` to ```True``` in ```da_particles.py``` or ```da_transfs.py```.
    
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
  The code is mainly based on https://github.com/meeg-ml-benchmarks/brain-age-benchmark-paper, on the class `SPDSW` in `spdsw/spdsw.py`, and on the class `KernelRidgeRegression` in `scikit-learn`.
