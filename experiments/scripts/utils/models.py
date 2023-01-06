import torch
import geoopt

import torch.nn as nn

from geoopt import ManifoldParameter, Stiefel
from functools import lru_cache, partial
from typing import Callable


class Translation(nn.Module):
    def __init__(self, d, n_freq, device):
        super().__init__()

        manifold_spdai = geoopt.SymmetricPositiveDefinite("AIM")        
        self._W = ManifoldParameter(
            torch.eye(
                d,
                dtype=torch.double,
                device=device
            )[None, :].repeat(n_freq, 1, 1),
            manifold=manifold_spdai
        )

        with torch.no_grad():
            self._W.proj_()

    def forward(self, X):
        return torch.matmul(self._W, torch.matmul(X, self._W.transpose(2, 1)))

class Rotation(nn.Module):
    def __init__(self, d, n_freq, device):
        super().__init__()

        manifold = Stiefel()        
        self._W = ManifoldParameter(
            torch.eye(
                d,
                dtype=torch.double,
                device=device)[None, :].repeat(n_freq, 1, 1),
            manifold=manifold
        )

        with torch.no_grad():
            self._W.proj_()

    def forward(self, X):
        return torch.matmul(self._W, torch.matmul(X, self._W.transpose(2, 1)))


class Transformations(nn.Module):
    def __init__(self, d, n_freq, device, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        self.translation = Translation(d, n_freq, device)
        self.rotation = Rotation(d, n_freq, device)

    def forward(self, X):
        Y = self.translation(X)
        Y = self.rotation(Y)
        return Y

# Taken from geoopt
# https://github.com/geoopt/geoopt/blob/master/geoopt/linalg/batch_linalg.py

@lru_cache(None)
def _sym_funcm_impl(func, **kwargs):
    func = partial(func, **kwargs)

    def _impl(x):
        e, v = torch.linalg.eigh(x, "U")
        return v @ torch.diag_embed(func(e)) @ v.transpose(-1, -2)

    return torch.jit.script(_impl)


def sym_funcm(
    x: torch.Tensor, func: Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    """Apply function to symmetric matrix.
    Parameters
    ----------
    x : torch.Tensor
        symmetric matrix
    func : Callable[[torch.Tensor], torch.Tensor]
        function to apply
    Returns
    -------
    torch.Tensor
        symmetric matrix with function applied to
    """
    return _sym_funcm_impl(func)(x)


def sym_reeig(x: torch.Tensor) -> torch.Tensor:
    r"""Symmetric matrix exponent.
    Parameters
    ----------
    x : torch.Tensor
        symmetric matrix
    Returns
    -------
    torch.Tensor
        :math:`\exp(x)`
    Notes
    -----
    Naive implementation of `torch.matrix_exp` seems to be fast enough
    """
    return sym_funcm(x, nn.Threshold(1e-4, 1e-4))
