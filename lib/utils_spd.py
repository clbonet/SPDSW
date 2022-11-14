import torch

def busemann_spd(logM, diagA):
    """
        Inputs
        logM: log of SPD matrices
        diagA: Diagonal of eigenvalues of Symmetric matrice with norm 1
    """
    C = diagA[None] * logM[:,None]
    return -C.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
