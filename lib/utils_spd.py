import torch


def dist_spd(A,B):
    A_ = torch.linalg.inv(A)
    C = torch.matmul(A_[:,None], B[None])**2
    tr = C.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
    return tr**(1/2)

def busemann_spd(logM, diagA):
    """
        Inputs
        logM: log of SPD matrices
        diagA: Diagonal of eigenvalues of Symmetric matrice with norm 1
    """
    C = diagA[None] * logM[:,None]
    return -C.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
