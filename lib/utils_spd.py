import torch

def busemann_spd(logM, diagA):
    """
        Inputs
        logM: log of SPD matrices
        diagA: Diagonal of eigenvalues of Symmetric matrice with norm 1 (format: should be preprocessed by using 
            diagA = torch.diagonal(A, dim1=-2, dim2=-1)
            diagA = diagA.unsqueeze(-1)
            diagA = diagA.repeat(1,1,2)
        )
    """
    C = diagA[None] * logM[:,None]
    return -C.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
