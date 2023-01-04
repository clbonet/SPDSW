import geoopt
import ot
import torch
import sys
import argparse

import numpy as np

from geoopt import linalg
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import make_pipeline

sys.path.append("../lib/data")
from get_data import get_data, get_cov


device = "cuda" if torch.cuda.is_available() else "cpu"

def LEDA(Xs, Xt, loss="emd", reg=1):
    d = Xs.shape[-1]

    manifold_spd = geoopt.SymmetricPositiveDefinite("LEM")        

    a = torch.ones((len(Xs),), device=device, dtype=torch.float64)/len(Xs)
    b = torch.ones((len(Xt),), device=device, dtype=torch.float64)/len(Xt)
    M = manifold_spd.dist(Xs[:,None], Xt[None])**2
    if loss == "emd":
        P = ot.emd(a, b, M)
    elif loss == "sinkhorn":
        P = ot.sinkhorn(a, b, M, reg)
            
    log_Xt = geoopt.linalg.sym_logm(Xt)            
    cpt = torch.matmul(P, log_Xt[None].reshape(-1, d*d)).reshape(-1,d,d)
            
    return geoopt.linalg.sym_expm(cpt*len(Xt))



def cross_session(session, loss="emd", reg=1, d=22):
    Xs, ys = get_data(session, True, "../dataset/")
    cov_Xs = torch.tensor(get_cov(Xs), device=device) #, dtype=torch.float32)
    ys = torch.tensor(ys, device=device, dtype=torch.long)-1

    Xt, yt = get_data(session, False, "../dataset/")
    cov_Xt = torch.tensor(get_cov(Xt), device=device) #, dtype=torch.float32)
    yt = torch.tensor(yt, device=device, dtype=torch.long)-1


    Xs2 = LEDA(cov_Xs[:,0,0], cov_Xt[:,0,0], loss=loss, reg=reg)

    ## SVM on original data
    log_Xs = linalg.sym_logm(cov_Xs[:,0,0]).detach().cpu().reshape(-1, d*d)
    log_Xt = linalg.sym_logm(cov_Xt[:,0,0]).detach().cpu().reshape(-1, d*d)

    clf = make_pipeline(GridSearchCV(LinearSVC(), {"C": np.logspace(-2, 2, 10)}, n_jobs=10))
    clf.fit(log_Xs, ys.cpu())
    score0 = clf.score(log_Xt, yt.cpu())

    ## SVM on shifted data
    log_Xs = linalg.sym_logm(Xs2).detach().cpu().reshape(-1, d*d)
    log_Xt = linalg.sym_logm(cov_Xt[:,:,0]).detach().cpu().reshape(-1, d*d)

#     clf = make_pipeline(GridSearchCV(SVC(), {"C": np.logspace(-2, 2, 10)}, n_jobs=10))
    clf = make_pipeline(GridSearchCV(LinearSVC(), {"C": np.logspace(-2, 2, 10)}, n_jobs=10))
    clf.fit(log_Xs, ys.cpu())
    score1 = clf.score(log_Xt, yt.cpu())

    return score0, score1


def cross_subject(session1, session2, loss="emd", reg=1, d=22):
    Xs, ys = get_data(session1, True, "../dataset/")
    cov_Xs = torch.tensor(get_cov(Xs), device=device) #, dtype=torch.float32)
    ys = torch.tensor(ys, device=device, dtype=torch.long)-1

    Xt, yt = get_data(session2, True, "../dataset/")
    cov_Xt = torch.tensor(get_cov(Xt), device=device) #, dtype=torch.float32)
    yt = torch.tensor(yt, device=device, dtype=torch.long)-1


    Xs2 = LEDA(cov_Xs[:,0,0], cov_Xt[:,0,0], loss=loss, reg=reg)

    ## SVM on original data
    log_Xs = linalg.sym_logm(cov_Xs[:,0,0]).detach().cpu().reshape(-1, d*d)
    log_Xt = linalg.sym_logm(cov_Xt[:,0,0]).detach().cpu().reshape(-1, d*d)

    clf = make_pipeline(GridSearchCV(LinearSVC(), {"C": np.logspace(-2, 2, 10)}, n_jobs=10))
    clf.fit(log_Xs, ys.cpu())
    score0 = clf.score(log_Xt, yt.cpu())

    ## SVM on shifted data
    log_Xs = linalg.sym_logm(Xs2).detach().cpu().reshape(-1, d*d)
    log_Xt = linalg.sym_logm(cov_Xt[:,:,0]).detach().cpu().reshape(-1, d*d)

#     clf = make_pipeline(GridSearchCV(SVC(), {"C": np.logspace(-2, 2, 10)}, n_jobs=10))
    clf = make_pipeline(GridSearchCV(LinearSVC(), {"C": np.logspace(-2, 2, 10)}, n_jobs=10))
    clf.fit(log_Xs, ys.cpu())
    score1 = clf.score(log_Xt, yt.cpu())

    return score0, score1


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="session", help="session or subject")
    parser.add_argument("--loss", type=str, default="emd", help="Which loss to use: emd or sinkhorn")
    args = parser.parse_args()
    
    if args.task == "session":
        L = []
        for s in range(1,10):
            score_source, score_target = cross_session(s, loss=args.loss, reg=1)
            L.append(score_target)
        
        np.savetxt("./results_leotda_cross_"+args.task+"_loss_"+args.loss, L, delimiter=",")
        
    elif args.task == "subject":
        results_source = np.zeros((5,5))
        results_target = np.zeros((5,5))

        for i, s1 in enumerate([1,3,7,8,9]):
            for j,s2 in enumerate([1,3,7,8,9]):
                if s1 != s2:
                    result_source, result_target = cross_subject(s1, s2, loss=args.loss, reg=1)
                    results_source[i,j] = result_source
                    results_target[i,j] = result_target
                    
        np.savetxt("./results_leotda_cross_"+args.task+"_loss_"+args.loss, results_target, delimiter=",")
    