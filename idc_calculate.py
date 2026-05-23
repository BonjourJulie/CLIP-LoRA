import torch
import numpy as np
import functools
import math

def estimate_id(X, algorithm='twoNN', k=100, fraction=0.9, full_output=False):
    if algorithm=='twoNN':
        return twoNN(X, fraction)

def twoNN(X,fraction=0.9,distances=False):
    if not distances:
        X=torch.cdist(X,X)
    Y=torch.topk(X, 3, dim=1, largest=False)[0]
    # clean data
    k1 = Y[:,1]
    k2 = Y[:,2]
    #remove zeros and degeneracies (k1==k2)
    old_k1=k1
    k1 = k1[old_k1!=0]
    k2 = k2[old_k1!=0]
    old_k1=k1
    k1 = k1[old_k1!=k2] 
    k2 = k2[old_k1!=k2]
    # n.of points to consider for the linear regression
    npoints = int(np.floor(len(k1)*fraction))
    # define mu and Femp
    N = len(k1)
    mu,_ = torch.sort(torch.divide(k2, k1).flatten())
    Femp = (torch.arange(1,N+1,dtype=X.dtype))/N
    # take logs (leave out the last element because 1-Femp is zero there)
    x = torch.log(mu[:-1])[0:npoints]
    y = -torch.log(1 - Femp[:-1])[0:npoints]
    # regression, on gpu if available
    y=y.to(x.device)
    slope=torch.linalg.lstsq(x.unsqueeze(-1),y.unsqueeze(-1))
    return slope.solution.squeeze()

def cat(l):
    return torch.cat(l, axis=1)
def shuffle(data):
    return data[torch.randperm(len(data))]
def standardize(data):
    return (data-data.mean(0))/(data.std(0)+1e-9)
def normalize(data):
    if len(data.shape)>1 and data.shape[1]>1:
        return torch.nn.functional.normalize(data)
    else:
        return data
def shuffle_keeping_class(data, labels):
    classes=torch.unique(labels)
    shuffled=torch.zeros_like(data)
    for c in classes:
        idx=torch.where(labels==c)[0]
        shuffled[idx]=data[idx][torch.randperm(len(idx))]
    return shuffled

def id_correlation(dataset1, dataset2, N=100, algorithm='twoNN', return_pvalue=True, k=100):
    dataset1=standardize(dataset1)
    dataset2=standardize(dataset2)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    id_1 = estimate_id(dataset1.to(device), algorithm, k).item()
    id_2 = estimate_id(dataset2.to(device), algorithm, k).item()
    upper_bound = id_1+id_2
    lower_bound = min(id_1, id_2)
    original_id = estimate_id(cat([dataset1, dataset2]).to(device), algorithm, k).item()
    corr= (upper_bound - original_id) / (upper_bound - lower_bound)
    if return_pvalue:
        shuffled_id=torch.zeros(N, dtype=torch.float)
        for i in range(N):
            shuffled_id[i]=estimate_id(cat([dataset1, shuffle(dataset2)]).to(device), algorithm, k).item()
        p=(((shuffled_id<original_id).sum()+1)/(N+1)).item() #according to permutation test
    else:
        p=None
    return {'corr': corr, 'p': p, 'id': original_id, 'id1': id_1, 'id2': id_2}