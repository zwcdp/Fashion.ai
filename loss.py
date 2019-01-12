import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


@torch.jit.script
def triplet_loss(sample_embedding, 
                 label_embedding, 
                 label, 
                 n_class:int=10, margin:float=5.0):
    """
    calculate
    max(0, margin + d(positive, anchor) - d(negative, anchor))
    """
    
    loss = torch.zeros((1), device=label.device)
    count = 0
    
    for y in range(n_class):
        if bool(((label==y).sum()) > 0):
            pos = sample_embedding[label == y, :]
            neg = sample_embedding[label != y, :]
            anc = label_embedding[label == y, :].mean(0, keepdim=True)
            d_pa = (((pos - anc)**2).sum(1)).mean(0)
            d_na = (((neg - anc)**2).sum(1)).mean(0)
            loss += F.softplus(margin + d_pa - d_na)
            count += 1
    
    return loss if count == 0 else loss/count

@@torch.jit.script
def gram_matrix(feat:torch.Tensor):
    batch, channel, _, _ = feat.size()
    vector = feat.view(batch, channel, -1)
    _, normalization = vector.view(batch,-1).size()
    gram_mat = torch.zeros((batch, channel, channel), 
        device=feat.device)
    for bdx in range(batch):
        v = vector[bdx, :, :]
        gram_mat[bdx, :, :] = torch.mm(v, v.t())
    return gram_mat#/normalization

# @torch.jit.script
# some weird jit shit
# it will not work if linear algebra is used
def euclid_dist(X):
    """X is in shape (n_sample, n_feature)"""
    n_sam, n_dim = X.size()
    dist = torch.zeros((n_sam, n_sam), device=X.device)
    for idx in range(n_sam):
        for jdx in range(idx+1, n_sam):
            dist[idx, jdx] = torch.sqrt(((X[idx,:] - X[jdx,:])**2).sum())
            dist[jdx, idx] = dist[idx, jdx]
    return dist

@torch.jit.script
def spectral_loss(feat:torch.Tensor, vec:torch.Tensor):
    batch, _, _, _ = feat.size()
    assert vec.size()[0] == batch
    
    gmat = gram_matrix(feat).view(batch, -1)
    
    # similarity between idx and jdx is cossim[idx, jdx]
    cossim = gmat@gmat.t()
    selfsim = torch.diag((cossim.diag())**(-1/2))
    cossim = selfsim@cossim@selfsim
    cossim = 0.5*(cossim + cossim.t())
    
    mse_vec = euclid_dist(vec)
    loss = 0.5*(cossim * (mse_vec**2)).mean()
            
    return loss