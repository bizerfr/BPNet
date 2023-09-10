import numpy as np
import torch
import torch.nn.functional as F

def embedding_loss(W_pred, pt_embed, match_indices, t_pull=0.0, t_push=2.0, eps=1e-8):
    """
    W_pred: B*Kmax*N
    pt_embed: B*dim*N
    match_indices: B*2*K,since the GT index of instance is inconsistent, use -1 to denote this
    """
    B, Kmax, N = W_pred.shape
    
    assert B == pt_embed.shape[0]
    assert N == pt_embed.shape[2]
    
    dim = pt_embed.shape[1]
    device = pt_embed.device
    
    ins_embed = torch.bmm(pt_embed, W_pred.permute(0, 2, 1)) # B*dim*Kmax
    ins_soft_num = torch.maximum(W_pred.sum(dim=2), torch.tensor(eps)).unsqueeze(1).expand(B, dim, Kmax)
    ins_embed = ins_embed / ins_soft_num  # B*dim*Kmax
    
    center = torch.bmm(ins_embed, W_pred) # B*dim*N
    
    dist = torch.linalg.vector_norm(pt_embed - center, ord=2, dim=1) 
    dist = dist - t_pull
    dist = F.relu(dist)
    loss_pull = torch.mean(dist)
    
    pair_wise_dist = torch.cdist(ins_embed.permute(0,2,1), ins_embed.permute(0,2,1), p=2) # B*Kmax*Kmax
    # select matched
    I_pred = match_indices[:,0,:] #B*K
    mask = np.zeros((B, Kmax, Kmax), dtype=bool)
    for b in range(B):
        I_pred_one_sample = I_pred[b, ...]
        valid = I_pred_one_sample[I_pred_one_sample!=-1]
        iv, jv = np.meshgrid(valid, valid, indexing='ij')
        mask[b, iv, jv] = True
        np.fill_diagonal(mask[b, ...], False)
    mask = np.triu(mask)
    
    valid_pair_dist = torch.masked_select(pair_wise_dist, torch.tensor(mask).to(device))
    valid_pair_dist = t_push - valid_pair_dist
    valid_pair_dist = F.relu(valid_pair_dist)
    if len(valid_pair_dist) == 0:
        loss_push = torch.tensor(0.0).to(device)
    else:
        loss_push = torch.mean(valid_pair_dist)
     
    return loss_pull, loss_push
    