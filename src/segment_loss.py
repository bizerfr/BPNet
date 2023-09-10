import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
from src.focal_loss import *

# ====================== point degree classification ===================
def pt_deg_cls_loss(pt_deg_logp, pt_deg_gt, clamp_min=0.0):
    """ 
    pt_deg_logp: B*C*N 
    pt_deg_gt: B*N
    """
    B = pt_deg_gt.shape[0]
    N = pt_deg_gt.shape[1]
    
    assert B == pt_deg_logp.shape[0]
    assert N == pt_deg_logp.shape[2]
    
    fl = focal_loss(pt_deg_logp, pt_deg_gt)

    fl = F.relu(fl - clamp_min)

    return fl

def deg_per_point_from_pt_probs(pt_deg_prob):
    """
    pt_deg_prob: B*C*N
    """
    pred_pt_deg_labels = torch.argmax(pt_deg_prob, dim=1)
    return pred_pt_deg_labels

def deg_per_point_from_ins(I_pred, I_deg_pred):
    """
    I_pred: B*N
    I_deg_pred: B*Kmax
    return: B*N
    """
    B = I_pred.shape[0]
    N = I_pred.shape[1]
    
    assert B == I_deg_pred.shape[0]
    
    batch_ind = torch.arange(B).unsqueeze(1).expand(B, N)
    pred_pt_deg_labels = I_deg_pred[batch_ind, I_pred]
    return pred_pt_deg_labels



# ================ primitive segmentation ================
def relaxed_iou_fast(pred, gt):
    """ 
    pred B*N*Kmax
    gt B*N*K
    returned cost is B*Kmax*K 
    """
    B, N, K = gt.shape
  
    assert B == pred.shape[0]
    assert N == pred.shape[1]

    norms_p = torch.unsqueeze(torch.sum(pred.abs(), 1), 2) # B*Kmax*1
    norms_g = torch.unsqueeze(torch.sum(gt.abs(), 1), 1)  # B*1*K
    
    cost = []
    for b in range(B):
        p = pred[b] # N*Kmax
        g = gt[b] # N*K
        c_batch = []
        dots = p.transpose(1, 0) @ g # Kmax*K
        r_iou = dots / (norms_p[b] + norms_g[b] - dots + 1e-8) 
        cost.append(r_iou)
    cost = torch.stack(cost, dim=0)
    
    return cost # B*Kmax*K

def hungarian_matching(cost):
    """ 
    cost: B*K1*K2
    return B*2*min(K1,K2)
    hungarian_matching computes the minimum, if we want to find best match,
    we should input -cost
    """
    device = cost.device
    B = cost.shape[0]
    K1 = cost.shape[1]
    K2 = cost.shape[2]
    K = K1 if K1 < K2 else K2
    match_indices = np.zeros([B, 2, K], dtype=np.int)
    for b in range(B):
        # return of linear_sum_assignment is sorted by rows
        # we transpose the matrix
        row_ind_, col_ind_ = linear_sum_assignment(cost[b].cpu().detach().numpy().T)  
        row_ind, col_ind = col_ind_, row_ind_
        match_indices[b, 0, :] = row_ind
        match_indices[b, 1, :] = col_ind
    return match_indices

def mean_relaxed_iou_loss(W_pred, I_gt):
    '''
    This function does not backprob gradient, only output matching indices
    input:
    W_pred - B*Kmax*N
    I_gt - B*N
    output:
    miou loss,
    match_indices - B*2*K,
    in each batch, row_ind represents the pred and col_ind represents the GT
    '''
    W_pred = W_pred.permute(0,2,1) # B*N*Kmax
    
    B = I_gt.shape[0]
    N = I_gt.shape[1]
    Kmax = W_pred.shape[2]
    
    assert B == W_pred.shape[0]
    assert N == W_pred.shape[1]
    
    # convert I_gt to one hot encoding
    # the int label of I_gt is inconsistent
    # it may contain all-zero column
    W_gt = F.one_hot(I_gt) #B*N*K, K is I_gt.max()+1
    W_gt = W_gt.float()
    K = W_gt.shape[2]
    
    # compute batch rIoU cost 
    cost = relaxed_iou_fast(W_pred, W_gt) # B*Kmax*K

    del W_gt
    
    assert B == cost.shape[0]
    assert Kmax == cost.shape[1]
    assert K == cost.shape[2]

    device = W_pred.device
    
    # hungarian_matching computes the minimum
    match_indices = hungarian_matching(-cost) # B*2*K

    matched_cost = torch.zeros(B, K).to(device)
    
    for b in range(B):
        # row_ind represents the pred
        row_ind = match_indices[b, 0, :]
        # col_ind should be 0,1,2,..., no skip, representing the GT
        col_ind = match_indices[b, 1, :] 
        matched_cost[b,:] = cost[b, row_ind, col_ind] # (K,)
        #assign unsed instance id to -1
        unique = I_gt[b,:].unique().cpu().detach().numpy()
        unique_num = unique.shape[0]
        uniques = np.repeat(unique, K).reshape(unique_num, K)
        unused = np.all(uniques != col_ind, axis=0)
        row_ind[unused] = -1 
        col_ind[unused] = -1
        
    del cost
        
    # remove zeros in the cost
    # this happens because the patch index in GT is not consistent, thus resulting 
    # empty vectors after one-hot encoding
    #matched_cost = matched_cost[matched_cost.nonzero(as_tuple=True)]
    matched_cost = matched_cost[matched_cost != 0]
    matched_cost = 1.0 - matched_cost
    loss_mean_riou =  torch.mean(matched_cost) 

    return loss_mean_riou, match_indices

def ins_per_point(W_pred):
    """
    W_pred: B*Kmax*N
    return: B*N
    """
    pred_labels = torch.argmax(W_pred, dim=1)
    return pred_labels


def reorder_ins_probs(W_pred, I_deg_prob, ctrlpts, I_gt):
    """ 
    W_pred- B*Kmax*N
    I_deg_prob - B*C*Kmax  
    ctrlpts - B*Kmax*4*4*4; (x,y,z,w) or (wx, wy, wz, w)
    I_gt - B*N
    reorder W_pred to match I_gt (B*Kmax*N)
    """
    W_pred = W_pred.permute(0,2,1) # B*N*Kmax

    B = I_gt.shape[0]
    N = I_gt.shape[1]
    
    assert B == W_pred.shape[0]
    assert B == I_deg_prob.shape[0]
    assert N == W_pred.shape[1]

    Kmax = W_pred.shape[2]
    assert Kmax == I_deg_prob.shape[2]
    
    # convert I_gt to one hot encoding
    # the int label of I_gt is inconsistent
    # it may contain all-zero column
    W_gt = F.one_hot(I_gt, num_classes=Kmax) #B*N*K_gt
    W_gt = W_gt.float()
    
    # compute batch rIoU cost 
    cost = relaxed_iou_fast(W_pred, W_gt) # B*Kmax_*Kmax_

    del W_gt
    
    assert B == cost.shape[0]
    assert cost.shape[1] == cost.shape[2]
    
    W_pred_reorder = W_pred.clone().detach()
    I_deg_prob_reorder = I_deg_prob.clone().detach()
    ctrlpts_reorder = ctrlpts.clone().detach()
    # hungarian_matching computes the minimum
    match_indices = hungarian_matching(-cost)

    del cost

    for b in range(B):
        # row_ind represents the pred
        row_ind = match_indices[b, 0, :]
        # col_ind should be 0,1,2,..., no skip, representing the GT
        col_ind = match_indices[b, 1, :] 
        
        assert len(row_ind) == len(col_ind)
        
        W_pred_reorder[b, :, col_ind] = W_pred[b, :, row_ind]
        I_deg_prob_reorder[b, :, col_ind] = I_deg_prob[b, :, row_ind]
        ctrlpts_reorder[b, col_ind, ...] = ctrlpts[b, row_ind, ...]
        
    W_pred_reorder = W_pred_reorder.permute(0,2,1) # B*Kmax*N
  
    return W_pred_reorder, I_deg_prob_reorder, ctrlpts_reorder

def reorder_ins_labels(I_pred, I_deg, ctrlpts, I_gt):
    """ 
    I_pred- B*N
    I_deg - B*Kmax
    I_gt - B*N
    ctrlpts - B*Kmax*4*4*4; (x,y,z,w) or (wx, wy, wz, w)
    reorder I_pred to match I_gt
    """
    
    B = I_gt.shape[0]
    N = I_gt.shape[1]
    
    assert B == I_pred.shape[0]
    assert N == I_pred.shape[1]
    
    Kmax = I_deg.shape[1]
    Kmax_ = max(I_pred.max(), I_gt.max()) + 1
    assert Kmax_ <= Kmax
    
    # convert I_pred to one hot encoding
    # the int label of I_pred is inconsistent
    # it may contain all-zero column
    # num_classes must be Kmax_
    W_pred_one_hot_ = F.one_hot(I_pred, num_classes=Kmax_) # B*N*Kmax_
    W_pred_one_hot_ = W_pred_one_hot_.float()
    
    # convert I_gt to one hot encoding
    # the int label of I_gt is inconsistent
    # it may contain all-zero column
    # num_classes must be Kmax_
    W_gt = F.one_hot(I_gt, num_classes=Kmax_) #B*N*K_gt
    W_gt = W_gt.float()
    
    # compute batch rIoU cost 
    cost = relaxed_iou_fast(W_pred_one_hot_, W_gt) # B*Kmax_*Kmax_

    del W_pred_one_hot_, W_gt
    
    assert B == cost.shape[0]
    assert cost.shape[1] == cost.shape[2]
    
    I_pred_reorder = I_pred.clone().detach()
    I_deg_reorder = I_deg.clone().detach()
    ctrlpts_reorder = ctrlpts.clone().detach()
    # hungarian_matching computes the minimum
    match_indices = hungarian_matching(-cost) #B*2*Kmax_

    del cost

    for b in range(B):
        # row_ind represents the pred
        row_ind = match_indices[b, 0, :]
        # col_ind should be 0,1,2,..., no skip, representing the GT
        col_ind = match_indices[b, 1, :] 
        
        assert len(row_ind) == len(col_ind)
                
        for i in range(len(row_ind)):
            select = I_pred[b, :] == row_ind[i]
            I_pred_reorder[b, select] = col_ind[i]
            
        I_deg_reorder[b, col_ind] = I_deg[b, row_ind]
        ctrlpts_reorder[b, col_ind, ...] = ctrlpts[b, row_ind, ...]
            
    return I_pred_reorder, I_deg_reorder, ctrlpts_reorder 

    
# ====================== instance degree regression ===================
def soft_voting_loss(I_deg_score, I_deg_gt, match_indices, clamp_min=0.0, eps=1e-8):
    """ 
    I_deg_score: B*C*Kmax, soft point number for each degree
    I_deg_gt: B*Kmax, instance deg label, -1 denotes unused index
    match_indices: B*2*K; since the GT index of instance is inconsistent, use -1 to denotes this
    """
    
    B, C, Kmax = I_deg_score.shape
    
    assert B == I_deg_gt.shape[0]
    assert B == match_indices.shape[0]
    assert Kmax == I_deg_gt.shape[1]
    
    K = match_indices.shape[2]
    assert Kmax >= K
    
    I_pred = match_indices[:,0,:]
    I_gt = match_indices[:,1,:]
    
    select_match = I_pred != -1 #(B, K)
    valid_num = np.count_nonzero(select_match, axis=1) #(B, K)
    batch_ind = np.repeat(np.arange(B), valid_num)
    match_I_deg_score = I_deg_score[batch_ind, :, I_pred[select_match]]
    # only normalize the matched 
    match_I_deg_logp = F.normalize(match_I_deg_score, p=1.0, dim=1)
    match_I_deg_logp = torch.maximum(match_I_deg_logp, torch.tensor(eps))
    match_I_deg_logp = torch.log(match_I_deg_logp)
    
    select_match = I_gt != -1 #(B, K)
    valid_num = np.count_nonzero(select_match, axis=1) #(B, K)
    batch_ind = np.repeat(np.arange(B), valid_num)
    match_I_deg_gt = I_deg_gt[batch_ind, I_gt[select_match]]
    
    assert torch.all(match_I_deg_gt!=-1)
    
    voting_loss = focal_loss(match_I_deg_logp, match_I_deg_gt)
    
    voting_loss = F.relu(voting_loss - clamp_min)
    
    return voting_loss

def deg_per_instance(I_deg_score):
    """
    I_deg_score: B*C*Kmax
    """
    pred_I_deg_labels = torch.argmax(I_deg_score, dim=1)
    return pred_I_deg_labels

def eval_primitive_type_acc(pred_labels, gt_labels, match_indices, C=9):
    """ 
    pred_labels (np array)  B*Kmax
    gt_labels (np array) B*Kmax, -1 denotes unused index
    match_indices: B*2*K; since the GT index of instance is inconsistent, use -1 to denotes this
    """
    assert isinstance(pred_labels, np.ndarray) 
    assert isinstance(gt_labels, np.ndarray)
    assert isinstance(match_indices, np.ndarray)
    
    B = gt_labels.shape[0]  # batch number
    Kmax = gt_labels.shape[1]
    
    assert B == pred_labels.shape[0]
    assert Kmax == pred_labels.shape[1]
    assert B == match_indices.shape[0]
    
    K = match_indices.shape[2]
    assert Kmax >= K 
    
    I_pred = match_indices[:,0,:]
    I_gt = match_indices[:,1,:]
    
    select_match = I_pred != -1 #(B, K)
    valid_num = np.count_nonzero(select_match, axis=1) #(B, K)
    batch_ind = np.repeat(np.arange(B), valid_num)
    match_pred_labels = pred_labels[batch_ind, I_pred[select_match]]
    
    select_match = I_gt != -1 #(B, K)
    valid_num = np.count_nonzero(select_match, axis=1) #(B, K)
    batch_ind = np.repeat(np.arange(B), valid_num)
    match_gt_labels = gt_labels[batch_ind, I_gt[select_match]]
    
    match_num = match_gt_labels.shape[0]
    
    assert match_num == match_pred_labels.shape[0]
    
    iou_part = 0.0
    for label_idx in range(C):
        locations_gt = (match_gt_labels == label_idx)
        locations_pred = (match_pred_labels == label_idx)
        I_locations = np.logical_and(locations_gt, locations_pred)
        U_locations = np.logical_or(locations_gt, locations_pred)
        I = np.sum(I_locations) + np.finfo(np.float32).eps
        U = np.sum(U_locations) + np.finfo(np.float32).eps
        iou_part = iou_part + I / U
              
    return iou_part / C

def eval_cluster_rand_score_one_sample(I_gt, I_pred):
    """ 
    I_gt - N*3
    I_pred - N
    """
    N = I_gt.shape[0]
    assert N == I_pred.shape[0]

    score = metrics.rand_score(I_gt, I_pred)
    return score

def eval_cluster_rand_score(I_gt, I_pred):
    """ 
    I_gt - B*N*3
    I_pred - B*N
    """
    B = I_gt.shape[0]
    N = I_gt.shape[1]
    
    assert B == I_pred.shape[0]
    assert N == I_pred.shape[1]
    
    scores = []
    for b in range(B):
        I_gt_ = I_gt[b, :]
        I_pred_ = I_pred[b, :]
        score = eval_cluster_rand_score_one_sample(I_gt_, I_pred_)
        scores.append(score)

    score = np.mean(scores)
    return score

def get_number_of_primitives(I_pred):
    """ 
    I_gt - B*N*3
    I_pred - B*N
    """
    B = I_pred.shape[0]
    
    num_list = []
    for b in range(B):
        I_pred_ = I_pred[b, :]
        unique_I_ = torch.unique(I_pred_)
        pred_ins_num = unique_I_.shape[0]
        num_list.append(pred_ins_num)
        
    avg_num = np.mean(num_list)
    return avg_num
    

    

    
