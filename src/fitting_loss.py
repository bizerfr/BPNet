import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import comb
from src.segment_utils import *
from src.segment_loss import *

def homogeneous_coordiantes(ctrlpts):
    """
    input:
        ctrlpts(x,y,z,w): B*Kmax*max_num_ctrlpts_u*max_num_ctrlpts_v*4
    output:
        (wx,wy,wz,w)
    """
    B, Kmax, max_num_ctrlpts_u, max_num_ctrlpts_v, _ = ctrlpts.shape
    
    ctrlpts = ctrlpts.reshape((B, Kmax, max_num_ctrlpts_u*max_num_ctrlpts_v, 4))
    coords = ctrlpts[:, :, :, 0:3] #B*Kmax*(max_num_ctrlpts_u*max_num_ctrlpts_v)*3
    weight = ctrlpts[:, :, :, 3] #B*Kmax*(max_num_ctrlpts_u*max_num_ctrlpts_v)
  
    # homogeneous coordiantes (wx, wy, wz, w)
    weighted_coords = coords * weight.unsqueeze(dim=3).expand(-1,-1,-1,3)
    homo_ctrlpts = torch.cat((weighted_coords, weight.unsqueeze(dim=3)), dim=3)
    homo_ctrlpts = homo_ctrlpts.reshape((B, Kmax,
                                         max_num_ctrlpts_u,
                                         max_num_ctrlpts_v,
                                         4))
    return homo_ctrlpts


def regularize_ctrlpts_weight(ctrlpts, I_deg_uv):
    """
    regularize the weight of control points according to the predicted valid instances and force invlide ctrlpts to be 0
    ctrlpts: B*Kmax*max_num_ctrlpts_u*max_num_ctrlpts_u*4; (x,y,z,w) 
    I_deg_uv: B*Kmax*2, instance degree
    """
    B, Kmax, max_num_ctrlpts_u, max_num_ctrlpts_v, _ = ctrlpts.shape
    
    device = ctrlpts.device
    
    assert B == I_deg_uv.shape[0]
    assert Kmax == I_deg_uv.shape[1]
    assert 2 == I_deg_uv.shape[2]
    
    max_num_ctrlpts = max_num_ctrlpts_u * max_num_ctrlpts_v
    
    I_deg_uv = I_deg_uv.cpu().detach().numpy()
    num_ctrlpts_u = I_deg_uv[:, :, 0] + 1 #B*Kmax
    num_ctrlpts_u = np.repeat(np.expand_dims(num_ctrlpts_u, axis=2), max_num_ctrlpts_u, axis=2)
    num_ctrlpts_v = I_deg_uv[:, :, 1] + 1 #B*Kmax
    num_ctrlpts_v = np.repeat(np.expand_dims(num_ctrlpts_v, axis=2), max_num_ctrlpts_v, axis=2)
    mask_u = np.tile(np.arange(max_num_ctrlpts_u)+1, (B, Kmax)).reshape(B, Kmax, max_num_ctrlpts_u) #B*Kmax*max_num_ctrlpts_u
    mask_u = mask_u <= num_ctrlpts_u 
    mask_u = mask_u.astype(np.int) #B*Kmax*max_num_ctrlpts_u
    mask_v = np.tile(np.arange(max_num_ctrlpts_v)+1, (B, Kmax)).reshape(B, Kmax, max_num_ctrlpts_v) #B*Kmax*max_num_ctrlpts_v
    mask_v = mask_v <= num_ctrlpts_v
    mask_v = mask_v.astype(np.int) #B*Kmax*max_num_ctrlpts_v
    
    mask_u = torch.tensor(mask_u).to(device)
    mask_v = torch.tensor(mask_v).to(device)
    ctrlpts = torch.einsum('bkuvc,bku,bkv->bkuvc', ctrlpts, mask_u, mask_v) 
    
    # regularize weight
    ctrlpts = ctrlpts.reshape((B, Kmax, max_num_ctrlpts_u*max_num_ctrlpts_v, 4))
    weight = ctrlpts[:,:,:,3] #B*Kmax*(max_num_ctrlpts_u*max_num_ctrlpts_v)
    weight = F.normalize(weight, p=1.0, dim=2)
    
    regularized_weight_ctrlpts = torch.cat((ctrlpts[:,:,:,0:3], weight.unsqueeze(dim=3)), dim=3)
        
    regularized_weight_ctrlpts = regularized_weight_ctrlpts.reshape((B,Kmax,
                                                            max_num_ctrlpts_u,
                                                            max_num_ctrlpts_v,
                                                            4))
            
    return regularized_weight_ctrlpts

def reorder_ctrlpts_pred(ctrlpts_pred, match_indices):
    """
    ctrlpts_pred: B*Kmax*max_num_ctrlpts_u*max_num_ctrlpts_u*4; (x,y,z,w) or (wx, wy, wz, w)
    match_indices: B*2*K; since the GT index of instance is inconsistent, use -1 to denote this
    this function will let the unmatched to be zero
    """
    B, Kmax, max_num_ctrlpts_u, max_num_ctrlpts_v, _= ctrlpts_pred.shape
    device = ctrlpts_pred.device
    
    ctrlpts_pred_reorder = torch.zeros((ctrlpts_pred.shape)).to(device)
    
    I_gt = match_indices[:,1,:] #B*K
    I_pred = match_indices[:,0,:] #B*K
    
    select_valid_I_gt = I_gt != -1 #B*K
    valid_num_gt = np.count_nonzero(select_valid_I_gt, axis=1) #B*K
    
    select_valid_I_pred = I_pred != -1 #B*K
    valid_num_pred = np.count_nonzero(select_valid_I_pred, axis=1) #B*K
    
    batch_ind = np.repeat(np.arange(B), valid_num_pred)
    assert np.all(valid_num_gt == valid_num_pred)
    
    ctrlpts_pred_reorder[batch_ind, I_gt[select_valid_I_gt], :, : ,:] =  \
            ctrlpts_pred[batch_ind, I_pred[select_valid_I_pred], :, : ,:]
              
    return  ctrlpts_pred_reorder      
      
def paras_loss(uv_pred, uv_gt):
    B, N, _ = uv_gt.shape
    
    assert B == uv_pred.shape[0]
    assert N == uv_pred.shape[1]
    
    loss_paras = F.mse_loss(uv_pred, uv_gt)
    return loss_paras

def ctrlpts_loss(ctrlpts_pred, I_deg_uv_pred, ctrlpts_gt, I_deg_uv_gt, match_indices, decode_degree_dict):
    """
    ctrlpts_pred: B*Kmax*max_num_ctrlpts_u*max_num_ctrlpts_u*4; (x,y,z,w) or (wx, wy, wz, w)
    I_deg_uv_pred: B*Kmax*2
    ctrlpts_gt: B*Kmax*max_num_ctrlpts_u*max_num_ctrlpts_u*4; (x,y,z,w) or (wx, wy, wz, w)
    I_deg_uv_gt: B*Kmax*2
    match_indices: B*2*K; since the GT index of instance is inconsistent, use -1 to denote this
    decode_degree_dict: decode int degree labels to degrees
    """
    B, Kmax, max_num_ctrlpts_u, max_num_ctrlpts_v, _= ctrlpts_gt.shape
    
    assert B == ctrlpts_pred.shape[0]
    assert Kmax == ctrlpts_pred.shape[1]
    assert B == I_deg_uv_pred.shape[0]
    assert Kmax == I_deg_uv_pred.shape[1]
    assert 2 == I_deg_uv_pred.shape[2]
    assert B == I_deg_uv_gt.shape[0]
    assert Kmax == I_deg_uv_gt.shape[1]
    assert 2 == I_deg_uv_gt.shape[2]
    assert B == match_indices.shape[0]
    
    K = match_indices.shape[2]
    assert Kmax >= K
    
    device = ctrlpts_pred.device
    
    I_gt = match_indices[:,1,:] #B*K
    I_pred = match_indices[:,0,:] #B*K
    
    select_valid_I = I_gt != -1 #B*K
    valid_num = np.count_nonzero(select_valid_I, axis=1) #B*K
    batch_ind = np.repeat(np.arange(B), valid_num)
    match_ctrlpts_gt = ctrlpts_gt[batch_ind, I_gt[select_valid_I], :, : ,:]
    match_I_deg_uv_gt = I_deg_uv_gt[batch_ind, I_gt[select_valid_I], :]
    
    select_valid_I = I_pred != -1 #B*K
    valid_num = np.count_nonzero(select_valid_I, axis=1) #B*K
    batch_ind = np.repeat(np.arange(B), valid_num)
    match_ctrlpts_pred = ctrlpts_pred[batch_ind, I_pred[select_valid_I], :, : ,:]
    match_I_deg_uv_pred = I_deg_uv_pred[batch_ind, I_pred[select_valid_I], :]
    match_ins_num = match_I_deg_uv_pred.shape[0]
    
    assert match_ins_num == match_ctrlpts_gt.shape[0]
    assert match_ins_num == match_ctrlpts_pred.shape[0]
    
    match_I_deg_uv_align = np.maximum(match_I_deg_uv_pred.cpu().detach().numpy(),
                                match_I_deg_uv_gt.cpu().detach().numpy()) #match_ins_num*2
    
    match_I_deg_align = encode_degrees_to_labels(match_I_deg_uv_align)
    
    loss_ctrlpts = torch.tensor(0.0).to(device)
    cnt = 0
    for deg_label, (deg_u, deg_v) in decode_degree_dict.items():
        num_ctrl_u, num_ctrl_v = deg_u + 1, deg_v + 1
        selected = deg_label == match_I_deg_align
        if np.all(~selected):
            continue
        truncted_pred = match_ctrlpts_pred[selected, 0 : num_ctrl_u, 0 : num_ctrl_v, :]
        truncted_gt = match_ctrlpts_gt[selected, 0 : num_ctrl_u, 0 : num_ctrl_v, :]
        loss_ctrlpts = loss_ctrlpts + F.mse_loss(truncted_pred, truncted_gt)
        cnt = cnt + 1
    
    if cnt == 0:
        return loss_ctrlpts
    
    loss_ctrlpts = loss_ctrlpts / cnt
    
    return loss_ctrlpts


def bernstein_basis_patch(deg, i, t):
    """
    deg: degree of Bernstein basis polynomials
    (deg, i) is the binomial coefficient
    t: B*N parameters
    """
    assert i <= deg
    binomial = comb(deg, i)
    binomial = torch.tensor(binomial)
    basis = binomial * torch.pow(t, i) * torch.pow(1 - t, deg - i)
    return basis

def reconstruct_coords_patch(deg_u, deg_v, ctrlpts_patch, uv_patch, eps=1e-12):
    """
    ctrl_pts_patch: max_num_ctrlpts_u*max_num_ctrlpts_v*4; (x,y,z,w)
    uv_patch: pt_num_in_patch*2
    """
    device = ctrlpts_patch.device
    
    pt_num_in_patch = uv_patch.shape[0]
    
    numerator = torch.zeros((pt_num_in_patch, 3)).to(device)
    denominator = torch.zeros((pt_num_in_patch, 3)).to(device)
    for i in range(deg_u + 1):
        for j in range(deg_v + 1):
            basis = bernstein_basis_patch(deg_u, i, uv_patch[:, 0]) * bernstein_basis_patch(deg_v, j, uv_patch[:, 1]) 
            basis = basis.unsqueeze(1).expand(-1,3)
            ctrlpts_coords = ctrlpts_patch[i, j, 0:3] 
            weight = ctrlpts_patch[i, j, 3] 
            numerator += weight * basis * ctrlpts_coords
            denominator += weight * basis
            
    denominator = torch.maximum(denominator, torch.tensor(eps))
    recon_coords = numerator / denominator
    return recon_coords

def bernstein_basis(max_deg_t, deg_t, t):
    """
    max_deg: max_degree of Bernstein basis polynomials
    deg_t: B*N
    t: B*N parameters
    return: bernstein_basis B*N*max_deg_t, where 0 is for invalid
    """
    B, N = t.shape
    
    dtype = t.dtype
    device = t.device
    
    assert B == deg_t.shape[0]
    assert N == deg_t.shape[1]
    
    max_num_ctrlpts_t = max_deg_t + 1
    
    deg_t = deg_t.cpu().detach().numpy() # B*N
    deg_t = np.repeat(np.expand_dims(deg_t, axis=2), max_num_ctrlpts_t, axis=2)
    index = np.tile(np.arange(max_num_ctrlpts_t), (B, N)).reshape(B, N, max_num_ctrlpts_t) #B*N*max_num_ctrlpts_t
    binomial = comb(deg_t, index) #B*N*max_num_ctrlpts_t, invalid binomial is 0
    selected = index <= deg_t
    index[~selected] = 0
    deg_t[~selected] = 0
    
    binomial = torch.tensor(binomial, dtype=dtype).to(device)
    index = torch.tensor(index).to(device)
    deg_t = torch.tensor(deg_t).to(device)
    t = t.unsqueeze(2).expand(B, N, max_num_ctrlpts_t)
    basis = binomial * torch.pow(t, index) * torch.pow(1 - t, deg_t - index)
    return basis

def reconstruct_coordinates(ctrlpts, uv, W, pt_deg_uv, eps=1e-12):
    """
    ctrl_pts: B*Kmax*max_num_ctrlpts_u*max_num_ctrlpts_v*4; (x,y,z,w)
    uv: B*N*2
    W: B*N*Kmax: instance segmentation probabilites
    pt_deg_uv: B*N*2
    return value: coordinates B*N*3
    """
    
    B, Kmax, max_num_ctrlpts_u, max_num_ctrlpts_v, _  = ctrlpts.shape
    N = uv.shape[1]
    
    max_deg_u, max_deg_v = max_num_ctrlpts_u - 1, max_num_ctrlpts_v - 1
    
    assert B == uv.shape[0]
    assert B == W.shape[0]
    assert N == W.shape[1]
    assert Kmax == W.shape[2]
    assert B == pt_deg_uv.shape[0]
    assert N == pt_deg_uv.shape[1]
            
    basis_u = bernstein_basis(max_deg_u, pt_deg_uv[:, :, 0], uv[:, :, 0]) #B*N*max_num_ctrlpts_u
    basis_v = bernstein_basis(max_deg_v, pt_deg_uv[:, :, 1], uv[:, :, 1]) #B*N*max_num_ctrlpts_v
    ctrlpts_coords = ctrlpts[:, :, :, :, 0:3] #B*Kmax*max_num_ctrlpts_u*max_num_ctrlpts_v*3
    ctrlpts_weight = ctrlpts[:, :, :, :, 3] #B*Kmax*max_num_ctrlpts_u*max_num_ctrlpts_v
    
    numerator = torch.einsum('biu,biv,bkuvc,bkuv,bik->bic', basis_u, basis_v, ctrlpts_coords, ctrlpts_weight, W) #B*N*3
    denominator = torch.einsum('biu,biv,bkuv,bik->bi', basis_u, basis_v, ctrlpts_weight, W) #B*N
    denominator = denominator.unsqueeze(2).expand(B,N,3)      
    denominator = torch.maximum(denominator, torch.tensor(eps))
    recon_coords = numerator / denominator #B*N*3
    
    return recon_coords

def reconstruct_coordinates_gt(ctrlpts_pred, uv, W_gt, pt_deg_uv_gt, match_indices, eps=1e-12):
    """
    ctrl_pts_pred: B*Kmax*max_num_ctrlpts_u*max_num_ctrlpts_v*4; (x,y,z,w)
    uv: B*N*2
    I_gt: B*N: instance segmentation ids
    pt_deg_uv_gt: B*N*2
    match_indices: B*2*K; since the GT index of instance is inconsistent, use -1 to denote this
    return value: coordinates B*N*3
    """
    
    B, Kmax, max_num_ctrlpts_u, max_num_ctrlpts_v, _  = ctrlpts_pred.shape
    N = uv.shape[1]
    K = match_indices.shape[2]
    
    max_deg_u, max_deg_v = max_num_ctrlpts_u - 1, max_num_ctrlpts_v - 1
    
    assert B == uv.shape[0]
    assert B == W_gt.shape[0]
    assert N == W_gt.shape[1]
    assert Kmax == W_gt.shape[2]
    assert B == pt_deg_uv_gt.shape[0]
    assert N == pt_deg_uv_gt.shape[1]
    assert B == match_indices.shape[0]
    assert Kmax >= K
    
    ctrlpts_pred_reorder = reorder_ctrlpts_pred(ctrlpts_pred, match_indices)
            
    basis_u = bernstein_basis(max_deg_u, pt_deg_uv_gt[:, :, 0], uv[:, :, 0]) #B*N*max_num_ctrlpts_u
    basis_v = bernstein_basis(max_deg_v, pt_deg_uv_gt[:, :, 1], uv[:, :, 1]) #B*N*max_num_ctrlpts_v
    ctrlpts_coords = ctrlpts_pred_reorder[:, :, :, :, 0:3] #B*Kmax*max_num_ctrlpts_u*max_num_ctrlpts_v*3
    ctrlpts_weight = ctrlpts_pred_reorder[:, :, :, :, 3] #B*Kmax*max_num_ctrlpts_u*max_num_ctrlpts_v
    
    numerator = torch.einsum('biu,biv,bkuvc,bkuv,bik->bic', basis_u, basis_v, ctrlpts_coords, ctrlpts_weight, W_gt) #B*N*3
    denominator = torch.einsum('biu,biv,bkuv,bik->bi', basis_u, basis_v, ctrlpts_weight, W_gt) #B*N
    denominator = denominator.unsqueeze(2).expand(B,N,3)      
    denominator = torch.maximum(denominator, torch.tensor(eps))
    recon_coords = numerator / denominator #B*N*3
    
    return recon_coords

@torch.enable_grad()
def reconstruct_normals(uv, recon_coords):
    """
    uv: B*N*2, must be the input for reconstruct_coordinates
    recon_coords: B*N*3; (x,y,z), must be the output from reconstruct_coords_from_deg
    """
    B, N, _ = uv.shape
    assert B == recon_coords.shape[0]
    assert N == recon_coords.shape[1]

    x = recon_coords[:, :, 0]
    y = recon_coords[:, :, 1]
    z = recon_coords[:, :, 2]
    
    x_gugv = torch.autograd.grad(outputs=x, inputs=uv, grad_outputs=torch.ones_like(x),
                                 retain_graph=True, create_graph=True)
    y_gugv = torch.autograd.grad(outputs=y, inputs=uv, grad_outputs=torch.ones_like(y),
                                 retain_graph=True, create_graph=True)
    z_gugv = torch.autograd.grad(outputs=z, inputs=uv, grad_outputs=torch.ones_like(z),
                                 retain_graph=True, create_graph=True)
        
    x_gu = x_gugv[0][:, :, 0].unsqueeze(dim=2) # B*N*1
    x_gv = x_gugv[0][:, :, 1].unsqueeze(dim=2) # B*N*1
    y_gu = y_gugv[0][:, :, 0].unsqueeze(dim=2) # B*N*1
    y_gv = y_gugv[0][:, :, 1].unsqueeze(dim=2) # B*N*1
    z_gu = z_gugv[0][:, :, 0].unsqueeze(dim=2) # B*N*1
    z_gv = z_gugv[0][:, :, 1].unsqueeze(dim=2) # B*N*1
            
    gu = torch.cat((x_gu, y_gu, z_gu), dim=2)
    gv = torch.cat((x_gv, y_gv, z_gv), dim=2)
     
    normals = torch.cross(gu, gv, dim=2) #B*N*3
    
    normals = F.normalize(normals, p=2.0, dim=2)
            
    return normals

@torch.enable_grad()
def reconstruct_coordinates_normals(ctrlpts, uv, W, pt_deg_uv, eps=1e-12):
    """
    ctrl_pts: B*Kmax*max_num_ctrlpts_u*max_num_ctrlpts_v*4; (x,y,z,w)
    uv: B*N*2
    W: B*N*Kmax: instance segmentation ids
    pt_deg_uv: B*N*2
    return value: coordinates B*N*3
    """
    
    ctrlpts.requires_grad_()
    uv.requires_grad_()
    
    #reconstruct_coords
    recon_coords = reconstruct_coordinates(ctrlpts, uv, W, pt_deg_uv, eps)
    recon_normals = reconstruct_normals(uv, recon_coords)
        
    return recon_coords, recon_normals  

@torch.enable_grad()
def reconstruct_coordinates_normals_gt(ctrlpts_pred, uv, W_gt, pt_deg_uv_gt, match_indices, eps=1e-12):
    """
    ctrl_pts: B*Kmax*max_num_ctrlpts_u*max_num_ctrlpts_v*4; (x,y,z,w)
    uv: B*N*2
    W_gt: B*N: instance segmentation ids
    pt_deg_uv: B*N*2
    return value: coordinates B*N*3
    """
    
    ctrlpts_pred.requires_grad_()
    uv.requires_grad_()
    
    #reconstruct_coords
    recon_coords = reconstruct_coordinates_gt(ctrlpts_pred, uv, W_gt, pt_deg_uv_gt, match_indices, eps)
    recon_normals = reconstruct_normals(uv, recon_coords)
        
    return recon_coords, recon_normals  



def coords_loss(recon_coords, sample_coords_gt):
    """
    recon_coords: B*N*3
    sample_coords_gt:B*N*3
    """
    
    B, N, dim = sample_coords_gt.shape
    assert B == recon_coords.shape[0]
    assert N == recon_coords.shape[1]
    assert dim == recon_coords.shape[2]

            
    loss_coords = F.mse_loss(sample_coords_gt, recon_coords)
    
    return loss_coords

def normals_loss(normal_pred, normal_gt):
    """
    normal_pred: B*N*3
    normal_gt:B*N*3
    """
    
    B, N, dim = normal_gt.shape
    assert B == normal_pred.shape[0]
    assert N == normal_pred.shape[1]
    assert dim == normal_pred.shape[2]

            
    product = torch.mul(normal_pred, normal_gt)
    inner_product = product.sum(dim=2)

    # filter (0.0, 0.0, 0.0) in normal_gt
    invalid_gt = normal_gt.abs().sum(dim=2) == 0
    inner_product = inner_product[~invalid_gt]
    
    assert inner_product.shape[0] > 0
    
    loss_normal = torch.mean(1 - inner_product.abs())
    
    return loss_normal


def eval_pt_normal_angle_diff(normal_pred, normal_gt, eps=1e-12):
    """ 
    pred_normals (np array)  B*N*3
    gt_normals (np array) B*N*3
    """
    assert isinstance(normal_pred, np.ndarray) 
    assert isinstance(normal_gt, np.ndarray)
    
    B = normal_gt.shape[0]  # batch number
    N = normal_gt.shape[1]
    
    assert B == normal_pred.shape[0]
    assert N == normal_pred.shape[1]
    
    assert normal_pred.shape[2] == normal_gt.shape[2]
    
    norm_pred = np.linalg.norm(normal_pred, ord=2, axis=2)
    norm_pred = np.expand_dims(norm_pred, axis=2)
    norm_gt = np.linalg.norm(normal_gt, ord=2, axis=2)
    norm_gt = np.expand_dims(norm_gt, axis=2)
    
    normal_pred = normal_pred / np.maximum(norm_pred, eps)
    normal_gt = normal_gt / np.maximum(norm_gt, eps)
    
    product = np.multiply(normal_pred, normal_gt)
    inner_product = np.sum(product, axis=2)
    inner_product_abs = np.absolute(inner_product)
    
    # filter (0.0, 0.0, 0.0) in normal_gt
    invalid_gt = np.sum(np.absolute(normal_gt), axis=2) == 0
    inner_product_abs = inner_product_abs[~invalid_gt]
    
    inner_product_abs = np.clip(inner_product_abs, -1.0 + eps, 1.0 - eps)
    
    angle_diff = np.arccos(inner_product_abs)
    
    mean_angle_diff = np.mean(angle_diff)
    
    return mean_angle_diff
    
  
                 
    
    

  