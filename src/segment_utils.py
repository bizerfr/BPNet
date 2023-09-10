import numpy as np
import torch

def encode_degrees_to_labels_one_sample(degrees, max_deg_u=3, max_deg_v=3):
    '''
    (-1, -1) - negative
    (1, 1) - 0, (1, 2) - 1, (1, 3) - 2
    (2, 1) - 3, (2, 2) - 4, (2, 3) - 5
    (3, 1) - 6, (3, 2) - 7, (3, 3) - 8
    ''' 
    assert max_deg_u > 0
    assert max_deg_v > 0
    assert len(degrees.shape) == 2
    
    is_tensor = torch.is_tensor(degrees)
    dtype = degrees.dtype
    
    N, _  = degrees.shape
    
    if is_tensor:
        device = degrees.device
        degrees = degrees.cpu().detach().numpy()
        
    deg_u = degrees[:, 0] # B*N
    deg_v = degrees[:, 1] # B*N
    degree_labels = (deg_u - 1) * max_deg_u + deg_v - 1
    
    degree_labels[degree_labels < 0] = -1
    
    if is_tensor: 
        return torch.tensor(degree_labels, dtype=dtype).to(device)
    else:
        return degree_labels.astype(dtype)

def encode_degrees_to_labels_batch(degrees, max_deg_u=3, max_deg_v=3):
    '''
    (-1, -1) - negative
    (1, 1) - 0, (1, 2) - 1, (1, 3) - 2
    (2, 1) - 3, (2, 2) - 4, (2, 3) - 5
    (3, 1) - 6, (3, 2) - 7, (3, 3) - 8
    '''  
    assert max_deg_u > 0
    assert max_deg_v > 0
    assert len(degrees.shape) == 3
    
    is_tensor = torch.is_tensor(degrees)
    dtype = degrees.dtype
    
    B, N, _  = degrees.shape
    
    if is_tensor:
        device = degrees.device
        degrees = degrees.cpu().detach().numpy()
        
    deg_u = degrees[:, :, 0] # B*N
    deg_v = degrees[:, :, 1] # B*N
    degree_labels = (deg_u - 1) * max_deg_u + deg_v - 1
    
    degree_labels[degree_labels < 0] = -1
    
    if is_tensor:
        return torch.tensor(degree_labels, dtype=dtype).to(device)
    else:
        return degree_labels.astype(dtype)

def encode_degrees_to_labels(degrees, max_deg_u=3, max_deg_v=3):
    '''
    (1, 1) - 0, (1, 2) - 1, (1, 3) - 2
    (2, 1) - 3, (2, 2) - 4, (2, 3) - 5
    (3, 1) - 6, (3, 2) - 7, (3, 3) - 8
    '''  
    assert max_deg_u > 0
    assert max_deg_v > 0
    
    if len(degrees.shape) == 3:
        return encode_degrees_to_labels_batch(degrees, max_deg_u=max_deg_u, max_deg_v=max_deg_v)
    elif len(degrees.shape) == 2:
        return encode_degrees_to_labels_one_sample(degrees, max_deg_u=max_deg_u, max_deg_v=max_deg_v)

def decode_labels_to_degrees_one_sample(degree_labels, max_deg_u=3, max_deg_v=3):
    '''
    {0:(1,1), 1:(1,2), 2:(1,3),  
     3:(2,1), 4:(2,2), 5:(2,3),  
     6:(3,1), 7:(3,2), 8:(3,3)}
    '''
    assert len(degree_labels.shape) == 1
    
    is_tensor = torch.is_tensor(degrees)
    dtype = degree_labels.dtype
    
    N = degree_labels.shape
    
    if is_tensor:
        device = degree_labels.device
        degree_labels = degree_labels.cpu().detach().numpy()
        
    degrees = np.full((N, 2), -1, dtype=np.int)
        
    degrees[:, 0] = degree_labels // max_deg_u + 1
    degrees[:, 1] = degree_labels - (degrees[:, 0] - 1) * max_deg_u + 1
    
    invalid = degree_labels == -1
    degrees[invalid, :] = -1
    
    if is_tensor:
        return torch.tensor(degrees, dtype=dtype).to(device)
    else:
        return degrees.astype(dtype)

def decode_labels_to_degrees_batch(degree_labels, max_deg_u=3, max_deg_v=3):
    '''
    {0:(1,1), 1:(1,2), 2:(1,3),  
     3:(2,1), 4:(2,2), 5:(2,3),  
     6:(3,1), 7:(3,2), 8:(3,3)}
    '''
    assert len(degree_labels.shape) == 2
    
    is_tensor = torch.is_tensor(degree_labels)
    dtype = degree_labels.dtype
    
    B, N = degree_labels.shape
    
    if is_tensor:
        device = degree_labels.device 
        degree_labels = degree_labels.cpu().detach().numpy()
        
    degrees = np.full((B, N, 2), -1, dtype=np.int)
        
    degrees[:, :, 0] = degree_labels // max_deg_u + 1
    degrees[:, :, 1] = degree_labels - (degrees[:, :, 0] - 1) * max_deg_u + 1
    
    invalid = degree_labels == -1
    degrees[invalid, :] = -1
    
    if is_tensor:
        return torch.tensor(degrees, dtype=dtype).to(device)
    else:
        return degrees.astype(dtype)

def decode_labels_to_degrees(degree_labels, max_deg_u=3, max_deg_v=3):
    '''
     0:(1,1), 1:(1,2), 2:(1,3),  
     3:(2,1), 4:(2,2), 5:(2,3),  
     6:(3,1), 7:(3,2), 8:(3,3)
    '''
    if len(degree_labels.shape) == 2:
        return decode_labels_to_degrees_batch(degree_labels, max_deg_u=max_deg_u, max_deg_v=max_deg_v)
    elif len(degree_labels.shape) == 1:
        return decode_labels_to_degrees_one_sample(degree_labels, max_deg_u=max_deg_u, max_deg_v=max_deg_v)    
        
def nms(pt_deg_logp, W_pred, I_deg_score, drop=0.01):
    """ 
    pt_deg_logp: B*C*N 
    W_pred: B*Kmax*N
    I_deg_score: B*C*Kmax, soft point number for each degree
    """
    B, C, N = pt_deg_logp.shape
    
    assert B == W_pred.shape[0]
    assert N == W_pred.shape[2]
    assert B == I_deg_score.shape[0]
    assert C == I_deg_score.shape[1]
    
    
    Kmax = W_pred.shape[1]
    
    assert Kmax == I_deg_score.shape[2]
    
    assert torch.abs(B*N - I_deg_score.sum()) < 1.0
    
    pt_deg_logp = pt_deg_logp.cpu().detach() #B*C*N 
    W_pred = W_pred.cpu().detach() #B*Kmax*N
    I_deg_score = I_deg_score.cpu().detach() #B*C*Kmax
    
    I_soft_size = I_deg_score.sum(dim=1) #B*Kmax
    invalid = I_soft_size < drop * N #B*Kmax
    W_pred[invalid, :] = 0.0
    
    pt_deg_pred = torch.argmax(pt_deg_logp, dim=1) #B*N
    I_deg_pred = torch.argmax(I_deg_score, dim=1) #B*Kmax
    
    pt_deg_pred = pt_deg_pred.unsqueeze(1).expand(B, Kmax, N)
    I_deg_pred = I_deg_pred.unsqueeze(2).expand(B, Kmax, N)
    
    invalid = pt_deg_pred != I_deg_pred #B*Kmax*N
    invalid = torch.any(invalid, dim=2) #B*Kmax
    invalid = ~invalid
    W_pred[invalid, :] = 0.0
    
    I_pred = torch.argmax(W_pred, dim=1) #B*N
    
    return I_pred 
         