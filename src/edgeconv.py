import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable


def knn(x, k1, k2):
    batch_size = x.shape[0]
    indices = np.arange(0, k2, k2 // k1)
    with torch.no_grad():
        distances = []
        for b in range(batch_size):
            inner = -2 * torch.matmul(x[b:b + 1].transpose(2, 1), x[b:b + 1])
            xx = torch.sum(x[b:b + 1] ** 2, dim=1, keepdim=True)
            pairwise_distance = -xx - inner - xx.transpose(2, 1)
            distances.append(pairwise_distance)
        distances = torch.stack(distances, 0)
        distances = distances.squeeze(1)
        try:
            idx = distances.topk(k=k2, dim=-1)[1][:, :, indices]
        except:
            import ipdb;
            ipdb.set_trace()
    return idx


def knn_points_normals(x, k1, k2):
    """
    The idea is to design the distance metric for computing 
    nearest neighbors such that the normals are not given
    too much importance while computing the distances.
    Note that this is only used in the first layer.
    """
    batch_size = x.shape[0]
    indices = np.arange(0, k2, k2 // k1)
    with torch.no_grad():
        distances = []
        for b in range(batch_size):
            p = x[b: b + 1, 0:3]
            n = x[b: b + 1, 3:6]

            inner = 2 * torch.matmul(p.transpose(2, 1), p)
            xx = torch.sum(p ** 2, dim=1, keepdim=True)
            p_pairwise_distance = xx - inner + xx.transpose(2, 1)

            inner = 2 * torch.matmul(n.transpose(2, 1), n)
            n_pairwise_distance = 2 - inner

            # This pays less attention to normals
            pairwise_distance = p_pairwise_distance * (1 + n_pairwise_distance)

            # This pays more attention to normals
            # pairwise_distance = p_pairwise_distance * torch.exp(n_pairwise_distance)

            # pays too much attention to normals
            # pairwise_distance = p_pairwise_distance + n_pairwise_distance

            distances.append(-pairwise_distance)

        distances = torch.stack(distances, 0)
        distances = distances.squeeze(1)
        try:
            idx = distances.topk(k=k2, dim=-1)[1][:, :, indices]
        except:
            import ipdb;
            ipdb.set_trace()
        del distances
    return idx


def get_graph_feature(x, k1=20, k2=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

    if idx is None:
        idx = knn(x, k1=k1, k2=k2)

    device = torch.device('cuda') if idx.is_cuda else torch.device("cpu")

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()

    try:
        feature = x.view(batch_size * num_points, -1)[idx, :]
    except:
        import ipdb;
        ipdb.set_trace()
        print(feature.shape)

    feature = feature.view(batch_size, num_points, k1, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k1, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)
    return feature


def get_graph_feature_with_normals(x, k1=20, k2=20, idx=None):
    """
    normals are treated separtely for computing the nearest neighbor
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

    if idx is None:
        idx = knn_points_normals(x, k1=k1, k2=k2)

    device = torch.device('cuda') if idx.is_cuda else torch.device('cpu')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()

    try:
        feature = x.view(batch_size * num_points, -1)[idx, :]
    except:
        import ipdb;
        ipdb.set_trace()
        print(feature.shape)

    feature = feature.view(batch_size, num_points, k1, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k1, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)
    return feature


class DGCNNEncoderGn(nn.Module):
    def __init__(self, use_normal=False, nn_nb=80):
        super(DGCNNEncoderGn, self).__init__()
        self.k = nn_nb
        self.dilation_factor = 1
        self.use_normal = use_normal 
        self.drop = 0.0 # close drop out
        
        self.bn1 = nn.GroupNorm(2, 64)
        self.bn2 = nn.GroupNorm(2, 64)
        self.bn3 = nn.GroupNorm(2, 128)
        self.bn4 = nn.GroupNorm(4, 256)
        self.bn5 = nn.GroupNorm(8, 1024)
        
        input_channels = 3
        if self.use_normal: 
            input_channels = 6

        self.conv1 = nn.Sequential(nn.Conv2d(input_channels * 2, 64, kernel_size=1, bias=False),
                                    self.bn1,
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                    self.bn2,
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                    self.bn3,
                                    nn.LeakyReLU(negative_slope=0.2))

        self.mlp1 = nn.Conv1d(256, 1024, 1)
        self.bnmlp1 = nn.GroupNorm(8, 1024)
        self.mlp1 = nn.Conv1d(256, 1024, 1)
        self.bnmlp1 = nn.GroupNorm(8, 1024)

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.shape[2]

        if self.use_normal == False: 
            # First edge conv
            x = get_graph_feature(x, k1=self.k, k2=self.k)

            x = self.conv1(x)
            x1 = x.max(dim=-1, keepdim=False)[0] # B*64*N

            # Second edge conv
            x = get_graph_feature(x1, k1=self.k, k2=self.k)
            x = self.conv2(x)
            x2 = x.max(dim=-1, keepdim=False)[0]

            # Third edge conv
            x = get_graph_feature(x2, k1=self.k, k2=self.k) # B*64*N
            x = self.conv3(x)
            x3 = x.max(dim=-1, keepdim=False)[0]  # B*128*N

            x_features = torch.cat((x1, x2, x3), dim=1)  # B*256*N
            x = F.relu(self.bnmlp1(self.mlp1(x_features)))

            x4 = x.max(dim=2)[0] # B*1024

            return x4, x_features # B*1024, B*256*N

        if self.use_normal == True: # with normals
            # First edge conv
            x = get_graph_feature_with_normals(x, k1=self.k, k2=self.k)
            x = self.conv1(x)
            x1 = x.max(dim=-1, keepdim=False)[0] # B*64*N

            # Second edge conv
            x = get_graph_feature(x1, k1=self.k, k2=self.k)
            x = self.conv2(x)
            x2 = x.max(dim=-1, keepdim=False)[0] # B*64*N

            # Third edge conv
            x = get_graph_feature(x2, k1=self.k, k2=self.k)
            x = self.conv3(x)
            x3 = x.max(dim=-1, keepdim=False)[0] # B*128*N

            x_features = torch.cat((x1, x2, x3), dim=1) # B*256*N
            x = F.relu(self.bnmlp1(self.mlp1(x_features)))
            x4 = x.max(dim=2)[0] # B*1024

            return x4, x_features


class EmbeddingDGCNGn(nn.Module):
    """
    Segmentation model that takes point cloud as input and returns per
    degree probability, membership matrix and similarity matrix. 
    """
    def __init__(self, use_normal=False, nn_nb=80):
        super(EmbeddingDGCNGn, self).__init__()
        self.use_normal = use_normal
        self.encoder = DGCNNEncoderGn(use_normal=use_normal, nn_nb=nn_nb)
        self.drop = 0.0 # close dropout

        self.conv1 = torch.nn.Conv1d(1024 + 256, 512, 1)
       
        self.bn1 = nn.GroupNorm(8, 512)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)

        self.bn2 = nn.GroupNorm(4, 256)

        self.softmax = torch.nn.Softmax(dim=1)
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.tanh = torch.nn.Tanh()
                
    def forward(self, points):
        batch_size = points.shape[0]
        num_points = points.shape[2]
        x, first_layer_features = self.encoder(points) # B*1024, B*256*N

        x = x.view(batch_size, 1024, 1).repeat(1, 1, num_points)
        x = torch.cat([x, first_layer_features], 1) # B*1280*N

        x = F.dropout(F.relu(self.bn1(self.conv1(x))), self.drop)    # B*512*N
        x_all = F.dropout(F.relu(self.bn2(self.conv2(x))), self.drop) # B*256*N
                           
        return x_all # B*256*N

class BezierEdgeConv(nn.Module):
    def __init__(self, use_normal=False, max_deg_u=3, max_deg_v=3, num_max_instances=75):
        super(BezierEdgeConv, self).__init__()
        self.use_normal = use_normal
        self.backbone = EmbeddingDGCNGn(use_normal=use_normal)
        self.drop = 0.0 # close dropout
        self.softmax = torch.nn.Softmax(dim=1)
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.tanh = nn.Tanh()
        
        num_degrees = max_deg_u * max_deg_v
        self.max_num_ctrlpts_u = max_deg_u + 1
        self.max_num_ctrlpts_v = max_deg_v + 1
        
        # self.encode_degree_dict = {(1,1):0, (1,2):1, (1,3):2,  
        #              (2,1):3, (2,2):4, (2,3):5,  
        #              (3,1):6, (3,2):7, (3,3):8}
        # self.decode_degree_dict = {0:(1,1), 1:(1,2), 2:(1,3),  
        #              3:(2,1), 4:(2,2), 5:(2,3),  
        #              6:(3,1), 7:(3,2), 8:(3,3)}
        self.encode_degree_dict = {}
        self.decode_degree_dict = {}
        for i in range(1, max_deg_u + 1):
            for j in range(1, max_deg_v + 1):
                self.encode_degree_dict[(i, j)] = (i - 1) * max_deg_u + (j - 1)
                self.decode_degree_dict[(i - 1) * max_deg_u + (j - 1)] = (i, j)

        input_channels = 3
        if self.use_normal: 
            input_channels = 6
             
        # point degree classification
        self.mlp_deg_fc1 = torch.nn.Conv1d(256+input_channels, 256, 1)
        self.mlp_deg_fc2 = torch.nn.Conv1d(256, num_degrees, 1)
        self.bn_deg1 = nn.GroupNorm(4, 256) 

        # membership matrix       
        self.mlp_seg_fc1 = torch.nn.Conv1d(256+input_channels+num_degrees, 256, 1)
        self.mlp_seg_fc2 = torch.nn.Conv1d(256, num_max_instances, 1)
        self.bn_seg1 = nn.GroupNorm(4, 256) 
        
        # uv regression      
        self.mlp_uv_fc1 = torch.nn.Conv1d(256+input_channels+num_degrees+num_max_instances, 256, 1)
        self.mlp_uv_fc2 = torch.nn.Conv1d(256, 2, 1)
        self.bn_uv1 = nn.GroupNorm(4, 256) 

        # control points (x, y, z, w)
        self.mlp_ctrlpts_fc1 = torch.nn.Conv1d(256+input_channels+num_degrees+2, 256, 1)
        self.mlp_ctrlpts_fc2 = torch.nn.Conv1d(256, self.max_num_ctrlpts_u*self.max_num_ctrlpts_v*4, 1)
        self.bn_ctrlpts1 = nn.GroupNorm(4, 256)
        
        # auto SE-block weight
        # self.auto_wegight_maxpool1 = nn.AdaptiveMaxPool1d(1)
        # self.auto_weight_fc1 = torch.nn.Conv1d(256+input_channels+num_degrees+num_max_instances, 32, 1) 
        # self.auto_weight_fc2 = torch.nn.Conv1d(32, 3, 1) 
        # self.bn_auto_weight1 = nn.GroupNorm(4, 32) 
                    
    def forward(self, xyz):
        # xyz:B*3(or 6)*N
        # pt_deg_logp: B*num_degrees*N; 
        # x_all:B*256*N
        # N is num_pts
        x_all = self.backbone(xyz) 
        
        # point degree classification
        x0 = torch.cat((x_all, xyz), dim=1)
        x = F.dropout(F.relu(self.bn_deg1(self.mlp_deg_fc1(x0))), self.drop)
        x = self.mlp_deg_fc2(x)
        pt_deg_logp = self.logsoftmax(x) # B*num_degrees*N 
        pt_deg_prob = torch.exp(pt_deg_logp) # B*num_degrees*N
        
        # membership matrix
        x0 = torch.cat((x_all, xyz, pt_deg_prob), dim=1)
        x = F.dropout(F.relu(self.bn_seg1(self.mlp_seg_fc1(x0))), self.drop)
        W_pred = self.mlp_seg_fc2(x)  # B*num_max_instances*N
        W_pred = self.softmax(W_pred) # B*num_max_instances*N 
        
        # instance degree score
        I_deg_score = pt_deg_prob @ W_pred.permute(0,2,1)  # B*num_degrees*num_max_instances
        
        # uv regression
        x0 = torch.cat((x_all, xyz, pt_deg_prob, W_pred), dim=1)
        x = F.dropout(F.relu(self.bn_uv1(self.mlp_uv_fc1(x0))), self.drop) #B*256*N
        uv = self.mlp_uv_fc2(x)  # B*2*N
        uv = torch.sigmoid(uv) # B*2*N

        # control points (x, y, z, w)
        x0 = torch.cat((x_all, xyz, pt_deg_prob, uv), dim=1) #B*(256+input_channels+num_degrees+2)*N
        x0 = torch.bmm(x0, W_pred.permute(0, 2, 1)) #B*(256+input_channels+num_degrees+2)*num_max_instances
        x = F.dropout(F.relu(self.bn_ctrlpts1(self.mlp_ctrlpts_fc1(x0))), self.drop) #B*256*num_max_instances
        ctrlpts = self.mlp_ctrlpts_fc2(x) #B*(max_num_ctrlpts_u*max_num_ctrlpts_v*4)*num_max_instances
        ctrlpts = ctrlpts.permute(0, 2, 1) #B*(max_num_ctrlpts_u*max_num_ctrlpts_v*4)*num_max_instances
        # regularize weights
        B, num_max_instances, _ = ctrlpts.shape
        ctrlpts = ctrlpts.reshape((B, num_max_instances, self.max_num_ctrlpts_u*self.max_num_ctrlpts_v, 4))
        coords = ctrlpts[:, :, :, 0:3] #B*num_max_instances*(max_num_ctrlpts_u*max_num_ctrlpts_v)*3
        coords = self.tanh(coords) #B*num_max_instances*(max_num_ctrlpts_u*max_num_ctrlpts_v)
        weight = ctrlpts[:, :, :, 3] #B*num_max_instances*(max_num_ctrlpts_u*max_num_ctrlpts_v)
        weight = F.softmax(weight, dim=2)
        
        ctrlpts = torch.cat((coords, weight.unsqueeze(dim=3)), dim=3)
        ctrlpts = ctrlpts.reshape((B,num_max_instances,
                                self.max_num_ctrlpts_u,
                                self.max_num_ctrlpts_v,
                                4))
        
        return pt_deg_logp, W_pred, I_deg_score, uv, ctrlpts, x_all
    


    
