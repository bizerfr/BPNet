import os
import shutil
import math
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter 
from src.edgeconv import BezierEdgeConv
from src.bezier_dataset import BezierDataset
from src.segment_loss import *
from src.segment_utils import *
from src.fitting_loss import *
from src.embedding_loss import *
from options import build_options

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.benchmark = False

g = torch.Generator()
g.manual_seed(0)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def weights_init(m):
    classname = m.__class__.__name__
    if classname in ('Conv1d', 'Linear'):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
            
def cuda_setup(use_gpu):
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device

def train(args): 
    print("train...")
    use_gpu = args.use_gpu 
    use_DataParallel = args.use_DataParallel 
    input_normal = args.input_normal
    output_normal = args.output_normal
    use_normal_loss = args.use_normal_loss
    num_workers = args.num_workers
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.learning_rate
    max_deg_u = args.max_deg_u
    max_deg_v = args.max_deg_v
    num_max_instances = args.num_max_instances
    shuffle_train = args.shuffle_train
    
    test_epoch_frequency = args.test_epoch_frequency
    result_dir = args.result_dir
    data_path = args.data_path
    checkpoint_path = args.checkpoint_path

    # Dataset
    print("init train Dataset")
    train_dataset = BezierDataset(root=data_path, batch_size=batch_size, split='train')
    print("init test Dataset")
    test_dataset = BezierDataset(root=data_path, batch_size=batch_size, split='test')
    
    print("init train DataLoader")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g)
        
    num_points = train_dataset.npoints
    assert num_points == test_dataset.npoints
    
    decode_degree_dict = train_dataset.decode_degree_dict

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    
    writer = SummaryWriter()
    device = cuda_setup(use_gpu)
    
    print("init model")
    # GT of patch id is inconsistent
    model = BezierEdgeConv(
        use_normal=input_normal,
        max_deg_u=max_deg_u,
        max_deg_v=max_deg_v,
        num_max_instances=num_max_instances
    )
    model = model.to(device)
    
    use_DataParallel = use_gpu and use_DataParallel and (torch.cuda.device_count() > 1)
    if use_DataParallel:
        print("use DataParallel")
        model = torch.nn.DataParallel(model)
    
    model.apply(weights_init)

    # different branch using different learning rates
    if use_DataParallel:
        base_params =  model.module.backbone.parameters()
    
        deg_cls_params = [ p for p in model.module.mlp_deg_fc1.parameters()]
        deg_cls_params += [ p for p in model.module.mlp_deg_fc2.parameters()]
        deg_cls_params += [ p for p in model.module.bn_deg1.parameters()]
    
        ins_seg_params = [ p for p in model.module.mlp_seg_fc1.parameters()]
        ins_seg_params += [ p for p in model.module.mlp_seg_fc2.parameters()]
        ins_seg_params += [ p for p in model.module.bn_seg1.parameters()]
    
        uv_reg_params = [ p for p in model.module.mlp_uv_fc1.parameters()]
        uv_reg_params += [ p for p in model.module.mlp_uv_fc2.parameters()]
        uv_reg_params += [ p for p in model.module.bn_uv1.parameters()]
    
        ctrl_reg_params = [ p for p in model.module.mlp_ctrlpts_fc1.parameters()]
        ctrl_reg_params += [ p for p in model.module.mlp_ctrlpts_fc2.parameters()]
        ctrl_reg_params += [ p for p in model.module.bn_ctrlpts1.parameters()]
    else:
        base_params =  model.backbone.parameters()
    
        deg_cls_params = [ p for p in model.mlp_deg_fc1.parameters()]
        deg_cls_params += [ p for p in model.mlp_deg_fc2.parameters()]
        deg_cls_params += [ p for p in model.bn_deg1.parameters()]
    
        ins_seg_params = [ p for p in model.mlp_seg_fc1.parameters()]
        ins_seg_params += [ p for p in model.mlp_seg_fc2.parameters()]
        ins_seg_params += [ p for p in model.bn_seg1.parameters()]
    
        uv_reg_params = [ p for p in model.mlp_uv_fc1.parameters()]
        uv_reg_params += [ p for p in model.mlp_uv_fc2.parameters()]
        uv_reg_params += [ p for p in model.bn_uv1.parameters()]
    
        ctrl_reg_params = [ p for p in model.mlp_ctrlpts_fc1.parameters()]
        ctrl_reg_params += [ p for p in model.mlp_ctrlpts_fc2.parameters()]
        ctrl_reg_params += [ p for p in model.bn_ctrlpts1.parameters()]
    
    params = [
        {"params": base_params, "lr": lr},
        {"params": deg_cls_params, "lr": lr*0.1},
        {"params": ins_seg_params, "lr": lr},
        {"params": uv_reg_params, "lr": lr},
        {"params": ctrl_reg_params, "lr": lr*0.1}]

    
    print("init optimizer")
    optimizer = optim.Adam(params)
    
    iter_train_times = math.ceil(len(train_dataset) / args.batch_size)
    
    start_epoch = 0
    # load pretrained model if checkpoint path is not empty
    if checkpoint_path:
        print("use checkpoint")
        checkpoint = torch.load(checkpoint_path)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if use_DataParallel:
            # save on GPU DataParallel and read on GPU DataParallel
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        else:
            # save on GPU DataParallel and read on a single CPU or GPU
            model.load_state_dict(checkpoint['model_module_state_dict'], strict=True)
    
    for epoch in range(start_epoch, epochs + 1):
        train_epoch_loss_pt_deg_cls = 0.0
        train_epoch_loss_mean_riou = 0.0
        train_epoch_loss_soft_voting = 0.0
        train_epoch_loss_pull = 0.0
        train_epoch_loss_push = 0.0
        train_epoch_loss_paras = 0.0
        train_epoch_loss_ctrlpts = 0.0
        train_epoch_loss_coords = 0.0
        train_epoch_loss_normals = 0.0
        train_epoch_loss = 0.0
        
        
        for i, train_data in enumerate(train_dataloader, 0):
            optimizer.zero_grad()
            model.train()
            
            batch_sample_coordinates, batch_sample_normals, \
                batch_sample_parameters, batch_sample_degrees, \
                    batch_sample_patch_ids,  batch_patch_degrees, \
                        batch_control_points, batch_file_ids = train_data
                        
            batch_sample_coordinates, batch_control_points = \
                        train_dataset.normalize(batch_sample_coordinates, batch_control_points, 
                                                batch_sample_patch_ids, batch_patch_degrees)
                        
            batch_sample_coordinates = batch_sample_coordinates.to(device)
            batch_sample_normals = batch_sample_normals.to(device)
            batch_sample_parameters = batch_sample_parameters.to(device)
            batch_sample_degrees = batch_sample_degrees.to(device)
            batch_sample_patch_ids = batch_sample_patch_ids.to(device) 
            batch_patch_degrees = batch_patch_degrees.to(device) 
            batch_control_points = batch_control_points.to(device)
            
            batch_pt_deg_labels = encode_degrees_to_labels(batch_sample_degrees, 
                                                           max_deg_u=max_deg_u, max_deg_v=max_deg_v)
            batch_ins_deg_labels = encode_degrees_to_labels(batch_patch_degrees,
                                                            max_deg_u=max_deg_u, max_deg_v=max_deg_v)

            batch_control_points = regularize_ctrlpts_weight(batch_control_points, batch_patch_degrees)
            batch_hcontrol_points = homogeneous_coordiantes(batch_control_points)
            
            if not input_normal:
                inputs = batch_sample_coordinates
            else:
                inputs = torch.cat([batch_sample_coordinates,
                                  batch_sample_normals],
                                  dim=2)

            pt_deg_logp, W_prob, I_deg_score, uv, ctrlpts, pt_embed = model(inputs.permute(0, 2, 1))
            
            I_deg_pred = deg_per_instance(I_deg_score)
            I_deg_uv_pred = decode_labels_to_degrees(I_deg_pred, max_deg_u=max_deg_u, max_deg_v=max_deg_v)
            I_pred = ins_per_point(W_prob)
            pt_voting_deg = deg_per_point_from_ins(I_pred, I_deg_pred)
            pt_voting_deg_uv = decode_labels_to_degrees(pt_voting_deg, max_deg_u=max_deg_u, max_deg_v=max_deg_v)
            
            ctrlpts = regularize_ctrlpts_weight(ctrlpts, I_deg_uv_pred)
            hctrlpts = homogeneous_coordiantes(ctrlpts)
            
            loss_pt_deg_cls = pt_deg_cls_loss(pt_deg_logp, batch_pt_deg_labels)
            loss_mean_riou, match_indices = mean_relaxed_iou_loss(W_prob, batch_sample_patch_ids)
            loss_soft_voting = soft_voting_loss(I_deg_score, batch_ins_deg_labels, match_indices)
            loss_paras = paras_loss(uv.permute(0, 2, 1), batch_sample_parameters)
            loss_ctrlpts = ctrlpts_loss(hctrlpts, I_deg_uv_pred,
                                        batch_hcontrol_points, batch_patch_degrees,
                                        match_indices, decode_degree_dict)
            loss_pull, loss_push = embedding_loss(W_prob, pt_embed, match_indices)
            
            if not output_normal:
                recon_coords = reconstruct_coordinates(ctrlpts, uv.permute(0, 2, 1), W_prob.permute(0, 2, 1), pt_voting_deg_uv, eps=1e-12)
            else:
                recon_coords, recon_normals = reconstruct_coordinates_normals(ctrlpts, uv.permute(0, 2, 1), W_prob.permute(0, 2, 1),
                                                                                     pt_voting_deg_uv, eps=1e-12)
                
            loss_coords = coords_loss(recon_coords, batch_sample_coordinates)
            if (use_normal_loss and output_normal):
                loss_normals = normals_loss(recon_normals, batch_sample_normals)
            else:
                loss_normals = torch.tensor(0.0).to(device)
            
            # decomposition:  loss_pt_deg_cls + loss_mean_riou + loss_soft_voting
            # fitting: loss_paras + loss_ctrlpts
            # embedding: loss_pull + loss_push
            # reconstruction: loss_coords (+ loss_normals)
            if (use_normal_loss and output_normal):
                loss = loss_pt_deg_cls + loss_mean_riou + loss_soft_voting + loss_pull + loss_push \
                   + loss_paras + loss_ctrlpts + loss_coords + loss_normals
            else:
                loss = loss_pt_deg_cls + loss_mean_riou + loss_soft_voting + loss_pull + loss_push \
                   + loss_paras + loss_ctrlpts + loss_coords
                
 
            loss.backward()
            optimizer.step()
       
            # record the epoch
            train_epoch_loss_pt_deg_cls += loss_pt_deg_cls
            train_epoch_loss_mean_riou += loss_mean_riou
            train_epoch_loss_soft_voting += loss_soft_voting
            train_epoch_loss_pull += loss_pull
            train_epoch_loss_push += loss_push
            train_epoch_loss_paras += loss_paras
            train_epoch_loss_ctrlpts += loss_ctrlpts
            train_epoch_loss_coords += loss_coords
            train_epoch_loss_normals += loss_normals
            train_epoch_loss += loss 
            
                  
            print(("[train-batch] epoch:%d, iters:%d, " 
                "loss_pt_deg_cls:%f, loss_mean_riou:%f, loss_soft_voting:%f, \ns"
                "loss_pull:%f, loss_push:%f \n"
                "loss_paras:%f, loss_ctrlpts:%f, \n"
                "loss_coords:%f, loss_normals:%f, \n"
                "loss:%f, \n") % 
                (epoch, iter_train_times * epoch + i,
                 loss_pt_deg_cls.item(), loss_mean_riou.item(), loss_soft_voting.item(),
                 loss_pull.item(), loss_push.item(),
                 loss_paras.item(), loss_ctrlpts.item(), loss_coords.item(), loss_normals.item(),
                 loss.item())) 
        
        train_epoch_loss_pt_deg_cls /= (i + 1)
        train_epoch_loss_mean_riou /= (i + 1)
        train_epoch_loss_soft_voting /= (i + 1)
        train_epoch_loss_pull /= (i + 1)
        train_epoch_loss_push /= (i + 1)
        train_epoch_loss_paras /= (i + 1)
        train_epoch_loss_ctrlpts /= (i + 1)
        train_epoch_loss_coords /= (i + 1)
        train_epoch_loss_normals /= (i + 1)
        train_epoch_loss /= (i + 1)
        
                            
        print(("[train-epoch] epoch:%d, "
              "train_epoch_loss_pt_deg_cls:%f, train_epoch_loss_mean_riou:%f, train_epoch_loss_soft_voting:%f, \n"
              "train_epoch_loss_pull:%f, train_epoch_loss_push:%f, \n"
              "train_epoch_loss_paras:%f, train_epoch_loss_ctrlpts:%f,\n "
              "train_epoch_loss_coords:%f, train_epoch_loss_normals: %f\n"
              "train_epoch_loss:%f,  \n") % 
              (epoch, 
               train_epoch_loss_pt_deg_cls, train_epoch_loss_mean_riou, train_epoch_loss_soft_voting,
               train_epoch_loss_pull, train_epoch_loss_push,
               train_epoch_loss_paras, train_epoch_loss_ctrlpts, 
               train_epoch_loss_coords, train_epoch_loss_normals,
               train_epoch_loss))
        
        writer_dict={"train_epoch_loss_pt_deg_cls" : train_epoch_loss_pt_deg_cls, 
                    "train_epoch_loss_mean_riou" : train_epoch_loss_mean_riou, 
                    "train_epoch_loss_soft_voting" : train_epoch_loss_soft_voting,
                    "train_epoch_loss_pull" : train_epoch_loss_pull,
                    "train_epoch_loss_push" : train_epoch_loss_push,
                    "train_epoch_loss_paras" : train_epoch_loss_paras,
                    "train_epoch_loss_ctrlpts" : train_epoch_loss_ctrlpts,
                    "train_epoch_loss_coords" : train_epoch_loss_coords,
                    "train_epoch_loss_normals" : train_epoch_loss_normals,
                    "train_epoch_loss" : train_epoch_loss}
            
        for title, value in writer_dict.items():
            writer.add_scalar("[train-epoch] " + title, value, epoch)
        
        if epoch % test_epoch_frequency == 0:
            # save the check-point
            epoch_dir = os.path.join(result_dir, str(epoch))
            if not os.path.exists(epoch_dir):
                os.makedirs(epoch_dir)
            if use_DataParallel:
                # save on GPU DataParallel and read on GPU DataParallel: model_state_dict
                # save on GPU DataParallel and read on a single CPU or GPU: model_module_state_dict
                torch.save(
                    {'epoch': epoch,
                    'model_state_dict': model.state_dict(),     
                    'model_module_state_dict': model.module.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict()}, 
                    '%s/checkpoint_%d.pt' % (epoch_dir, epoch))
            else:
                torch.save(
                    {'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, 
                    '%s/checkpoint_%d.pt' % (epoch_dir, epoch))
                            
            writer.flush()
            
if __name__ == '__main__':
    args = build_options()
    train(args)