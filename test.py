import os
import argparse
import shutil
import math
import torch
import torch.optim as optim
from src.edgeconv import BezierEdgeConv
from src.bezier_dataset import *
from src.segment_loss import *
from src.fitting_loss import *
from src.embedding_loss import *
from src.segment_utils import *
from options import build_options
            
def cuda_setup(use_gpu):
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device

def test(args):
    
    use_gpu = args.use_gpu 
    use_DataParallel = args.use_DataParallel  
    input_normal = args.input_normal
    output_normal = args.output_normal
    use_normal_loss = args.use_normal_loss
    num_workers = args.num_workers
    max_deg_u = args.max_deg_u
    max_deg_v = args.max_deg_v
    num_max_instances = args.num_max_instances
    batch_size = 1 # we set batch_size to be 1 only for testing
    
    result_dir = args.result_dir
    data_path = args.data_path
    checkpoint_path = './result/model-150/checkpoint_150.pt'
    
    device = cuda_setup(use_gpu)
    
    print("init model")
    model = BezierEdgeConv(
        use_normal=input_normal,
        max_deg_u=3, max_deg_v=3,
        num_max_instances=num_max_instances
    )
    model = model.to(device)
 
    
    use_DataParallel = use_gpu and use_DataParallel and (torch.cuda.device_count() > 1)
    if use_DataParallel:
        print("use DataParallel")
        model = torch.nn.DataParallel(model)
     
    print("init test Dataset")
    test_dataset = BezierDataset(root=data_path, batch_size=batch_size, split='test')
    print("init test DataLoader")
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False)
    
    
    # load pretrained model if its path is not empty
    if checkpoint_path:
        print("use checkpoint")
        checkpoint = torch.load(checkpoint_path)
        if use_DataParallel:
            # save on GPU DataParallel and read on GPU DataParallel
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        else:
            # save on GPU DataParallel and read on a single CPU or GPU
            model.load_state_dict(checkpoint['model_module_state_dict'], strict=True)
    else:
        print("no input checkpoint")

    # validate on the whole test data         
    test_avg_primitive_type_acc = 0.0
    test_avg_mean_angle_diff = 0.0
    test_avg_rand_score = 0.0
    test_avg_num = 0.0
       
    model.eval()
    with torch.no_grad():
        for j, test_data in enumerate(test_dataloader, 0):
            batch_sample_coordinates, batch_sample_normals, \
                batch_sample_parameters, batch_sample_degrees, \
                    batch_sample_patch_ids,  batch_patch_degrees, \
                        batch_control_points, batch_file_ids = test_data
                        
            batch_sample_coordinates, batch_control_points = \
                        test_dataset.normalize(batch_sample_coordinates, batch_control_points, 
                                                batch_sample_patch_ids, batch_patch_degrees)
            
            batch_sample_coordinates = batch_sample_coordinates.to(device)
            batch_sample_normals = batch_sample_normals.to(device)
            batch_sample_parameters = batch_sample_parameters.to(device)
            batch_sample_degrees = batch_sample_degrees.to(device)
            batch_sample_patch_ids = batch_sample_patch_ids.to(device) 
            batch_patch_degrees = batch_patch_degrees.to(device)  
            batch_control_points = batch_control_points.to(device)  
            
            batch_ins_deg_labels = encode_degrees_to_labels(batch_patch_degrees, max_deg_u=max_deg_u, max_deg_v=max_deg_v)

            batch_control_points = regularize_ctrlpts_weight(batch_control_points, batch_patch_degrees)
            
            if not input_normal:
                inputs = batch_sample_coordinates
            else:
                inputs = torch.cat([batch_sample_coordinates,
                                  batch_sample_normals],
                                  dim=2)
                    
            pt_deg_logp, W_prob, I_deg_score, uv, ctrlpts, _ = model(inputs.permute(0, 2, 1))
            
            I_deg_pred = deg_per_instance(I_deg_score)
            I_deg_uv_pred = decode_labels_to_degrees(I_deg_pred, max_deg_u=max_deg_u, max_deg_v=max_deg_v)
            I_pred = ins_per_point(W_prob)
            pt_voting_deg = deg_per_point_from_ins(I_pred, I_deg_pred)
            pt_voting_deg_uv = decode_labels_to_degrees(pt_voting_deg, max_deg_u=max_deg_u, max_deg_v=max_deg_v)
            
            ctrlpts = regularize_ctrlpts_weight(ctrlpts, I_deg_uv_pred)
            hctrlpts = homogeneous_coordiantes(ctrlpts)         
           
            _, match_indices = mean_relaxed_iou_loss(W_prob, batch_sample_patch_ids)
            # match pred_batch_part_labels to GT
            I_pred_reorder, I_deg_reorder, hctrlpts_reorder = reorder_ins_labels(
                        I_pred, I_deg_pred, hctrlpts, batch_sample_patch_ids)
         
           
            
            if not output_normal:
                recon_coords = reconstruct_coordinates(ctrlpts, uv.permute(0, 2, 1), W_prob.permute(0, 2, 1), pt_voting_deg_uv, eps=1e-12)
            else:
                recon_coords, recon_normals = reconstruct_coordinates_normals(ctrlpts, uv.permute(0, 2, 1), W_prob.permute(0, 2, 1),
                                                                                     pt_voting_deg_uv, eps=1e-12)
            loss_coords = coords_loss(recon_coords, batch_sample_coordinates)
            if (use_normal_loss and output_normal):
                loss_normals = normals_loss(recon_normals, batch_sample_normals)
            else:
                loss_normals = 0.0
           

            primitive_type_acc = eval_primitive_type_acc(I_deg_pred.cpu().detach().numpy(),
                                                batch_ins_deg_labels.cpu().detach().numpy(),
                                                match_indices)
            cluster_rand_score = eval_cluster_rand_score(batch_sample_patch_ids.cpu().detach().numpy(), 
                                                I_pred_reorder.cpu().detach().numpy())
        
            if output_normal:
                mean_angle_diff = eval_pt_normal_angle_diff(recon_normals.cpu().detach().numpy(),
                                          batch_sample_normals.cpu().detach().numpy())
                
            num = get_number_of_primitives(I_pred_reorder)
            
            test_avg_primitive_type_acc += primitive_type_acc
            test_avg_rand_score += cluster_rand_score
            test_avg_mean_angle_diff += mean_angle_diff
            test_avg_num += num
            
            # save
            save_dir = os.path.join(result_dir, "Test-Segmentation")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            test_dataset.color_map_patch_obj(batch_sample_coordinates.cpu().detach().numpy(), 
                                            I_pred_reorder.cpu().detach().numpy(), 
                                            save_dir, batch_file_ids)
            
                                                     
        test_avg_primitive_type_acc /= (j + 1)
        test_avg_rand_score /= (j + 1)
        test_avg_mean_angle_diff /= (j + 1)
        test_avg_num /= (j + 1)
        
        print(("[test-epoch] test_avg_primitive_type_acc:%f, test_avg_rand_score:%f, test_avg_mean_angle_diff:%f, avg_num:%f") % 
                    (test_avg_primitive_type_acc, test_avg_rand_score, test_avg_mean_angle_diff, test_avg_num))

            
if __name__ == '__main__':
    args = build_options()
    test(args)