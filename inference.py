import os
import argparse
import shutil
import math
import torch
import torch.optim as optim
from src.edgeconv import BezierEdgeConv
from src.xyz_dataset import *
from src.segment_loss import *
from src.fitting_loss import *
from src.segment_utils import *
            
def cuda_setup(use_gpu):
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device

# Parsing Arguments
parser = argparse.ArgumentParser()
# Experiment Settings
parser.add_argument('--use_gpu', type=bool, default=True, help='if use GPU or not')
parser.add_argument('--use_DataParallel', type=bool, default=True, help='if use torch.nn.DataParallel or not')
parser.add_argument('--input_normal', type=bool, default=True, help='input normals to NN or not')
parser.add_argument('--output_normal', type=bool, default=False, help='output normals or not')
parser.add_argument('--num_workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 32]')
parser.add_argument('--max_deg_u', type=int, default=3, help='Maximum degree number in one sample')
parser.add_argument('--max_deg_v', type=int, default=3, help='Maximum degree number in one sample')
parser.add_argument('--num_max_instances', type=int, default=75, help='Maximum patch number in one sample')
parser.add_argument('--result_dir', type=str, default='result', help='result folder')
parser.add_argument('--data_path', type=str, 
        default='./data/points-with-normals', help='data root path')
parser.add_argument('--checkpoint_path', type=str, 
    default='./result/model-150/checkpoint_150.pt', 
    help='checkpoint path')

def test():
    args = parser.parse_args()
    
    use_gpu = args.use_gpu 
    use_DataParallel = args.use_DataParallel  
    input_normal = args.input_normal
    output_normal = args.output_normal
    num_workers = args.num_workers
    batch_size = args.batch_size
    max_deg_u = args.max_deg_u
    max_deg_v = args.max_deg_v
    num_max_instances = args.num_max_instances
    
    result_dir = args.result_dir
    data_path = args.data_path
    checkpoint_path = args.checkpoint_path
    
    
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
    test_dataset = XYZDataset(data_dir=data_path, batch_size=batch_size)
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
       
    model.eval()
    with torch.no_grad():
        for j, test_data in enumerate(test_dataloader, 0):
            batch_sample_coordinates, batch_sample_normals, batch_file_ids = test_data
                        
            batch_sample_coordinates, batch_sample_normals = \
                        test_dataset.normalize(batch_sample_coordinates, batch_sample_normals)
           
            batch_sample_coordinates = batch_sample_coordinates.to(device)
            batch_sample_normals = batch_sample_normals.to(device)
            
            if not input_normal:
                inputs = batch_sample_coordinates
            else:
                inputs = torch.cat([batch_sample_coordinates,
                                  batch_sample_normals],
                                  dim=2)
             
            pt_deg_logp, W_prob, I_deg_score, uv, ctrlpts, pt_embed = model(inputs.permute(0, 2, 1))
          
            I_pred = ins_per_point(W_prob)

                                          
            save_dir = os.path.join(result_dir, "Inference-Segmentation")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            test_dataset.color_map_patch_obj(batch_sample_coordinates.cpu().detach().numpy(), 
                                            I_pred.cpu().detach().numpy(), 
                                            save_dir, batch_file_ids)
                        
                    
            
if __name__ == '__main__':
    test()