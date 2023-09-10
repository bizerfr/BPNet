import argparse
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Parsing Arguments
parser = argparse.ArgumentParser()
# Experiment Settings
parser.add_argument('--use_gpu', type=int, default=1, help='if use GPU or not')
parser.add_argument('--use_DataParallel', type=int, default=1, help='if use torch.nn.DataParallel or not')
parser.add_argument('--input_normal', type=int, default=1, help='input normals to NN or not')
parser.add_argument('--output_normal', type=int, default=1, help='output normals or not')
parser.add_argument('--use_normal_loss', type=int, default=1, help='use normal loss or not')
parser.add_argument('--num_workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--epochs', type=int, default=150, help='Number of epochs [default: 50]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
parser.add_argument('--max_deg_u', type=int, default=3, help='Maximum degree number in one sample')
parser.add_argument('--max_deg_v', type=int, default=3, help='Maximum degree number in one sample')
parser.add_argument('--num_max_instances', type=int, default=75, help='Maximum patch number in one sample')
parser.add_argument('--test_epoch_frequency', type=int, default=1, help='test frequency of the epoch')
parser.add_argument('--save_obj_frequency', type=int, default=10, help='save frequency of obj file')
parser.add_argument('--result_dir', type=str, default='result', help='result folder')
parser.add_argument('--shuffle_train', type=int, default=1, help='result folder')
parser.add_argument('--shuffle_test', type=int, default=0, help='result folder')
parser.add_argument('--data_path', type=str, 
    default='./data/ABC-Decomposition/release/json_8192', help='data root path')
parser.add_argument('--checkpoint_path', type=str, 
    default='', help='checkpoint_path') 

def build_options():
    args = parser.parse_args()
    return args