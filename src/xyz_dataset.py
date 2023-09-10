import os
import numpy as np
import sys

import torch
from torch.utils.data import Dataset
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from geomdl import BSpline
from geomdl import utilities
from geomdl import exchange

from src.data_io import *

degree_color_map = plt.get_cmap('Set1')

"""
Read point clouds with normals
Format:
x0 y0 z0 nx0 ny0 nz0
x1 y1 z1 nx1 ny1 nz1
x2 y2 z2 nx2 ny2 nz2
...
"""
class XYZDataset(Dataset):
    def __init__(self, data_dir, has_normals=True, max_deg_u=3, max_deg_v=3, npoints=8192, batch_size=32):
        self.data_dir = data_dir
        self.has_normals = has_normals
        self.npoints = npoints
        self.file_paths = [os.path.join(self.data_dir, file_name) for file_name in os.listdir(self.data_dir)]
        self.max_deg_u = max_deg_u
        self.max_deg_v = max_deg_v
        self.encode_degree_dict = {}
        self.decode_degree_dict = {}
        for i in range(1, max_deg_u + 1):
            for j in range(1, max_deg_v + 1):
                self.encode_degree_dict[(i, j)] = (i - 1) * max_deg_u + (j - 1)
                self.decode_degree_dict[(i - 1) * max_deg_u + (j - 1)] = (i, j)
                  
    def __getitem__(self, index):
        
        file_path = self.file_paths[index]
        
        if not self.has_normals:
            df = pd.read_csv(file_path, delimiter="\s+", names=["x", "y", "z"])
            xyz = df[['x', 'y', 'z']].to_numpy(dtype=np.float32)
        else:
            df = pd.read_csv(file_path, delimiter="\s+", names=["x", "y", "z", "nx", "ny", "nz"])
            xyz = df[['x', 'y', 'z']].to_numpy(dtype=np.float32)
            normal = df[['nx', 'ny', 'nz']].to_numpy(dtype=np.float32)
        
        filename = os.path.basename(file_path)
        file_id = os.path.splitext(filename)[0]
        
        if not self.has_normals:
            return xyz, file_id
        else:
            return xyz, normal, file_id
        
    def __len__(self):
        return len(self.file_paths)  
    
    def normalize_xyz(self, sample_coordinates):
        '''
        sample_coordinates: B*N*3
        '''
        B, N, _ = sample_coordinates.shape
        
        centroid = torch.mean(sample_coordinates, dim=1)
        centroid = centroid.unsqueeze(1).expand(B, N, 3)
        sample_coordinates = sample_coordinates - centroid
        
        m = torch.sqrt(torch.sum(sample_coordinates**2, dim=2))
        m = torch.max(m, dim=1)[0]
        m = m.unsqueeze(1).unsqueeze(2).expand(B, N, 3)
        
        sample_coordinates = sample_coordinates / m
                        
        return sample_coordinates    
    
    def normalize_normal(self, sample_normals):
        '''
        sample_normals: B*N*3
        '''
        B, N, _ = sample_normals.shape
        
        m = torch.sqrt(torch.sum(sample_normals**2, dim=2))
        m = m.unsqueeze(2).expand(B, N, 3)
        
        sample_normals = sample_normals / m
                        
        return sample_normals 
    
    def normalize(self, sample_coordinates, sample_normals):
        return self.normalize_xyz(sample_coordinates), self.normalize_normal(sample_normals)
    
    def color_map_degree_obj(self, sample_coordinates, degree_labels, save_dir, file_ids):
        # we should convert tensor to numpy before calling
        assert isinstance(sample_coordinates, np.ndarray)
        assert isinstance(degree_labels, np.ndarray)
        
        colors = degree_color_map(degree_labels) 
        if (len(sample_coordinates.shape) == 3):
            file_paths = [os.path.join(save_dir, file_id + '.obj') for file_id in file_ids]
            #write_colored_points_obj_batch(sample_coordinates, colors, file_paths)
            write_colored_points_with_labels_obj_batch(sample_coordinates, colors, degree_labels, file_paths)
        elif (len(sample_coordinates.shape) == 2):
            file_id = file_ids # just one sample
            file_path = os.path.join(save_dir, file_id + '.obj')
            #write_colored_points_obj_batch(mesh_v, colors, file_path)
            write_colored_points_with_labels_obj(sample_coordinates, colors, degree_labels, file_path)
        else:
            print("input error!") 
    
    def color_map_patch_obj(self, sample_coordinates, patch_labels, save_dir, file_ids):
        # we should convert tensor to numpy before calling
        assert isinstance(sample_coordinates, np.ndarray)
        assert isinstance(patch_labels, np.ndarray)
        
        config_file_path = "./configs/part_color_mapping.json"
        patch_cmp = export_color_map_from_config(config_file_path)
        colors = get_colors_per_point(patch_cmp, patch_labels)
        if (len(sample_coordinates.shape) == 3):
            file_paths = [os.path.join(save_dir, file_id + '.obj') for file_id in file_ids]
            #write_colored_points_obj_batch(sample_coordinates, colors, file_paths)
            write_colored_points_with_labels_obj_batch(sample_coordinates, colors, patch_labels, file_paths)
        elif (len(sample_coordinates.shape) == 2):
            file_id = file_ids # just one sample
            file_path = os.path.join(save_dir, file_id + '.obj')
            #write_colored_points_obj_batch(mesh_v, colors, file_path)
            write_colored_points_with_labels_obj(sample_coordinates, colors, patch_labels, file_path)
        else:
            print("input error!") 
            
    def write_geomdl_json(self, I_pred, I_deg_pred, ctrlpts, save_dir, file_ids):
        "ctrlpts should be (wx, wy, wz, w)"
        if (len(I_deg_pred.shape) == 2):
            file_paths = [os.path.join(save_dir, file_id + '.json') for file_id in file_ids]
            ctrlpts_list = ctrlpts
            write_geomdl_json_batch(self.decode_degree_dict, I_pred, I_deg_pred, ctrlpts_list, save_dir, file_paths)
        elif (len(I_deg_pred.shape) == 1):
            file_id = file_ids # just one sample
            file_path = os.path.join(save_dir, file_id + '.json')
            ctrlpts_sample = ctrlpts
            write_geomdl_json(self.decode_degree_dict, I_pred, I_deg, ctrlpts_sample, save_dir, file_path)
        else:
            print("input error!") 

   
    