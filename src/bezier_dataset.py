import os
import json
import random
import numpy as np
import sys

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from geomdl import BSpline
from geomdl import utilities
#from geomdl import multi
from geomdl import exchange

from src.data_io import *




degree_color_map = plt.get_cmap('Set1')


class BezierDataset(Dataset):
    def __init__(self, root, max_deg_u=3, max_deg_v=3, num_max_instances=75, batch_size=32, split='train'):
        self.root = root
        self.split = split
        self.data_dir = os.path.join(self.root, self.split)
        self.file_paths = [os.path.join(self.data_dir, file_name) for file_name in os.listdir(self.data_dir)]
        self.npoints = self.get_npoints(index=0)
        #self.global_max_patch_num, self.global_max_patch_id = self.get_global_max_patch_info()
        self.num_max_instances = num_max_instances
        self.max_deg_u = max_deg_u
        self.max_deg_v = max_deg_v
        self.max_num_ctrlpts_u = max_deg_u + 1
        self.max_num_ctrlpts_v = max_deg_v + 1
        self.batch_size = batch_size
        self.encode_degree_dict = {}
        self.decode_degree_dict = {}
        for i in range(1, max_deg_u + 1):
            for j in range(1, max_deg_v + 1):
                self.encode_degree_dict[(i, j)] = (i - 1) * max_deg_u + (j - 1)
                self.decode_degree_dict[(i - 1) * max_deg_u + (j - 1)] = (i, j)
        
        #assert self.num_max_instances >= self.global_max_patch_num
        #assert self.num_max_instances >= self.global_max_patch_id + 1
              
    def __getitem__(self, index):
        sample_coordinates = np.zeros((self.npoints, 3), dtype=np.float32)
        sample_normals = np.zeros((self.npoints, 3), dtype=np.float32)
        sample_parameters = np.zeros((self.npoints, 2), dtype=np.float32)
        sample_degrees = np.zeros((self.npoints, 2), dtype=np.int)
        sample_patch_ids = np.zeros((self.npoints, ), dtype=np.int)
        patch_degrees = np.full((self.num_max_instances, 2), -1, dtype=np.int)
        # (x, y, z, w)
        control_points = np.zeros((self.num_max_instances, self.max_deg_u+1, self.max_deg_v+1, 4), dtype=np.float32)
        
        file_path = self.file_paths[index]
        
        with open(file_path) as f:
            d = json.load(f)
            n = 0
            surfaces = d["surfaces"]
            patch_num = surfaces["count"]
            bezier_patch_arr = surfaces["data"]
            
            for p in range(patch_num):
                bezier_patch_dict = bezier_patch_arr[p]
                patch_id = bezier_patch_dict["patch_id"]
                # get control points
                patch_control_points_dict = bezier_patch_dict["control_points"]
                patch_degree = patch_control_points_dict["degree"]
                patch_ctrl_pts_data_arr = patch_control_points_dict["data"]
              
                patch_ctrl_pts = np.zeros((self.max_deg_u + 1, self.max_deg_v + 1, 4), dtype=np.float32)
                
                patch_degrees[patch_id, ...] = patch_degree
                
                for i in range(patch_degree[0] + 1):
                    for j in range(patch_degree[1] + 1):
                        patch_coordinates_weight_dict = patch_ctrl_pts_data_arr[i][j]
                        patch_coordinates = patch_coordinates_weight_dict["coordinates"]
                        patch_weight = patch_coordinates_weight_dict["weight"]
                        
                        patch_ctrl_pts[i][j][:] =  \
                                    [patch_coordinates[0], patch_coordinates[1], patch_coordinates[2], patch_weight]    
                        
                control_points[patch_id] = patch_ctrl_pts
                
                # get samples
                patch_samples_dict = bezier_patch_dict["sample_points"]
                patch_samples_num = patch_samples_dict["number"]
                patch_samples_data_arr = patch_samples_dict["data"]
                for i in range(patch_samples_num):
                    pt_coord_norm_para_dict = patch_samples_data_arr[i]
                    pt_coordinates = pt_coord_norm_para_dict["coordinates"]
                    pt_normal = pt_coord_norm_para_dict["normal"]
                    pt_parameters = pt_coord_norm_para_dict["parameters"]

                    # normals are nan if they are in degenerate cases
                    if np.isnan(pt_normal).any():
                        pt_normal = [0.0, 0.0, 0.0] 
                    
                    sample_coordinates[n, ...] = pt_coordinates
                    sample_normals[n, ...] = pt_normal
                    sample_parameters[n, ...] = pt_parameters
                    sample_degrees[n, ...] = patch_degree
                    sample_patch_ids[n] = patch_id
                    n += 1
                    
        filename = os.path.basename(file_path)
        file_id = os.path.splitext(filename)[0]
        
        return sample_coordinates, sample_normals, sample_parameters, \
                sample_degrees, sample_patch_ids, patch_degrees, \
                    control_points, file_id
        
    def __len__(self):
        return len(self.file_paths)
    
    def get_npoints(self, index=0):  
        npoints = 0
        file_path = self.file_paths[index]
        with open(file_path) as f:
            d = json.load(f)
            surfaces = d["surfaces"]
            patch_num = surfaces["count"]
            bezier_patch_arr = surfaces["data"]
            for p in range(patch_num):
                bezier_patch_dict = bezier_patch_arr[p]
                # get samples
                patch_samples_dict = bezier_patch_dict["sample_points"]
                patch_samples_num = patch_samples_dict["number"] 
                npoints += patch_samples_num  
        return npoints  
    
    def get_global_max_patch_info(self):  
        '''
        in this dataset, patch id is not a consistent number,
        this means global_max_patch_id may greater than global_max_patch_num
        we need to find global_max_patch_id for encoding
        ''' 
        global_max_patch_num = 0
        global_max_patch_id = 0
        for file_path in self.file_paths:
            with open(file_path) as f:
                d = json.load(f)
                surfaces = d["surfaces"]
                patch_num = surfaces["count"]
                bezier_patch_arr = surfaces["data"]
                if patch_num > global_max_patch_num:
                    global_max_patch_num = patch_num 
                for p in range(patch_num):
                    bezier_patch_dict = bezier_patch_arr[p]
                    patch_id = bezier_patch_dict["patch_id"]   
                    if patch_id > global_max_patch_id:
                        global_max_patch_id = patch_id    
        return global_max_patch_num, global_max_patch_id
       
    def normalize_one_sample(self, sample_coordinates, ctrlPts, sample_patch_ids, patch_degrees):
        '''
        sample_coordinates: pt_num*3
        ctrlPts: num_max_instances*4*4*4 (x,y,z,w)
        sample_patch_ids: pt_num
        patch_degrees: num_max_instances*2
        '''
        pt_num, _ = sample_coordinates.shape
        num_max_instances, max_ctrlpts_num_u, max_ctrlpts_num_v, _  = ctrlPts.shape
        
        assert pt_num == sample_patch_ids.shape[0]
        assert num_max_instances == patch_degrees.shape[0]
        
        centroid = torch.mean(sample_coordinates, dim=0)
        sample_coordinates = sample_coordinates - centroid
        
        m1 = torch.max(torch.sqrt(torch.sum(sample_coordinates**2, dim=1)))
        
        ctrlPts_coords = ctrlPts[:,:,:,0:3].reshape((num_max_instances*max_ctrlpts_num_u*max_ctrlpts_num_v, 3))
        
        m2 = torch.max(torch.sqrt(torch.sum(ctrlPts_coords**2, dim=1)))
        
        m = torch.max(m1, m2)
        
        sample_coordinates = sample_coordinates / m
        
        selected_patchs = torch.unique(sample_patch_ids)
        for patch_id in selected_patchs:
            deg_u, deg_v = patch_degrees[patch_id]
            ctrlPts[patch_id, 0:(deg_u+1), 0:(deg_v+1), 0:3] = \
                 ctrlPts[patch_id, 0:(deg_u+1), 0:(deg_v+1), 0:3] - centroid
                 
        ctrlPts = ctrlPts / m
                        
        return sample_coordinates, ctrlPts 
    
    def normalize_batch(self, batch_sample_coordinates, batch_ctrlPts, batch_sample_patch_ids, batch_patch_degrees):
        batch_size, pt_num, _ = batch_sample_coordinates.shape
        assert batch_size == batch_ctrlPts.shape[0]
        assert batch_size == batch_sample_patch_ids.shape[0]
        assert pt_num == batch_sample_patch_ids.shape[1]
 
        for b in range(batch_size):
            sample_coordinates = batch_sample_coordinates[b, :]
            ctrlPts = batch_ctrlPts[b, ...]
            sample_patch_ids = batch_sample_patch_ids[b, :]
            patch_degrees = batch_patch_degrees[b, ...]
            batch_sample_coordinates[b], batch_ctrlPts[b] =   \
                self.normalize_one_sample(sample_coordinates, ctrlPts, sample_patch_ids, patch_degrees)

        return batch_sample_coordinates, batch_ctrlPts
    
    def normalize(self, sample_coordinates, ctrlPts, sample_patch_ids, patch_degrees):
        if (len(sample_coordinates.shape) == 3):
            return self.normalize_batch(sample_coordinates, ctrlPts, sample_patch_ids, patch_degrees)
        elif (len(sample_coordinates.shape) == 2):
            return self.normalize_one_sample(sample_coordinates, ctrlPts, sample_patch_ids, patch_degrees)
        else:
            print("input error!")
                          
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
            
    def write_points_with_normals(self, sample_coordinates, sample_normals, save_dir, file_ids):
        assert sample_coordinates.shape == sample_normals.shape
        if (len(sample_coordinates.shape) == 3):
            file_paths = [os.path.join(save_dir, file_id + '.xyz') for file_id in file_ids]  
            write_points_with_normals_batch(sample_coordinates, sample_normals, file_paths)
        elif (len(sample_coordinates.shape) == 2):
            file_id = file_ids # just one sample
            file_path = os.path.join(save_dir, file_id + '.xyz')
            write_points_with_normals(sample_coordinates, sample_normals, file_path)
        else:
            print("input error!") 

    