import json
import random
import numpy as np
from geomdl import NURBS
from geomdl import utilities
from geomdl import multi
from geomdl import exchange

def write_points_with_normals(points, normals, filepath):
    '''
    points:  N*3; normals: N*3; filepath:str
    '''
    num_pts = points.shape[0]
    assert points.shape == normals.shape
        
    with open(filepath, 'a') as fp:
        for i in range(num_pts):
            s = '{} {} {} {} {} {}\n'.format(points[i, 0], points[i, 1], points[i, 2], 
                                               normals[i, 0], normals[i, 1], normals[i, 2])
            fp.write(s)
            
def write_points_with_normals_batch(points, normals, filepaths):
    '''
    mesh_v:  B*N*3; color: B*N*3; filepaths:N
    '''
    batch_size = points.shape[0]
    assert batch_size == len(normals)
    
    for b in range(batch_size):
        write_points_with_normals(points[b, :, :], normals[b, :, :], filepaths[b])

def write_colored_points_obj(mesh_v, colors, filepath):
    '''
    mesh_v:  N*3; color: N*3; filepath:str
    '''
    num_pts = mesh_v.shape[0]
    assert num_pts == len(colors)
    
    with open(filepath, 'a') as fp:
        for i in range(num_pts):
            s = 'v {} {} {} {} {} {}\n'.format(mesh_v[i, 0], mesh_v[i, 1], mesh_v[i, 2], 
                                               colors[i, 0], colors[i, 1], colors[i, 2])
            fp.write(s)
            
def write_colored_points_obj_batch(mesh_v, colors, filepaths):
    '''
    mesh_v:  B*N*3; color: B*N*3; filepaths:N
    '''
    batch_size = mesh_v.shape[0]
    assert batch_size == len(colors)
    
    for b in range(batch_size):
        write_colored_points_obj(mesh_v[b, :, :], colors[b, :, :], filepaths[b])
        
def write_colored_points_with_labels_obj(mesh_v, colors, ids, filepath):
    '''
    mesh_v:  N*3; color: N*3; ids: N; filepath:str
    '''
    num_pts = mesh_v.shape[0]
    assert num_pts == len(colors)
    
    with open(filepath, 'a') as fp:
        for i in range(num_pts):
            s = 'v {} {} {} {} {} {}  #{}\n'.format(mesh_v[i, 0], mesh_v[i, 1], mesh_v[i, 2], 
                                               colors[i, 0], colors[i, 1], colors[i, 2],
                                               ids[i])
            fp.write(s)
            
def write_colored_points_with_labels_obj_batch(mesh_v, colors, ids, filepaths):
    '''
    mesh_v:  B*N*3; color: B*N*3; ids: B*N; filepaths:N
    '''
    batch_size = mesh_v.shape[0]
    assert batch_size == len(colors)
    
    for b in range(batch_size):
        write_colored_points_with_labels_obj(mesh_v[b, :, :], colors[b, :, :], ids[b], filepaths[b])
  
def export_color_map_from_config(cofig_file_path):
    with open(cofig_file_path) as f:
        cmap = json.load(f)
        return cmap
    
def get_colors_per_point(cmap, batch_ids):
    '''
    cmap is a list
    '''    
    cmap = np.array(cmap)
    colors = cmap[batch_ids]
    return colors

def write_geomdl_json(decode_degree_dict, I_pred_one_sample, I_deg_one_sample, 
                      ctrlpts_one_sample, save_dir, file_path):
    surf_list = []    
    ins_set = I_pred_one_sample.unique()
    for ins in ins_set:
        deg = I_deg_one_sample[ins]
        deg_u, deg_v = decode_degree_dict[deg.item()]
        num_ctrlpts_u, num_ctrlpts_v = deg_u + 1, deg_v + 1
        #(wx, wy, wz, w)
        ctrlpts_patch = ctrlpts_one_sample[ins, 0:num_ctrlpts_u, 0:num_ctrlpts_v, :]
        
        surf = NURBS.Surface()
        # Set degrees
        surf.degree_u = deg_u
        surf.degree_v = deg_v
        # Set control points
        surf.ctrlpts2d = ctrlpts_patch.tolist()
        # Set knot vectors
        surf.knotvector_u = utilities.generate_knot_vector(surf.degree_u, surf.ctrlpts_size_u)
        surf.knotvector_v = utilities.generate_knot_vector(surf.degree_v, surf.ctrlpts_size_v)
        surf_list.append(surf)

    try:
        multi_surf = multi.SurfaceContainer(surf_list)
        exchange.export_json(multi_surf, file_path)
    except:
        #import ipdb;
        #ipdb.set_trace()
        pass

def write_geomdl_json_batch(decode_degree_dict, I_pred, I_deg_pred, ctrlpts, save_dir, file_paths):
    B, N = I_pred.shape
    assert B == I_deg_pred.shape[0]
    assert B == ctrlpts.shape[0]
    
    Kmax = I_deg_pred.shape[1]
    assert Kmax == ctrlpts.shape[1]
    
    for b in range(B):
        file_path = file_paths[b]
        I_pred_one_sample = I_pred[b, ...]
        I_deg_pred_one_sample = I_deg_pred[b, :]
        ctrlpts_one_sample = ctrlpts[b, ...]
        write_geomdl_json(decode_degree_dict, I_pred_one_sample, 
                          I_deg_pred_one_sample, ctrlpts_one_sample, save_dir, file_path)
        
    
