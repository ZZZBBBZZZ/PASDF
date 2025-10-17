import sys
import os
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.abspath(os.path.join(current_dir, "../../PASDF")))

sys.path = [path for path in sys.path if 'liwq' not in path and 'zbz' not in path]

from utils import utils_mesh
import point_cloud_utils as pcu
from glob import glob
from datetime import datetime
import yaml
import pybullet as pb
import trimesh
import gc

"""
For each object, sample points and store their distance to the nearest triangle.
"""


dataset_path = os.path.join(current_dir, "Real3D_AD")
results_path = os.path.join(current_dir, "../results")
config_files_path = os.path.join(current_dir, "../config_files")
print(dataset_path)
def combine_sample_latent(samples, latent_class):
    latent_class_full = np.tile(latent_class, (samples.shape[0], 1))
    return np.hstack((latent_class_full, samples))

def main(cfg):
    obj_paths = sorted(glob(os.path.join(dataset_path, '*', '*normalized.ply')))
    samples_dict = dict()
    idx_str2int_dict = dict()
    idx_int2str_dict = dict()

    for obj_idx, obj_path in enumerate(obj_paths):
        print(obj_idx, obj_path)
        obj_idx_str = os.sep.join(obj_path.split(os.sep)[-2:-1])
        idx_str2int_dict[obj_idx_str] = obj_idx
        idx_int2str_dict[obj_idx] = obj_idx_str
        samples_dict[obj_idx] = dict()

        try:
            verts, faces = pcu.load_mesh_vf(obj_path)
            mesh_original = utils_mesh._as_mesh(trimesh.load(obj_path))

            if not mesh_original.is_volume or not mesh_original.is_watertight:
                print(f"Skipping mesh due to invalid geometry: {obj_path}")
            if not mesh_original.is_watertight:
                verts, faces = pcu.make_mesh_watertight(mesh_original.vertices, mesh_original.faces, 50000)
                mesh_original = trimesh.Trimesh(vertices=verts, faces=faces)

        except Exception as e:
            print(e)
            continue

        verts = np.array(mesh_original.vertices)
        faces = faces.astype(np.int64)
 

        p_vol = np.random.rand(cfg['num_samples_in_volume'], 3) * 2 - 1
        v_min, v_max = verts.min(0), verts.max(0)
        p_bbox = np.random.uniform(low=v_min, high=v_max, size=(cfg['num_samples_in_bbox'], 3))
 

        fid_surf, bc_surf = pcu.sample_mesh_random(verts, faces, cfg['num_samples_on_surface'])
 
        p_surf = pcu.interpolate_barycentric_coords(faces, fid_surf, bc_surf, verts)
        p_total = np.vstack((p_vol, p_bbox, p_surf))

        # print("Before signed_distance_to_mesh")
        sdf, _, _ = pcu.signed_distance_to_mesh(p_total, verts, faces)
        # print("After signed_distance_to_mesh")

        samples_dict[obj_idx]['sdf'] = sdf
        samples_dict[obj_idx]['samples_latent_class'] = combine_sample_latent(p_total, np.array([obj_idx], dtype=np.int32))

        del verts, faces, mesh_original, p_total, sdf
        gc.collect()

    os.makedirs(os.path.join(results_path, "Real3D_AD"), exist_ok=True)
    np.save(os.path.join(results_path, "Real3D_AD", f'samples_dict_{cfg["dataset"]}.npy'), samples_dict)
    np.save(os.path.join(results_path, "Real3D_AD", f'idx_str2int_dict.npy'), idx_str2int_dict)
    np.save(os.path.join(results_path, "Real3D_AD", f'idx_int2str_dict.npy'), idx_int2str_dict)

if __name__ == '__main__':
    cfg_path = os.path.join(config_files_path, 'extract_sdf.yaml')
    with open(cfg_path, 'rb') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    main(cfg)


