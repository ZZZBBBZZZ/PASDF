import numpy as np
import torch
from torch.utils.data import Dataset
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, "../../PASDF")))
sys.path = [path for path in sys.path if 'liwq' not in path and 'zbz' not in path]
import results
import trimesh
import pysdf
results_path = os.path.join(current_dir, "../results")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def combine_sample_latent(samples, latent_class):
#     """Combine each sample (x, y, z) with the latent code generated for this object.
#     Args:
#         samples: collected points, np.array of shape (N, 3)
#         latent: randomly generated latent code, np.array of shape (1, args.latent_size)
#     Returns:
#         combined hstacked latent code and samples, np.array of shape (N, args.latent_size + 3)
#     """
#     latent_class_full = np.tile(latent_class, (samples.shape[0], 1))   # repeat the latent code N times for stacking
#     return np.hstack((latent_class_full, samples))

# def farthest_point_sample(point, npoint):
#     """
#     Input:
#         xyz: pointcloud data, [N, D]
#         npoint: number of samples
#     Return:
#         centroids: sampled pointcloud index, [npoint, D]
#     """
#     N, D = point.shape
#     xyz = point[:,:3]
#     centroids = np.zeros((npoint,))
#     distance = np.ones((N,)) * 1e10
#     farthest = np.random.randint(0, N)
#     for i in range(npoint):
#         centroids[i] = farthest
#         centroid = xyz[farthest, :]
#         dist = np.sum((xyz - centroid) ** 2, -1)
#         mask = dist < distance
#         distance[mask] = dist[mask]
#         farthest = np.argmax(distance, -1)
#     point = point[centroids.astype(np.int32)]
#     return point

# def pc_normalize(pc):
#     centroid = np.mean(pc, axis=0)
#     pc = pc - centroid
#     m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
#     pc = pc / m
#     return pc,m,centroid

# class SDFDataset(Dataset):
#     """
#     TODO: adapting to handle multiple objects
#     """
#     def __init__(self, dataset_name):
#         samples_dict = np.load(os.path.join(os.path.dirname(results.__file__), f'samples_dict_{dataset_name}.npy'), allow_pickle=True).item()
#         # water_path = '/home/lc/Desktop/liwq/datasets/Real3D-AD-OBJ/airplane/128_template.obj'
#         watertight_mesh = trimesh.load(water_path,
#                                        force="mesh",
#                                        process=True)
#         f = pysdf.SDF(watertight_mesh.vertices, watertight_mesh.faces)
#         surface = farthest_point_sample(np.array(watertight_mesh.vertices),10000)
#         surface_label = f(surface)
#         v_min, v_max = surface.min(0), surface.max(0)
#         p_bbox = np.random.uniform(low=[v_min[0], v_min[1], v_min[2]], high=[v_max[0], v_max[1], v_max[2]], size=(10000, 3))
#         box_label = f(p_bbox)
#         near_points,m,center = pc_normalize(surface)
#         box = box-center
#         box = box/m
#         p_total = np.vstack((near_points, box))
#         sdf = np.vstack(surface_label,box_label)
#         self.sdf = sdf
#         self.latent_class = combine_sample_latent(p_total, np.array([0], dtype=np.int32))
#         # self.data = dict()
#         # for obj_idx in list(samples_dict.keys()):  # samples_dict.keys() for all the objects
#         #     for key in samples_dict[obj_idx].keys():   # keys are ['samples', 'sdf', 'latent_class', 'samples_latent_class']
#         #         value = torch.from_numpy(samples_dict[obj_idx][key]).float().to(device)
#         #         if len(value.shape) == 1:    # increase dim if monodimensional, needed to vstack
#         #             value = value.view(-1, 1)
#         #         if key not in list(self.data.keys()):
#         #             self.data[key] = value
#         #         else:
#         #             self.data[key] = torch.vstack((self.data[key], value))
#         # return

#     def __len__(self):
#         # return self.data['sdf'].shape[0]
#         return self.sdf.shape[0]

#     def __getitem__(self, idx):
#         # latent_class = self.data['samples_latent_class'][idx, :]
#         # sdf = self.data['sdf'][idx]
#         latent_class = self.latent_class[idx]
#         sdf = self.sdf[idx]
#         return latent_class, sdf

class SDFDataset(Dataset):
    """
    TODO: adapting to handle multiple objects
    """
    def __init__(self, dataset_name,class_idx):
        samples_dict = np.load(os.path.join(results_path, dataset_name,f'samples_dict_{dataset_name}.npy'), allow_pickle=True).item()
        self.data = dict()
        for obj_idx in list(samples_dict.keys()):  # samples_dict.keys() for all the objects
            if obj_idx == class_idx:
                for key in samples_dict[obj_idx].keys():   # keys are ['samples', 'sdf', 'latent_class', 'samples_latent_class']
                    value = torch.from_numpy(samples_dict[obj_idx][key]).float().to(device)
                    if len(value.shape) == 1:    # increase dim if monodimensional, needed to vstack
                        value = value.view(-1, 1)
                    if key not in list(self.data.keys()):
                        self.data[key] = value
                    else:
                        self.data[key] = torch.vstack((self.data[key], value))
        return

    def __len__(self):
        return self.data['sdf'].shape[0]

    def __getitem__(self, idx):
        latent_class = self.data['samples_latent_class'][idx, :]
        sdf = self.data['sdf'][idx]
        return latent_class, sdf

if __name__=='__main__':
    dataset_name = "ShapeNetCore"
    dataset = SDFDataset(dataset_name)
