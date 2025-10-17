# -*- coding: utf-8 -*-
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, "../../PASDF")))
sys.path = [path for path in sys.path if 'liwq' not in path and 'zbz' not in path]
import glob
import pathlib
import numpy as np
import open3d as o3d
import time
import torch
from torch.utils.data import Dataset

class Dataset_Real3D_AD_test(Dataset):
    def __init__(self, dataset_dir: str, cls_name: str, num_points: int = 0, 
                 normalize: bool = True, scale_factor: float = 17.744022369384766, template_path: str = None):
        self.dataset_dir = dataset_dir
        self.cls_name = cls_name
        self.num_points = num_points
        self.normalize = normalize
        self.scale_factor = scale_factor

        self.template_path = os.path.join(current_dir, f"../{template_path}")
        self.template_path_cls = None
        self.template_pcd = self._get_template_pcd()

        test_dir = os.path.join(dataset_dir, cls_name, "test")
        all_pcds = glob.glob(os.path.join(test_dir, "*.pcd"))
        self.test_sample_list = [s for s in all_pcds if "temp" not in s]
        self.gt_dir = os.path.join(dataset_dir, cls_name, "gt")

        if len(self.test_sample_list) == 0:
            raise FileNotFoundError(f"No test samples found in {test_dir}")

    def _get_template_pcd(self):
        cls_path = os.path.join(self.template_path, self.cls_name)
        for f in os.listdir(cls_path):
            if f.endswith((".ply", ".obj")):
                self.template_path_cls = os.path.join(cls_path, f)
                break
        template_pcd = self._read_mesh(self.template_path_cls)     
        return template_pcd

    def _normalize(self, point_cloud):
        center = np.average(point_cloud, axis=0)
        point_cloud = point_cloud - np.expand_dims(center,axis=0)
        point_cloud = point_cloud / self.scale_factor
        return point_cloud

    def _read_mesh(self,file_path):
        mesh = o3d.io.read_triangle_mesh(file_path)
        if not mesh.has_vertices():
            print(f"Failed to load mesh from {file_path}")
            return None
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = mesh.vertices
        return point_cloud

    def _read_pcd(self, path):
        pcd = o3d.io.read_point_cloud(path)
        pts = np.array(pcd.points, dtype=np.float32)
        pts = np.unique(pts, axis=0) 
        return pts


    def __len__(self):
        return len(self.test_sample_list)
      
    def __getitem__(self, idx):
        sample_path = self.test_sample_list[idx]
        # print(sample_path)
        if "good" in sample_path:

            points = o3d.io.read_point_cloud(sample_path)
            points = np.array(points.points)
            mask = np.zeros((points.shape[0],), dtype=np.float32)
            label = 0
        else:
            filename = pathlib.Path(sample_path).stem
            txt_path = os.path.join(self.gt_dir, filename + ".txt")
            data = np.loadtxt(txt_path)  
            points = data[:, :3].astype(np.float32)
            mask = data[:, 3].astype(np.float32)
            label = 1
   
        if self.normalize:
            points = self._normalize(points)

        if self.num_points and points.shape[0] > self.num_points:
                choice = np.random.choice(points.shape[0], self.num_points, replace=False)
                points = points[choice]
                mask = mask[choice] if mask.shape[0] == len(choice) else mask[:points.shape[0]]

        points_pc = points

        points_t = np.asarray(points_pc)
        mask_t = torch.from_numpy(mask).float()
        label_t = torch.tensor(label, dtype=torch.long) 
        template_points_t = np.asarray(self.template_pcd.points, dtype=np.float32)
        return {"points": points_t, "mask": mask_t, "label": label_t, "template_points": template_points_t}
