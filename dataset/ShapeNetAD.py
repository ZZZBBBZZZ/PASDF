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
import pybullet as pb
class Dataset_ShapeNetAD_test(Dataset):
    def __init__(self, dataset_dir: str, cls_name: str, num_points: int = 0, 
                 normalize: bool = True, scale_factor: float = 1.0, template_path: str = None):
        self.dataset_dir = dataset_dir
        self.cls_name = cls_name
        self.num_points = num_points
        self.normalize = normalize
        self.scale_factor = float(scale_factor)


        self.template_path = os.path.join(current_dir, f"../{template_path}") if template_path else None
        self.template_path_cls = None 
        self.template_pcd = self._get_template_pcd()  

 
        test_dir = os.path.join(dataset_dir, cls_name, "test")
        all_pcds = glob.glob(os.path.join(test_dir, "**/*.pcd"), recursive=True)

        self.test_sample_list = [s for s in all_pcds if "template" not in os.path.basename(s)]

        self.gt_dir = os.path.join(dataset_dir, cls_name, "GT")

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
        if not self.normalize:
            return point_cloud
            
        center = np.average(point_cloud, axis=0)
        point_cloud = point_cloud - np.expand_dims(center, axis=0)
        point_cloud = point_cloud / self.scale_factor
        return point_cloud

    def _read_pcd(self, path):
        pcd = o3d.io.read_point_cloud(path)
        pts = np.array(pcd.points, dtype=np.float32)
        pts = np.unique(pts, axis=0)  
        return pts

    def _read_mesh(self, file_path):
        mesh = o3d.io.read_triangle_mesh(file_path)
        if not mesh.has_vertices():
            print(f"Failed to load mesh from {file_path}")
            return None
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = mesh.vertices
        return point_cloud

    def rotate_pointcloud(self, pointcloud_A, rpy_BA=[np.pi / 2, 0, 0]):
        rot_Q = pb.getQuaternionFromEuler(rpy_BA)
        rot_M = np.array(pb.getMatrixFromQuaternion(rot_Q)).reshape(3, 3)
        pointcloud_B = np.einsum('ij,kj->ki', rot_M, pointcloud_A)
        return pointcloud_B

    def shapenet_rotate(self, point_cloud):
        '''In Shapenet, the front is the -Z axis with +Y still being the up axis. This function rotates the object to align with the canonical reference frame.
        '''
        if isinstance(point_cloud, o3d.geometry.PointCloud):
            verts_original = np.array(point_cloud.points)
        else:
            verts_original = point_cloud
            
        verts = self.rotate_pointcloud(verts_original, [np.pi/2, 0, -np.pi/2])
        return verts
    # def _reorder_labels(self,label_path, anomaly_path_pcd):
    #     """
    #     重新排序标签以匹配点云中的点顺序。

    #     :param label_path: 标签文件的路径，文件应为 CSV 格式，标签在最后一列。
    #     :param anomaly_path_pcd: 点云文件的路径。
    #     :return: 重新排序后的标签数组。
    #     """
    #     if "vase" in label_path:
    #         num_round = 7
    #     else:
    #         num_round = 6

    #     # 加载点云
    #     # pcd = o3d.io.read_point_cloud(anomaly_path_pcd)
    #     pcd_points = anomaly_path_pcd
    #     # print(pcd_points)
    #     pcd_points_rounded = np.round(pcd_points, num_round)  # 四舍五入到六位小数

    #     # 加载标签
    #     try:
    #         label_data = np.loadtxt(label_path, delimiter=' ')  # 假设标签文件是 CSV 格式
    #     except ValueError as e:
    #         print(f"Error loading label file {label_path}: {e}")
    #         return np.array([])  # 或者抛出异常

    #     # 加载标签
    #     # label_data = np.loadtxt(label_path, delimiter=',')
    #     label_points = label_data[:, :-1]  # 假设标签前面的列是坐标
    #     labels = label_data[:, -1]  # 假设标签在最后一列

    #     # label_points_rounded = arrayround(label_points, num_round)
    #     label_points_rounded = np.round(label_points, num_round)  # 四舍五入到六位小数



    #     # 创建一个查找表，以点云点为键，标签为值
    #     label_dict = {tuple(point): label for point, label in zip(label_points_rounded, labels)}

    #     # 根据点云中的点顺序重新排序标签
    #     reordered_labels = []
    #     for idx, point in enumerate(pcd_points_rounded):
            
    #         point_tuple = tuple(point)
    #         if point_tuple in label_dict:
    #             reordered_labels.append(label_dict[point_tuple])
    #         else:
    #             # 这里因为四舍五入的原因，会有一些稍微的偏差导致个别点不能完全匹配
    #             # print(label_path)
    #             # print(point)
    #             # print(pcd_points[idx])
    #             # print(point in pcd_points_rounded)
    #             # print(point in label_points_rounded)
    #             # print("***label有问题****")
    #             # with open('label_dict.txt', 'w') as file:
    #             #     for key, value in label_dict.items():
    #             #         file.write(f"{key}: {value}\n")
    #             # # 保存数组到文本文件，使用普通的四舍五入保留八位小数
    #             # np.savetxt(' pcd_points_rounded.txt', pcd_points_rounded, fmt='%.10f', delimiter=',')
    #             # np.savetxt(' label_points_rounded.txt', label_points_rounded, fmt='%.10f', delimiter=',')
    #             distances = np.linalg.norm(label_points_rounded - point, axis=1)
    #             closest_index = np.argmin(distances)
    #             closest_label = labels[closest_index]

    #             reordered_labels.append(closest_label)
    #             # print(closest_label)
            
    #             # print("当前点:", point)
    #             # print("最近点:", label_points_rounded[closest_index])
    #     return np.array(reordered_labels)
    
    def _reorder_labels(self, label_path: str, pcd_points_rot: np.ndarray) -> np.ndarray:
        if pcd_points_rot.size == 0:
            return np.array([], dtype=np.float32)

        num_round = 7 if "vase" in (label_path or "") else 6
        scale = 10 ** num_round

        try:
            label_data = np.loadtxt(label_path, delimiter=' ')
        except ValueError as e:
            print(f"Error loading label file {label_path}: {e}")
            return np.zeros((pcd_points_rot.shape[0],), dtype=np.float32)


        lbl_pts = label_data[:, :-1]
        labels = label_data[:, -1].astype(np.float32)

        pcd_r = np.round(pcd_points_rot, num_round)
        lbl_r = np.round(lbl_pts, num_round)

        pcd_i = (pcd_r * scale).astype(np.int64)
        lbl_i = (lbl_r * scale).astype(np.int64)

        key_to_idx = {tuple(row): i for i, row in enumerate(lbl_i)}

        N = pcd_i.shape[0]
        out = np.empty((N,), dtype=np.float32)
        hit = np.zeros((N,), dtype=bool)
        for k, key in enumerate(map(tuple, pcd_i)):
            j = key_to_idx.get(key, -1)
            if j >= 0:
                out[k] = labels[j]
                hit[k] = True

        if not np.all(hit):
            miss_idx = np.where(~hit)[0]
            try:
                from scipy.spatial import cKDTree
                tree = cKDTree(lbl_r)
                _, nn = tree.query(pcd_r[miss_idx], k=1, workers=-1)
                out[miss_idx] = labels[nn]
            except Exception:
                kd = o3d.geometry.KDTreeFlann(o3d.utility.Vector3dVector(lbl_r.astype(np.float64)))
                for ii, p in enumerate(pcd_r[miss_idx]):
                    _, idxs, _ = kd.search_knn_vector_3d(p.astype(np.float64), 1)
                    out[miss_idx[ii]] = labels[idxs[0]]
        return out


    def __len__(self) -> int:
        return len(self.test_sample_list)

    def __getitem__(self, idx: int):

        sample_path = self.test_sample_list[idx]
        fname = pathlib.Path(sample_path).stem
        print(sample_path)
        
        points = self._read_pcd(sample_path)

        if "positive" in os.path.basename(sample_path):
            mask = np.zeros((points.shape[0],), dtype=np.float32)
            label = 0
        else:
            txt_path = os.path.join(self.gt_dir, fname + ".txt")
            mask = self._reorder_labels(txt_path, points)
            label = 1

        points = self.shapenet_rotate(points) 
        # It is very important to rotate the template point cloud, because the training data has been flipped
        template_points = self.shapenet_rotate(self.template_pcd) 

        if self.normalize:
            points = self._normalize(points)

        if self.num_points and points.shape[0] > self.num_points:
            choice = np.random.choice(points.shape[0], self.num_points, replace=False)
            points = points[choice]
            mask = mask[choice] if mask.shape[0] == len(choice) else mask[:points.shape[0]]

        points_t = np.asarray(points, dtype=np.float32)
        mask_t = torch.from_numpy(mask.astype(np.float32))
        label_t = torch.tensor(label, dtype=torch.long)
        template_points_t = np.asarray(template_points, dtype=np.float32)
        
        return {
            "points": points_t,
            "mask": mask_t,
            "label": label_t,
            "template_points":template_points_t
        }
    

# def test_batch_loading():
#     from torch.utils.data import DataLoader
    
#     dataset_dir = "/home/dataset/Anomaly-ShapeNet-v2/dataset/16384"
#     cls_name = "bag0"
#     template_path = "/data/ShapeNetAD"
#     dataset = Dataset_ShapeNetAD_test(
#         dataset_dir=dataset_dir,
#         cls_name=cls_name,
#         normalize=True,
#         template_path = template_path,
#     )

#     dataloader = DataLoader(
#         dataset,
#         batch_size=1,
#         shuffle=False,
#         num_workers=0  
#     )
    

#     for i, batch in enumerate(dataloader):
#         print(f"batch {i + 1}:")
#         print(f"  points: {batch['points'].shape}")
#         print(f"  mask: {batch['mask'].shape}")
#         print(f"  label: {batch['label']}")
        


# if __name__ == "__main__":
#     test_batch_loading()

