# -*- coding: utf-8 -*-
import os
import glob
import pathlib
import numpy as np
import open3d as o3d
import torch
from torch.utils.data import Dataset


class Dataset_ShapeNetAD_test(Dataset):
    """
    Real3D-AD 测试集数据读取类

    - good 样本：直接从 test/*.pcd 读取点云
    - 缺陷样本：从 gt/{stem}.txt 读取 [x, y, z, mask]
    - 归一化：减中心并按固定比例缩放（默认 17.744）
    - 支持每类一个 template_path，用于模型配准或对齐
    """

    def __init__(self, dataset_dir: str, cls_name: str, num_points: int = 0,
                 normalize: bool = True, scale: float = 17.744022369384766):
        """
        Args:
            dataset_dir (str): 数据集根目录
            cls_name (str): 类别名称，例如 'car'、'airplane'
            num_points (int): 若 >0 则进行随机采样或FPS采样（此处默认不采样）
            normalize (bool): 是否归一化点云
            scale (float): 归一化缩放系数
        """
        self.dataset_dir = dataset_dir
        self.cls_name = cls_name
        self.num_points = num_points
        self.normalize = normalize
        self.scale = scale

        # 构建 test 文件列表
        test_dir = os.path.join(dataset_dir, cls_name, "test")
        all_pcds = glob.glob(os.path.join(test_dir, "*.pcd"))
        # 排除模板文件
        self.test_sample_list = [s for s in all_pcds if "temp" not in s]
        # 缺陷 gt 文件路径前缀
        self.gt_dir = os.path.join(dataset_dir, cls_name, "gt")

        if len(self.test_sample_list) == 0:
            raise FileNotFoundError(f"No test samples found in {test_dir}")

    # -----------------------
    # 归一化函数（与你的代码保持一致）
    # -----------------------
    def _normalize(self, point_cloud: np.ndarray) -> np.ndarray:
        center = np.average(point_cloud, axis=0)
        point_cloud = point_cloud - center[None, :]
        point_cloud = point_cloud / self.scale
        return point_cloud

    # -----------------------
    # 点云读取
    # -----------------------
    def _read_pcd(self, path: str) -> np.ndarray:
        pcd = o3d.io.read_point_cloud(path)
        pts = np.array(pcd.points, dtype=np.float32)
        pts = np.unique(pts, axis=0)  # 去重
        return pts

    # -----------------------
    # 数据集接口
    # -----------------------
    def __len__(self):
        return len(self.test_sample_list)

    def __getitem__(self, idx):
        sample_path = self.test_sample_list[idx]

        if "good" in sample_path:
            # 正常样本
            points = self._read_pcd(sample_path)
            mask = np.zeros((points.shape[0],), dtype=np.float32)
            label = 0
        else:
            # 缺陷样本，从 GT 文件读取
            filename = pathlib.Path(sample_path).stem
            txt_path = os.path.join(self.gt_dir, filename + ".txt")
            data = np.genfromtxt(txt_path, delimiter=" ")
            points = data[:, :3].astype(np.float32)
            mask = data[:, 3].astype(np.float32)
            label = 1

        if self.normalize:
            points = self._normalize(points)

        # 转为 Tensor
        points_t = torch.from_numpy(points).float()
        mask_t = torch.from_numpy(mask).float()
        label_t = torch.tensor(label, dtype=torch.long)

        return {
            "points": points_t,
            "mask": mask_t,
            "label": label_t,
            "path": sample_path,
            "class": self.cls_name,
        }
