
from __future__ import annotations
import numpy as np
import torch
import gc
from typing import Tuple
import open3d as o3d
import os
import time
# -----------------------------
# Core utilities
# -----------------------------
def _array2samples_distance(array1: torch.Tensor, array2: torch.Tensor) -> torch.Tensor:
    """Average nearest neighbor distance from array1 to array2."""
    norms1 = torch.sum(array1**2, dim=1, keepdim=True)   # (N,1)
    norms2 = torch.sum(array2**2, dim=1, keepdim=True)   # (M,1)
    dists = norms1 + norms2.T - 2 * array1 @ array2.T    # (N,M)
    min_dists, _ = torch.min(dists, dim=1)               # (N,)
    min_dists = torch.clamp(min_dists, min=0.0)
    return torch.mean(torch.sqrt(min_dists))

# def _chamfer_distance_tensor(a: np.ndarray, b: np.ndarray) -> float:
#     """Chamfer distance (CPU-only) using torch tensors. Returns ×100 for scaling consistency."""
#     with torch.no_grad():
#         A = torch.from_numpy(a).float()
#         B = torch.from_numpy(b).float()
#         norms1 = torch.sum(A ** 2, dim=1, keepdim=True)
#         norms2 = torch.sum(B ** 2, dim=1, keepdim=True)
#         dists = norms1 + norms2.T - 2 * A @ B.T
#         dists = torch.clamp(dists, min=0.0)

#         min_dists1 = torch.min(dists, dim=1)[0]
#         min_dists2 = torch.min(dists, dim=0)[0]

#         chamfer = 0.5 * (torch.mean(torch.sqrt(min_dists1)) + torch.mean(torch.sqrt(min_dists2)))
#     return float(chamfer.item() * 100.0)
def _chamfer_distance_tensor(a: np.ndarray, b: np.ndarray, max_points: int = 5000) -> float:
    """directed Chamfer distance, due to the characteristics of the real3d-ad dataset"""
    if a.shape[0] > max_points:
        idx = np.random.choice(a.shape[0], max_points, replace=False)
        a = a[idx]
    if b.shape[0] > max_points:
        idx = np.random.choice(b.shape[0], max_points, replace=False)
        b = b[idx]

    with torch.no_grad():
        A = torch.from_numpy(a).float()
        B = torch.from_numpy(b).float()

        norms1 = torch.sum(A ** 2, dim=1, keepdim=True)
        norms2 = torch.sum(B ** 2, dim=1, keepdim=True)
        dists = norms1 + norms2.T - 2 * (A @ B.T)
        dists = torch.clamp(dists, min=0.0)

        min_dists1 = torch.min(dists, dim=1)[0]
        # min_dists2 = torch.min(dists, dim=0)[0]

        # chamfer = 0.5 * (torch.mean(torch.sqrt(min_dists1)) + torch.mean(torch.sqrt(min_dists2)))
        chamfer = torch.mean(torch.sqrt(min_dists1))
    return float(chamfer.item() * 100.0)

def _preprocess_point_cloud(pcd: o3d.geometry.PointCloud, voxel_size: float):
    """Downsample, estimate normals, and compute FPFH features."""
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2.0
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    radius_feature = voxel_size * 5.0
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    return pcd_down, fpfh


def _execute_global_registration(
    source_down, target_down, source_fpfh, target_fpfh, voxel_size: float
):
    """RANSAC-based coarse alignment on FPFH features."""
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        True,  # mutual filter
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    return result


def _refine_registration_with_icp(
    source, target, voxel_size: float, initial_transformation: np.ndarray
):
    """Point-to-point ICP refinement with a tight threshold."""
    distance_threshold = voxel_size * 0.4
    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        distance_threshold,
        initial_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
    )
    return result

def _to_pointcloud(obj: object) -> o3d.geometry.PointCloud:
    """Convert ndarray / torch.Tensor / PointCloud to Open3D PointCloud (float64)."""
    if isinstance(obj, o3d.geometry.PointCloud):
        return o3d.geometry.PointCloud(obj)

    if isinstance(obj, torch.Tensor):
        obj = obj.detach().cpu().numpy()

    if isinstance(obj, np.ndarray):
        if obj.ndim != 2 or obj.shape[1] != 3:
            raise ValueError(f"Expected ndarray of shape (N,3), got {obj.shape}")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obj.astype(np.float64, copy=False))
        return pcd

    raise TypeError(f"Unsupported type for point cloud: {type(obj)}")
def save_point_cloud_txt(pcd: o3d.geometry.PointCloud, filename: str):
    """Save Open3D point cloud as txt file (x y z per line)."""
    np.savetxt(filename, np.asarray(pcd.points), fmt="%.6f")

def load_mesh_and_convert_to_point_cloud(file_path):
    mesh = o3d.io.read_triangle_mesh(file_path)
    if not mesh.has_vertices():
        print(f"Failed to load mesh from {file_path}")
        return None
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = mesh.vertices
    return point_cloud


def register_point_clouds(
    source,
    target,
    voxel_size,
    cd_threshold = 1.6,
    iteration_count = 1,
    max_iterations = 10,

) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud, float]:
    """
    Register `source` to `target` with coarse-to-fine alignment and iterative stopping by Chamfer distance.

    Returns (registered_source, target_copy, loss).
    """
    src = _to_pointcloud(source)
    tgt = _to_pointcloud(target)

    src_down, src_fpfh = _preprocess_point_cloud(src, voxel_size)
    tgt_down, tgt_fpfh = _preprocess_point_cloud(tgt, voxel_size)

    ransac_result = _execute_global_registration(src_down, tgt_down, src_fpfh, tgt_fpfh, voxel_size)
    
    icp_result = _refine_registration_with_icp(src, tgt, voxel_size, ransac_result.transformation)

    src.transform(icp_result.transformation)
    src_down.transform(icp_result.transformation)

    src_pts = np.asarray(src_down.points)
    tgt_pts = np.asarray(tgt_down.points)

    if src_pts.size == 0 or tgt_pts.size == 0:
        cd_loss = float("inf")
    else:
        cd_loss = _chamfer_distance_tensor(src_pts, tgt_pts)

    # Recursive refinement if loss too high
    if cd_loss > cd_threshold and iteration_count < max_iterations:
        next_threshold = cd_threshold + 0.1 
        return register_point_clouds(
            src, tgt, voxel_size,
            cd_threshold=next_threshold,
            iteration_count=iteration_count + 1,
            max_iterations=max_iterations,
        )
    return src, tgt, float(cd_loss)