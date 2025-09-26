"""FlowRenderer
================

从 STL 气泡模型构建三维气泡场并输出合成数据。

主要流程:
    1. 读取气泡三角网格并执行上采样与体积缩放。
    2. 在给定体积内进行泊松圆盘采样，放置气泡实例并生成整体 STL。
"""

import os
import random
from datetime import datetime
import pyvista as pv
import numpy as np
from tqdm import trange
import multiprocessing as mp
import argparse

try:
    import cupy as cp
except Exception:  # pragma: no cover - CuPy may be unavailable on some machines
    cp = None


def get_array_module(prefer_gpu: bool = True):
    """Return CuPy when available, otherwise fall back to NumPy."""
    if prefer_gpu and cp is not None:
        return cp
    return np


def to_numpy(array):
    """Convert either a NumPy or CuPy array to NumPy."""
    if cp is not None and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return np.asarray(array)

def upsample_and_scale_mesh(stl_files, num_clusters, chosen_volume, sample_spacing):
    """随机挑选 STL 模型并上采样，使其体积匹配目标值。

    参数:
        stl_files: STL 文件路径列表。
        num_clusters: 期望的点云数量上限，用以判断是否需要进一步上采样。
        chosen_volume: 当前气泡分配到的体积 (mm^3)。
        sample_spacing: 上采样的初始采样间距。

    返回:
        stl_file: 选中的 STL 文件路径。
        mesh: 上采样、平滑、填洞后的网格。
        mesh_origin: 与 mesh 同步缩放的原始网格。
        chosen_volume: 输入的目标体积，方便外层统计。
    """
    stl_file = random.choice(stl_files)
    mesh_origin = pv.read(stl_file)
    mesh = upsample_point_cloud(mesh_origin.points, num_clusters, sample_spacing)
    # 平滑并填补缺口，避免投影时出现尖锐噪声
    mesh.smooth_taubin(n_iter=10, pass_band=5, inplace=True)
    mesh = mesh.fill_holes(100)
    volume = mesh.volume
    # 通过立方根缩放保证体积匹配目标值
    scale_factor = (chosen_volume / volume) ** (1/3)
    mesh.points *= scale_factor
    mesh_origin.points *= scale_factor
    return stl_file, mesh, mesh_origin, chosen_volume

def upsample_point_cloud(points, num_clusters, sample_spacing):
    """自适应减少采样间距以重建更稠密的气泡曲面网格。"""
    cloud = pv.PolyData(points)
    sample_spacing = 0.1
    while True:
        mesh = cloud.reconstruct_surface(nbr_sz=10, sample_spacing=sample_spacing)
        new_points = np.asarray(mesh.points)
        if new_points.shape[0] < num_clusters * 1.1:
            # 点数不足时缩小采样间距以提升密度
            sample_spacing *= 0.8
        else:
            break
    return mesh

def generate_points_in_cube(num_points, cube_size=np.array([100, 100, 100]), num=100, poisson_max_iter=100000, prefer_gpu=True):
    """在指定长方体内执行近似泊松圆盘采样以获得气泡中心。"""
    xp = get_array_module(prefer_gpu=prefer_gpu)
    cube_size_np = np.asarray(cube_size, dtype=np.float32)
    cube_size_xp = xp.asarray(cube_size_np)
    radial_limit = cube_size_xp[0].item() / 2 * 0.85
    center_xy = cube_size_xp[:2] / 2
    min_spacing = float(cube_size_np.min() / (num ** (1 / 3)) * 0.5)

    accepted = xp.empty((0, 3), dtype=cube_size_xp.dtype)
    look_up_num = 0

    while accepted.shape[0] < num_points:
        remaining_attempts = poisson_max_iter - look_up_num
        if remaining_attempts <= 0:
            raise RuntimeError("超过最大迭代次数，可能无法在给定条件下生成足够的点")

        batch_size = min(max(128, (num_points - accepted.shape[0]) * 8), remaining_attempts)
        candidates = xp.random.rand(batch_size, 3).astype(cube_size_xp.dtype) * cube_size_xp
        look_up_num += batch_size

        mask_xy = xp.linalg.norm(candidates[:, :2] - center_xy, axis=1) < radial_limit
        if accepted.shape[0] > 0:
            diffs = candidates[:, None, :] - accepted[None, :, :]
            distances = xp.linalg.norm(diffs, axis=2)
            min_distances = distances.min(axis=1)
            mask_dist = min_distances > min_spacing
        else:
            mask_dist = xp.ones(batch_size, dtype=bool)

        mask = mask_xy & mask_dist
        selected_count = int(mask.sum())
        if selected_count:
            new_points = candidates[mask]
            accepted = xp.concatenate([accepted, new_points], axis=0)

    return to_numpy(accepted[:num_points])

def generater(stl_files, base_path, volume_size_x, volume_size_y, volume_height, gas_holdups, poisson_max_iter, sample_spacing):
    """生成目标气含率的气泡云，并输出位置表与合并 STL。"""
    for gas_holdup in gas_holdups:
        expected_volume = volume_size_x * volume_size_y * volume_height * gas_holdup

        names = []
        meshes = []
        meshes_origin = []
        volumes = []
        allocated_meshes = []
        allocated_origin_meshes = []
        total_volume = 0

        chosen_volumes = []
        xp = get_array_module()
        while total_volume < expected_volume:
            batch_size = max(32, int((expected_volume - total_volume) * 10))
            samples = xp.random.lognormal(mean=3.5, sigma=1.0, size=batch_size)
            for chosen_volume in to_numpy(samples) * 0.001:
                chosen_volumes.append(chosen_volume)
                total_volume += chosen_volume
                if total_volume >= expected_volume:
                    break

        # 并行执行上采样及体积缩放，加速气泡实例准备
        with mp.Pool(processes=mp.cpu_count()) as pool:
            mesh_data = pool.starmap(upsample_and_scale_mesh, [(stl_files, 20000, vol, sample_spacing) for vol in chosen_volumes])

        for stl_file, mesh, mesh_origin, volume in mesh_data:
            names.append(stl_file)
            meshes.append(mesh)
            meshes_origin.append(mesh_origin)
            volumes.append(volume * 10)

        points = generate_points_in_cube(len(meshes),
                                            cube_size=np.array([volume_size_x, volume_size_y, volume_height]) * 1.2, poisson_max_iter = poisson_max_iter)

        for mesh, mesh_origin, point in zip(meshes, meshes_origin, points):  
            # 将每个气泡平移到随机采样位置
            mesh.points += point
            allocated_meshes.append(mesh)
            mesh_origin.points += point
            allocated_origin_meshes.append(mesh_origin)

        mesh = pv.merge([mesh for mesh in allocated_meshes])
        mesh_origin = pv.merge([mesh for mesh in allocated_origin_meshes])
        mesh_origin.save(os.path.join(base_path, f'{gas_holdup}.stl'))  # 输出整体三维气泡场，方便后续检查

if __name__ =='__main__':
    # 命令行参数配置：可控制气泡数量、空间尺寸与渲染超参数
    parser = argparse.ArgumentParser(description='流场生成器与渲染器')
    parser.add_argument('--stl_path', type=str, default=r"/home/suxh/mount/dataset/mesh_20250619", help='STL文件的路径')
    parser.add_argument('--save_path', type=str, default=r"3Dbubbleflowrender/", help='保存路径')
    parser.add_argument('-num','--flow_num', type=int, default=50, help='生成数量')
    parser.add_argument('-x','--volume_size_x', type=int, default=5, help='流场宽度[mm]')
    parser.add_argument('-y','--volume_size_y', type=int, default=5, help='流场深度[mm]')
    parser.add_argument('-hh','--volume_height', type=int, default=15, help='流场高度[mm]')
    parser.add_argument('--gas_holdup', type=float, default=0.01, help='气含率')
    parser.add_argument('--poisson_max_iter', type=int, default=100000, help='泊松圆盘采样最大迭代次数')
    parser.add_argument('--sample_spacing', type=int, default=0.02, help='点云上采样的采样距离')

    args = parser.parse_args()

    # 收集输入目录下全部 STL 模型，供随机采样使用
    stl_files = [os.path.join(args.stl_path, f) for f in os.listdir(args.stl_path) if f.endswith('.stl')]

    for num in trange(args.flow_num):
        # 使用时间戳区分每次生成的数据集
        timestamp = datetime.now().strftime('%Y%m%dT%H%M%S-%f')
        base_path = f'{args.save_path}/{timestamp}'
        os.makedirs(base_path, exist_ok=True)
        generater(stl_files, base_path, args.volume_size_x, args.volume_size_y, args.volume_height, gas_holdups=[args.gas_holdup], poisson_max_iter=args.poisson_max_iter, sample_spacing = args.sample_spacing)

        print("保存路径:", base_path)
        print("生成数量:", args.flow_num)
        print("流场宽度[mm]:", args.volume_size_x)
        print("流场深度[mm]:", args.volume_size_y)
        print("流场高度[mm]:", args.volume_height)
        print("气含率:", args.gas_holdup)

        print("采样距离:", args.sample_spacing)
