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
from concurrent.futures import ThreadPoolExecutor
import pyvista as pv
import numpy as np
from tqdm import trange
import multiprocessing as mp
import argparse

try:
    import cupy as cp  # GPU array backend
except ImportError:  # pragma: no cover - GPU optional
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
    cube_size_xp = xp.asarray(cube_size, dtype=xp.float32)

    radial_limit_sq = (cube_size_xp[0] * xp.asarray(0.5 * 0.85, dtype=cube_size_xp.dtype)) ** 2
    center_xy = cube_size_xp[:2] * xp.asarray(0.5, dtype=cube_size_xp.dtype)

    num_array = xp.asarray(num, dtype=cube_size_xp.dtype)
    if hasattr(xp, "cbrt"):
        root_num = xp.cbrt(num_array)
    else:  # NumPy<1.10 fallback
        root_num = xp.power(num_array, xp.asarray(1.0 / 3.0, dtype=cube_size_xp.dtype))

    min_spacing = cube_size_xp.min() / root_num
    min_spacing *= xp.asarray(0.5, dtype=cube_size_xp.dtype)
    min_spacing_sq = min_spacing * min_spacing

    accepted = xp.empty((0, 3), dtype=cube_size_xp.dtype)
    look_up_num = 0

    while accepted.shape[0] < num_points:
        remaining_attempts = poisson_max_iter - look_up_num
        if remaining_attempts <= 0:
            raise RuntimeError("超过最大迭代次数，可能无法在给定条件下生成足够的点")

        batch_size = min(max(128, (num_points - int(accepted.shape[0])) * 8), remaining_attempts)
        random_sample = xp.random.rand(batch_size, 3)
        candidates = xp.asarray(random_sample, dtype=cube_size_xp.dtype) * cube_size_xp
        look_up_num += batch_size

        delta_xy = candidates[:, :2] - center_xy
        mask_xy = xp.sum(delta_xy * delta_xy, axis=1) < radial_limit_sq

        if accepted.shape[0] > 0:
            diffs = candidates[:, None, :] - accepted[None, :, :]
            distances_sq = xp.sum(diffs * diffs, axis=2)
            min_distances_sq = distances_sq.min(axis=1)
            mask_dist = min_distances_sq > min_spacing_sq
        else:
            mask_dist = xp.ones(batch_size, dtype=bool)

        mask = mask_xy & mask_dist
        selected_count = int(mask.sum())
        if selected_count > 0:
            accepted = xp.concatenate([accepted, candidates[mask]], axis=0)

    return to_numpy(accepted[:num_points])

def generater(stl_files, base_path, volume_size_x, volume_size_y, volume_height, gas_holdups, poisson_max_iter, sample_spacing, mesh_workers=None, flow_index=0):
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

        xp = get_array_module()
        xp_dtype = xp.float32 if hasattr(xp, "float32") else np.float32
        chosen_volume_chunks = []
        while total_volume < expected_volume:
            batch_size = max(32, int((expected_volume - total_volume) * 10))
            samples = xp.random.lognormal(mean=3.5, sigma=1.0, size=batch_size)
            samples = xp.asarray(samples, dtype=xp_dtype) * xp.asarray(0.001, dtype=xp_dtype)

            cumulative = xp.cumsum(samples)
            threshold = expected_volume - total_volume
            viable_mask = cumulative <= threshold
            count = int(viable_mask.sum())
            if count == 0:
                count = 1  # 至少取一个样本，避免死循环

            chosen_chunk = samples[:count]
            chosen_volume_chunks.append(chosen_chunk)
            total_volume += float(to_numpy(cumulative[count - 1]))

            if total_volume >= expected_volume:
                break

        if chosen_volume_chunks:
            chosen_volumes_xp = xp.concatenate(chosen_volume_chunks, axis=0)
        else:
            chosen_volumes_xp = xp.empty((0,), dtype=xp_dtype)

        chosen_volumes = to_numpy(chosen_volumes_xp)
        chosen_volumes_list = [float(v) for v in np.asarray(chosen_volumes).ravel()]

        tasks = [(stl_files, 20000, vol, sample_spacing) for vol in chosen_volumes_list]

        # 并行执行上采样及体积缩放，加速气泡实例准备
        pool_size = mesh_workers or mp.cpu_count()
        if pool_size <= 1 or not tasks:
            mesh_data = [upsample_and_scale_mesh(*task) for task in tasks]
        else:
            current = mp.current_process()

            if getattr(current, "daemon", False):
                # Daemon 进程无法再派生子进程，退化为线程池避免报错
                with ThreadPoolExecutor(max_workers=pool_size) as executor:
                    mesh_data = list(executor.map(lambda params: upsample_and_scale_mesh(*params), tasks))
            else:
                ctx = mp.get_context("spawn")
                with ctx.Pool(processes=pool_size) as pool:
                    mesh_data = pool.starmap(upsample_and_scale_mesh, tasks)

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
        mesh_origin.save(os.path.join(base_path, f'{gas_holdup}_{volume_size_x}_{volume_size_y}_{volume_height}.stl'))  # 输出整体三维气泡场，方便后续检查


def _run_single_flow(task):
    """Worker helper used to generate an individual flow field."""
    stl_files, config, flow_index = task
    timestamp = datetime.now().strftime('%Y%m%dT%H%M%S-%f')
    base_dir = f"{timestamp}-flow{flow_index:04d}"
    base_path = os.path.join(config['save_path'], base_dir)
    os.makedirs(base_path, exist_ok=True)

    generater(
        stl_files,
        base_path,
        config['volume_size_x'],
        config['volume_size_y'],
        config['volume_height'],
        gas_holdups=[config['gas_holdup']],
        poisson_max_iter=config['poisson_max_iter'],
        sample_spacing=config['sample_spacing'],
        mesh_workers=config['mesh_workers'],
        flow_index=flow_index,
    )
    return flow_index, base_path


def _print_flow_summary(result, args):
    """Standardized console output once a flow field finishes generating."""
    flow_index, base_path = result
    print(f"[flow {flow_index}] 保存路径:", base_path)
    print("生成数量:", args.flow_num)
    print("流场宽度[mm]:", args.volume_size_x)
    print("流场深度[mm]:", args.volume_size_y)
    print("流场高度[mm]:", args.volume_height)
    print("气含率:", args.gas_holdup)
    print("采样距离:", args.sample_spacing)


if __name__ =='__main__':
    # 命令行参数配置：可控制气泡数量、空间尺寸与渲染超参数
    parser = argparse.ArgumentParser(description='流场生成器与渲染器')
    parser.add_argument('--stl_path', type=str, default=r"/home/suxh/mount/dataset/mesh_20250619", help='STL文件的路径')
    parser.add_argument('--save_path', type=str, default=r"3Dbubbleflowrender/", help='保存路径')
    parser.add_argument('-num','--flow_num', type=int, default=50, help='生成数量')
    parser.add_argument('-x','--volume_size_x', type=int, default=5, help='流场宽度[mm]')
    parser.add_argument('-y','--volume_size_y', type=int, default=5, help='流场深度[mm]')
    parser.add_argument('-hh','--volume_height', type=int, default=15, help='流场高度[mm]')
    parser.add_argument('--gas_holdup', type=float, default=0.05, help='气含率')
    parser.add_argument('--poisson_max_iter', type=int, default=100000, help='泊松圆盘采样最大迭代次数')
    parser.add_argument('--sample_spacing', type=int, default=0.02, help='点云上采样的采样距离')
    parser.add_argument('--workers', type=int, default=8, help='并行生成流场的进程数')
    parser.add_argument('--mesh_workers', type=int, default=0, help='每个流场内上采样进程数，0 表示自动分配')

    args = parser.parse_args()

    # 收集输入目录下全部 STL 模型，供随机采样使用
    stl_files = [os.path.join(args.stl_path, f) for f in os.listdir(args.stl_path) if f.endswith('.stl')]

    os.makedirs(args.save_path, exist_ok=True)

    flow_workers = max(1, args.workers)
    if args.mesh_workers and args.mesh_workers > 0:
        mesh_workers = args.mesh_workers
    else:
        mesh_workers = max(1, mp.cpu_count() // flow_workers)

    config = {
        'save_path': args.save_path,
        'volume_size_x': args.volume_size_x,
        'volume_size_y': args.volume_size_y,
        'volume_height': args.volume_height,
        'gas_holdup': args.gas_holdup,
        'poisson_max_iter': args.poisson_max_iter,
        'sample_spacing': args.sample_spacing,
        'mesh_workers': mesh_workers,
    }

    if flow_workers <= 1 or args.flow_num == 1:
        for flow_index in trange(args.flow_num):
            result = _run_single_flow((stl_files, config, flow_index))
            _print_flow_summary(result, args)
    else:
        tasks = [(stl_files, config, idx) for idx in range(args.flow_num)]
        progress = trange(args.flow_num, desc='Generating', leave=False)
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=flow_workers) as pool:
            for result in pool.imap_unordered(_run_single_flow, tasks):
                _print_flow_summary(result, args)
                progress.update(1)
        progress.close()
