# FlowRenderer.py
import os
import random
import cv2
from datetime import datetime
import pyvista as pv
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter, gaussian_filter
import csv
import shutil
from pyinstrument import Profiler
from tqdm import trange
import multiprocessing as mp
import argparse

def upsample_and_scale_mesh(stl_files, num_clusters, chosen_volume, sample_spacing):
    stl_file = random.choice(stl_files)
    mesh_origin = pv.read(stl_file)
    mesh = upsample_point_cloud(mesh_origin.points, num_clusters, sample_spacing)
    mesh.smooth_taubin(n_iter=10, pass_band=5, inplace=True)
    mesh = mesh.fill_holes(100)
    volume = mesh.volume
    scale_factor = (chosen_volume / volume) ** (1/3)
    mesh.points *= scale_factor
    mesh_origin.points *= scale_factor
    return stl_file, mesh, mesh_origin, chosen_volume

def generate_uniform_points_on_sphere(N=1000):
    phi = (np.sqrt(5) - 1) / 2
    n = np.arange(0, N)
    z = ((2*n + 1) / N - 1)
    x = (np.sqrt(1 - z**2)) * np.cos(2 * np.pi * (n + 1) * phi)
    y = (np.sqrt(1 - z**2)) * np.sin(2 * np.pi * (n + 1) * phi)
    points = np.stack([x, y, z], axis=-1)
    return points

def cv2_enhance_contrast(img, factor):
    mean = np.uint8(cv2.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))[0])
    img_deg = np.ones_like(img) * mean
    return cv2.addWeighted(img, factor, img_deg, 1-factor, 0.0)

def upsample_point_cloud(points, num_clusters, sample_spacing):
    cloud = pv.PolyData(points)
    sample_spacing = 0.1
    while True:
        mesh = cloud.reconstruct_surface(nbr_sz=10, sample_spacing=sample_spacing)
        new_points = np.asarray(mesh.points)
        if new_points.shape[0] < num_clusters * 1.1:
            sample_spacing *= 0.8
        else:
            break
    return mesh

def generate_points_in_cube(num_points, cube_size=np.array([100,100,100]), num=100, poisson_max_iter = 100000):
    rnd_points = []
    Look_up_num = 0
    while len(rnd_points) < num_points:
        if Look_up_num >= poisson_max_iter:
            raise RuntimeError("超过最大迭代次数，可能无法在给定条件下生成足够的点")
        x, y, z = np.random.rand(3) * cube_size
        if all(np.linalg.norm(np.array([x, y, z]) - p) > (cube_size.min() / (num**(1/3)) * 0.5) for p in rnd_points)\
                and np.linalg.norm(np.array([x, y]) - cube_size[:2]/2) < cube_size[0]/2*0.85:
            rnd_points.append([x, y, z])
        Look_up_num += 1
    return np.array(rnd_points)

def pixel_coloring(masks_path, alpha, all_points, all_vectors, min_x, min_y, scale, canvas_range_x, canvas_range_y):
    bboxes = []
    bub_conts = []
    mapped_points = np.ones((canvas_range_x, canvas_range_y))
    for points, vectors in zip(all_points, all_vectors):      # 每一个points都是一个气泡
        mask = vectors[:, 2] > -99099999
        filtered_points = points[mask]
        filtered_normals = vectors[mask]
        angles = filtered_normals[:, 2] / np.linalg.norm(filtered_normals, axis=1)
        M = angles ** alpha
        min_mapped_x, max_mapped_x, min_mapped_y, max_mapped_y = float('inf'), float('-inf'), float('inf'), float('-inf')
        for i in range(filtered_points.shape[0]):
            mapped_x, mapped_y = (filtered_points[i, 0] - min_x) * scale, (filtered_points[i, 1] - min_y) * scale
            min_mapped_x, max_mapped_x = min(min_mapped_x, mapped_x), max(max_mapped_x, mapped_x)
            min_mapped_y, max_mapped_y = min(min_mapped_y, mapped_y), max(max_mapped_y, mapped_y)
            l1 = np.square(mapped_x - np.floor(mapped_x)) + np.square(mapped_y - np.floor(mapped_y))
            l2 = np.square(mapped_x - np.ceil(mapped_x)) + np.square(mapped_y - np.floor(mapped_y))
            l3 = np.square(mapped_x - np.floor(mapped_x)) + np.square(mapped_y - np.ceil(mapped_y))
            l4 = np.square(mapped_x - np.ceil(mapped_x)) + np.square(mapped_y - np.ceil(mapped_y))
            total = l1 + l2 + l3 + l4
            for x, y, l in [(int(np.floor(mapped_x)), int(np.floor(mapped_y)), l1),
                            (int(np.ceil(mapped_x)), int(np.floor(mapped_y)), l2),
                            (int(np.floor(mapped_x)), int(np.ceil(mapped_y)), l3),
                            (int(np.ceil(mapped_x)), int(np.ceil(mapped_y)), l4),
                            (int(np.floor(mapped_x)) - 1, int(np.floor(mapped_y)), l1),
                            (int(np.floor(mapped_x)), int(np.floor(mapped_y)) - 1, l1),
                            (int(np.ceil(mapped_x)) + 1, int(np.floor(mapped_y)), l2),
                            (int(np.ceil(mapped_x)), int(np.floor(mapped_y)) - 1, l2),
                            (int(np.floor(mapped_x)) - 1, int(np.ceil(mapped_y)), l3),
                            (int(np.floor(mapped_x)), int(np.ceil(mapped_y)) + 1, l3),
                            (int(np.ceil(mapped_x)) + 1, int(np.ceil(mapped_y)), l4),
                            (int(np.ceil(mapped_x)), int(np.ceil(mapped_y)) + 1, l4)]:
                if x < 0 or x >= canvas_range_x or y < 0 or y >= canvas_range_y:
                    continue
                if mapped_points[x, y] != 1:
                    mapped_points[x, y] = 1
                mapped_points[x, y] += M[i] * (total - l) / total

        bboxes.append((min_mapped_x, max_mapped_x, min_mapped_y, max_mapped_y))
        mask_image = np.zeros((canvas_range_x, canvas_range_y), dtype=np.uint8)
        for i in range(filtered_points.shape[0]):
            mapped_x, mapped_y = (filtered_points[i, 0] - min_x) * scale, (filtered_points[i, 1] - min_y) * scale
            if 0 <= int(mapped_x) < canvas_range_x and 0 <= int(mapped_y) < canvas_range_y:
                mask_image[int(mapped_x), int(mapped_y)] = 255
        mask_image_path = os.path.join(masks_path, f'mask_{len(bboxes)}.png')
        kernel = np.ones((5,5),np.uint8)
        mask_image = cv2.dilate(mask_image, kernel, iterations=1)
        cv2.imwrite(mask_image_path, mask_image)

        ret, mask = cv2.threshold(mask_image, 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        if len(contours) >= 1:
            second_contour = contours[0]
            bub_conts.append(second_contour)
    return bboxes, bub_conts, mapped_points

def process_projection(mp_args):
    i_projection, point_fibonacci, allocated_meshes, allocated_origin_meshes, base_path, gas_holdup, v, scale, alpha, truncation = mp_args
    masks_path = os.path.join(base_path, f'{str(i_projection).zfill(3)}/masks')
    if not os.path.exists(masks_path):
        os.makedirs(masks_path)
    meshes = [mesh.rotate_vector(np.cross(point_fibonacci, v), np.arccos(np.dot(point_fibonacci, v)) * 180 / np.pi, inplace=False) for mesh in allocated_meshes]
    meshes_origin = [mesh.rotate_vector(np.cross(point_fibonacci, v), np.arccos(np.dot(point_fibonacci, v)) * 180 / np.pi, inplace=False) for mesh in allocated_origin_meshes]
    mesh_fibonacci = pv.merge([mesh for mesh in meshes_origin])
    mesh_fibonacci.save(os.path.join(base_path, f'{str(i_projection).zfill(3)}/{gas_holdup}_fibonacci.stl'))

    all_points = [mesh.points for mesh in meshes]
    all_vectors = [mesh.point_normals for mesh in meshes]
    volume_size_x = max([np.max(point[:, 0]) for point in all_points]) - min([np.min(point[:, 0]) for point in all_points])
    volume_size_y = max([np.max(point[:, 1]) for point in all_points]) - min([np.min(point[:, 1]) for point in all_points])
    canvas_range_x = int(scale * volume_size_x)
    canvas_range_y = int(scale * volume_size_y)

    # Sort all_points and all_vectors based on the maximum value of the last column of each array in all_points
    sorted_indices = np.argsort([np.max(point[:, 2]) for point in all_points])[::-1]
    all_points = [all_points[i] for i in sorted_indices]
    all_vectors = [all_vectors[i] for i in sorted_indices]

    min_x, max_x = np.min([np.min(point[:, 0]) for point in all_points]), np.max([np.max(point[:, 0]) for point in all_points])
    min_y, max_y = np.min([np.min(point[:, 1]) for point in all_points]), np.max([np.max(point[:, 1]) for point in all_points])
    bboxes, bub_conts, mapped_points = pixel_coloring(masks_path, alpha, all_points, all_vectors, min_x, min_y, scale, canvas_range_x, canvas_range_y)

    indices = np.where(mapped_points == 1)
    mapped_points[indices] = 0
    mapped_points = gaussian_filter(mapped_points, sigma=1)
    
    mapped_points_normalized = np.clip(mapped_points / mapped_points.max(), 0, truncation) / truncation
    mapped_points_normalized[indices] = 1

    mapped_points_normalized = gaussian_filter(mapped_points_normalized, sigma=0.75)
    mapped_points_normalized = median_filter(mapped_points_normalized, size=5)
    mapped_points_normalized = (mapped_points_normalized - mapped_points_normalized.min()) / (mapped_points_normalized.max() - mapped_points_normalized.min())

    mapped_points_normalized = (mapped_points_normalized * 255).astype(np.uint8).T
    mapped_points_normalized = cv2.cvtColor(mapped_points_normalized, cv2.COLOR_GRAY2RGB)
    mapped_points_normalized = cv2_enhance_contrast(mapped_points_normalized, 2)

    image_with_bboxes = mapped_points_normalized.copy()
    with open(os.path.join(base_path, f'{str(i_projection).zfill(3)}/bubbly_flow.txt'), 'w') as f:
        for i, bbox in enumerate(bboxes):
            min_mapped_x, max_mapped_x, min_mapped_y, max_mapped_y = map(int, bbox)
            is_overlapping = False
            for j, other_bbox in enumerate(bboxes):
                if i != j:
                    other_min_mapped_x, other_max_mapped_x, other_min_mapped_y, other_max_mapped_y = map(int, other_bbox)
                    if (min_mapped_x <= other_max_mapped_x and max_mapped_x >= other_min_mapped_x and
                            min_mapped_y <= other_max_mapped_y and max_mapped_y >= other_min_mapped_y):
                        is_overlapping = True
                        break
            color = (0, 0, 192) if is_overlapping else (53, 130, 84)
            cv2.rectangle(image_with_bboxes, (min_mapped_x - 2, min_mapped_y - 2), (max_mapped_x + 2, max_mapped_y + 2), color, 2)
            f.write(f"{int(is_overlapping)} {(min_mapped_x + min_mapped_x)/ 2 / canvas_range_x} {(min_mapped_y + min_mapped_y)/ 2 / canvas_range_y} {(max_mapped_x-min_mapped_x)/canvas_range_x} {(max_mapped_y-min_mapped_y)/canvas_range_y}\n")
    image_with_bboxes = cv2.transpose(image_with_bboxes)
    mapped_points_normalized = cv2.transpose(mapped_points_normalized)
    cv2.imwrite(os.path.join(base_path, f'{str(i_projection).zfill(3)}/bubbly_flow.png'), mapped_points_normalized)
    cv2.imwrite(os.path.join(base_path, f'{str(i_projection).zfill(3)}/bubbly_flow_bboxes.png'), image_with_bboxes)

    SAM_background_merge = np.zeros((canvas_range_x + 128 * 2, canvas_range_y + 128 * 2, 3), np.uint8)
    colors = []
    for xx in range(len(bub_conts)):
        bub_conts[xx][:, 0, 0] += 128
        bub_conts[xx][:, 0, 1] += 128
    for bub_cont in bub_conts:
        if bub_cont.shape[0] > 6:
            zeros = np.ones((SAM_background_merge.shape), dtype=np.uint8) * 255
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.drawContours(SAM_background_merge, [bub_cont], -1, color=color, thickness=cv2.FILLED)
            colors.append(tuple(c / 255 for c in color))

    SAM_background_merge = SAM_background_merge[128 : canvas_range_x + 128, 128 : canvas_range_y + 128]
    cv2.imwrite(os.path.join(base_path, f'{str(i_projection).zfill(3)}/mask.png'), SAM_background_merge)
    
    image = cv2.imread(os.path.join(base_path, f'{str(i_projection).zfill(3)}/bubbly_flow.png'))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    img = np.ones((image.shape[0], image.shape[1], 4))
    img[:,:,3] = 0
    mask_files = [f for f in os.listdir(masks_path) if f.endswith('.png')]
    for (color, mask_file) in zip(colors, mask_files):
        mask_path = os.path.join(masks_path, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        m = mask > 0
        img[m] = np.concatenate([color, [0.4]])
    ax.imshow(img)
    plt.axis('off')
    plt.savefig(os.path.join(base_path, f'{str(i_projection).zfill(3)}/mask_merge.png'), bbox_inches='tight', pad_inches=0, dpi=150)

    image = cv2.imread(os.path.join(base_path, f'{str(i_projection).zfill(3)}/bubbly_flow_bboxes.png'))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    img = np.ones((image.shape[0], image.shape[1], 4))
    img[:,:,3] = 0
    mask_files = [f for f in os.listdir(masks_path) if f.endswith('.png')]
    for (color, mask_file) in zip(colors, mask_files):
        mask_path = os.path.join(masks_path, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        m = mask > 0
        img[m] = np.concatenate([color, [0.4]])
    ax.imshow(img)
    plt.axis('off')
    plt.savefig(os.path.join(base_path, f'{str(i_projection).zfill(3)}/mask_merge_bboxes.png'), bbox_inches='tight', pad_inches=0, dpi=150)

def generater(stl_files, base_path, volume_size_x, volume_size_y, volume_height, gas_holdups, alpha, truncation, poisson_max_iter, sample_spacing):
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
        while total_volume < expected_volume:
            chosen_volume = np.random.lognormal(mean=3.5, sigma=1.0) / 1000
            chosen_volumes.append(chosen_volume)
            total_volume += chosen_volume

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
            mesh.points += point
            allocated_meshes.append(mesh)
            mesh_origin.points += point
            allocated_origin_meshes.append(mesh_origin)

        with open(os.path.join(base_path, 'names_points.csv'), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Name', 'X', 'Y', 'Z', 'Volume'])
            for name, point, volume in zip(names, points, volumes):
                writer.writerow([name, point[0], point[1], point[2], volume])

        mesh = pv.merge([mesh for mesh in allocated_meshes])
        mesh_origin = pv.merge([mesh for mesh in allocated_origin_meshes])
        mesh_origin.save(os.path.join(base_path, f'{gas_holdup}.stl'))

        points_fibonacci = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
        points_output_path = os.path.join(base_path, "points_fibonacci.csv")
        np.savetxt(points_output_path, points_fibonacci, delimiter=",")
        v = np.array([0.01, 0.01, 1])
        i_projection = 0
        scale = 100

        # Prepare data for multiprocessing
        args_list = []
        for i_projection, point_fibonacci in enumerate(points_fibonacci):
            mp_args = (i_projection, point_fibonacci, allocated_meshes, allocated_origin_meshes, base_path, gas_holdup, v, scale, alpha, truncation)
            args_list.append(mp_args)

        # Use multiprocessing to process each projection in parallel
        with mp.Pool(processes=mp.cpu_count()) as pool:
            pool.map(process_projection, args_list)
                
    # 保存当前py文件到base_path目录下
    current_file_path = __file__
    destination_path = os.path.join(base_path, 'FlowRenderer.py')
    shutil.copy(current_file_path, destination_path)

if __name__ =='__main__':

    parser = argparse.ArgumentParser(description='流场生成器与渲染器')
    parser.add_argument('--stl_path', type=str, default=r"dataset/bubbles_mesh", help='STL文件的路径')
    parser.add_argument('--save_path', type=str, default=r"3Dbubbleflowrender/", help='保存路径')
    parser.add_argument('-num','--flow_num', type=int, default=50, help='生成数量')
    parser.add_argument('-x','--volume_size_x', type=int, default=5, help='流场宽度[mm]')
    parser.add_argument('-y','--volume_size_y', type=int, default=5, help='流场深度[mm]')
    parser.add_argument('-hh','--volume_height', type=int, default=15, help='流场高度[mm]')
    parser.add_argument('--gas_holdup', type=float, default=0.01, help='气含率')
    parser.add_argument('-a','--alpha', type=int, default=4, help='向量指数:Alpha')
    parser.add_argument('-t','--truncation', type=float, default=0.75, help='截断值')
    parser.add_argument('--poisson_max_iter', type=int, default=100000, help='泊松圆盘采样最大迭代次数')
    parser.add_argument('--sample_spacing', type=int, default=0.02, help='点云上采样的采样距离')

    args = parser.parse_args()

    stl_files = [os.path.join(args.stl_path, f) for f in os.listdir(args.stl_path) if f.endswith('.stl')]

    for num in trange(args.flow_num):
        timestamp = datetime.now().strftime('%Y%m%dT%H%M%S-%f')
        base_path = f'{args.save_path}/{timestamp}'
        os.makedirs(base_path, exist_ok=True)
        
        # profiler = Profiler()
        # profiler.start()
        generater(stl_files, base_path, args.volume_size_x, args.volume_size_y, args.volume_height, gas_holdups=[args.gas_holdup], alpha=args.alpha, truncation=args.truncation, poisson_max_iter=args.poisson_max_iter, sample_spacing = args.sample_spacing)
        # profiler.stop()
        # profiler.print()


        print("保存路径:", base_path)
        print("生成数量:", args.flow_num)
        print("流场宽度[mm]:", args.volume_size_x)
        print("流场深度[mm]:", args.volume_size_y)
        print("流场高度[mm]:", args.volume_height)
        print("气含率:", args.gas_holdup)
        print("向量指数:Alpha:", args.alpha)
        print("截断值:", args.truncation)
        print("采样距离:", args.sample_spacing)