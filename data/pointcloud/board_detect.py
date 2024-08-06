import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

def read_ply(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd

def filter_point_cloud(pcd, nb_points=80, radius=0.1):
    pcd_filtered, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    pcd_filtered = pcd.select_by_index(ind)
    # pcd_filtered = pcd.voxel_down_sample(voxel_size=0.01)
    return pcd_filtered

def remove_walls(pcd, x_min=-1.5, x_max=1.5, y_min=-1.5, y_max=1.5, z_min=-1, z_max=3):
    points = np.asarray(pcd.points)
    mask = (points[:, 0] > x_min) & (points[:, 0] < x_max) & \
           (points[:, 1] > y_min) & (points[:, 1] < y_max) & \
           (points[:, 2] > z_min) & (points[:, 2] < z_max)
    pcd_filtered = pcd.select_by_index(np.where(mask)[0])
    return pcd_filtered

def cluster_points(pcd, eps=0.025, min_samples=10):
    points = np.asarray(pcd.points)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    return labels

def visualize_clusters(pcd, labels):
    max_label = labels.max()
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels == -1] = 0  # Noise points are black
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcd])

def main(ply_file):
    pcd = read_ply(ply_file)
    pcd_filtered = filter_point_cloud(pcd)
    pcd_filtered = remove_walls(pcd_filtered)
    labels = cluster_points(pcd_filtered)
    visualize_clusters(pcd_filtered, labels)
    return pcd_filtered, labels


# PLY 파일 경로를 지정
ply_file = "./accumulated_pointcloud.ply"
pcd_filtered, labels = main(ply_file)

