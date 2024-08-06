import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

def read_ply(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd

def filter_point_cloud(pcd, nb_points=80, radius=0.05):
    pcd_filtered, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    pcd_filtered = pcd.select_by_index(ind)
    return pcd_filtered

def remove_walls(pcd, x_min=-1.5, x_max=1.5, y_min=-1.5, y_max=1.5, z_min=-1, z_max=3):
    points = np.asarray(pcd.points)
    mask = (points[:, 0] > x_min) & (points[:, 0] < x_max) & \
           (points[:, 1] > y_min) & (points[:, 1] < y_max) & \
           (points[:, 2] > z_min) & (points[:, 2] < z_max)
    pcd_filtered = pcd.select_by_index(np.where(mask)[0])
    return pcd_filtered

def detect_holes_boundaries(projected_points, radius=0.01, min_neighbors=12):
    kdtree = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(projected_points)))
    boundary_points = []

    for i, point in enumerate(projected_points):
        [k, idx, _] = kdtree.search_radius_vector_3d(point, radius)
        if k < min_neighbors:
            boundary_points.append(point)

    boundary_points = np.array(boundary_points)
    return boundary_points

def cluster_points(pcd, eps=0.02, min_samples=10):
    points = np.asarray(pcd.points)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    return labels

def remove_small_clusters(pcd, labels, min_size):
    unique_labels = set(labels)
    new_labels = labels.copy()
    for label in unique_labels:
        if label == -1:
            continue  # Ignore noise points
        label_indices = np.where(labels == label)[0]
        if len(label_indices) < min_size:
            new_labels[label_indices] = -1  # Mark small clusters as noise
    return new_labels

def pca_analysis(points):
    pca = PCA(n_components=3)
    pca.fit(points)
    explained_variance = pca.explained_variance_
    return explained_variance

def find_most_planar_cluster(pcd, labels):
    unique_labels = set(labels)
    most_planar_label = None
    min_variance_ratio = float('inf')
    
    for label in unique_labels:
        if label == -1:
            continue  # Ignore noise points
        cluster_indices = np.where(labels == label)[0]
        cluster_points = np.asarray(pcd.select_by_index(cluster_indices).points)
        
        if len(cluster_points) < 3:  # PCA requires at least 3 points
            continue
        
        explained_variance = pca_analysis(cluster_points)
        variance_ratio = explained_variance[2] / explained_variance[0]  # Ratio of smallest to largest variance
        
        if variance_ratio < min_variance_ratio:
            min_variance_ratio = variance_ratio
            most_planar_label = label
    
    if most_planar_label is not None:
        planar_cluster_indices = np.where(labels == most_planar_label)[0]
        planar_cluster = pcd.select_by_index(planar_cluster_indices)
        return planar_cluster
    else:
        return None

def detect_planes(pcd, distance_threshold=0.03, ransac_n=3, num_iterations=1000):
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                             ransac_n=ransac_n,
                                             num_iterations=num_iterations)
    return plane_model, inliers

def svd_plane_fitting(points):
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    U, S, Vt = np.linalg.svd(centered_points)
    normal = Vt[2, :]
    d = -centroid.dot(normal)
    return normal[0], normal[1], normal[2], d

def project_points_to_plane(points, a, b, c, d):
    normal = np.array([a, b, c])
    normal_norm = np.linalg.norm(normal)
    projected_points = points - (points.dot(normal) + d)[:, np.newaxis] * normal / (normal_norm ** 2)
    return projected_points

def main(ply_file, min_cluster_size):
    pcd = read_ply(ply_file)
    pcd_filtered = filter_point_cloud(pcd)
    pcd_filtered = remove_walls(pcd_filtered)
    labels = cluster_points(pcd_filtered)
    labels = remove_small_clusters(pcd_filtered, labels, min_cluster_size)
    planar_cluster = find_most_planar_cluster(pcd_filtered, labels)
    
    if planar_cluster:
        plane_model, inliers = detect_planes(planar_cluster)
        inlier_cloud = planar_cluster.select_by_index(inliers)
        
        # SVD를 사용하여 평면 방정식 구하기
        points = np.asarray(inlier_cloud.points)
        a, b, c, d = svd_plane_fitting(points)
        print(f"Plane equation: {a}x + {b}y + {c}z + {d} = 0")
        
        # 평면에 점들을 사영시키기
        all_points = np.asarray(planar_cluster.points)
        projected_points = project_points_to_plane(all_points, a, b, c, d)
        
        # 구멍 주변 경계점 검출
        boundary_points = detect_holes_boundaries(projected_points)
        
        # 시각화
        projected_cloud = o3d.geometry.PointCloud()
        projected_cloud.points = o3d.utility.Vector3dVector(projected_points)
        
        boundary_cloud = o3d.geometry.PointCloud()
        boundary_cloud.points = o3d.utility.Vector3dVector(boundary_points)
        boundary_cloud.paint_uniform_color([0, 0, 0])  # 경계점을 검정색으로 시각화
        
        #o3d.visualization.draw_geometries([projected_cloud, boundary_cloud])
        o3d.visualization.draw_geometries([boundary_cloud])
    else:
        print("No planar cluster found")

# PLY 파일 경로를 지정하고 최소 클러스터 크기 설정
ply_file = "./accumulated_pointcloud.ply"
min_cluster_size = 1000  # 원하는 최소 클러스터 크기로 설정
main(ply_file, min_cluster_size)

