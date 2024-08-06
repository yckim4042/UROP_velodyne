#!/usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

class VelodyneBlobDetector:
    def __init__(self):
        self.point_cloud_sub = rospy.Subscriber("/velodyne_points", PointCloud2, self.point_cloud_callback)
        self.accumulated_points = []
        self.point_cloud_data = None

    def point_cloud_callback(self, msg):
        points = []
        for point in pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True):
            points.append([point[0], point[1], point[2], point[3]])
        self.accumulated_points.extend(points)

    def process_point_cloud(self):
        if not self.accumulated_points:
            rospy.logwarn("No point cloud data accumulated")
            return

        # 1. 포인트 클라우드 데이터 읽기 및 필터링
        self.point_cloud_data = np.array(self.accumulated_points)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.point_cloud_data[:, :3])
        pcd_filtered = filter_point_cloud(pcd)

        # 2. 반사율 이미지 생성 및 블랍 검출
        reflec_image = create_reflectivity_image(self.point_cloud_data)
        keypoints = find_blobs(reflec_image)

        # 3. 블랍을 검출한 후, 가로로 200픽셀 내에서 가장 많은 블랍이 포함된 이미지의 일부분 선택
        best_section = select_best_section(reflec_image, keypoints)

        # 4. 선택한 이미지를 16픽셀로 분할한 후, 정확히 12개의 블랍이 검출될 때까지 파라미터를 조절
        split_images = split_image(best_section)
        all_keypoints = []
        for split_img in split_images:
            keypoints, params = adjust_blobs(split_img)
            all_keypoints.extend(keypoints)

        # 5. 포인트 클라우드 데이터와 블랍 좌표 매핑
        blob_points = map_blobs_to_point_cloud(all_keypoints, self.point_cloud_data)

        # 6. 클러스터 검출 및 평면 사영
        labels = cluster_points(pcd_filtered)
        similar_cluster = find_most_similar_cluster(pcd_filtered, blob_points, labels)

        if similar_cluster:
            # 평면 검출 및 사영
            points = np.asarray(similar_cluster.points)
            a, b, c, d = svd_plane_fitting(points)
            projected_points = project_points_to_plane(points, a, b, c, d)

            # 7. 경계점 검출 및 시각화
            boundary_points = detect_holes_boundaries(projected_points)
            boundary_labels = cluster_boundary_points(boundary_points)
            colored_boundary_cloud = create_colored_point_cloud(boundary_points, boundary_labels)

            # 시각화
            visualize_results(pcd_filtered, colored_boundary_cloud, projected_points, points)
        else:
            rospy.logwarn("No similar cluster found")

def filter_point_cloud(pcd, nb_points=80, radius=0.05):
    pcd_filtered, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    pcd_filtered = pcd.select_by_index(ind)
    return pcd_filtered

def create_reflectivity_image(point_data, img_width=1024, img_height=64):
    points = point_data[:, :3]
    reflectivity = point_data[:, 3]

    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()

    x_normalized = (points[:, 0] - x_min) / (x_max - x_min)
    y_normalized = (points[:, 1] - y_min) / (y_max - y_min)

    x_img = (x_normalized * (img_width - 1)).astype(np.int32)
    y_img = (y_normalized * (img_height - 1)).astype(np.int32)

    reflec_image = np.zeros((img_height, img_width), dtype=np.float32)
    reflec_image[y_img, x_img] = reflectivity

    return reflec_image

def find_blobs(image):
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 255
    params.filterByArea = True
    params.minArea = 5
    params.maxArea = 2000
    params.filterByCircularity = True
    params.minCircularity = 0.7
    params.filterByConvexity = True
    params.minConvexity = 0.8
    params.filterByInertia = True
    params.minInertiaRatio = 0.5

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(image)
    return keypoints

def select_best_section(image, keypoints):
    height, width = image.shape
    num_sections = width // 200
    blob_count = np.zeros(num_sections)

    for keypoint in keypoints:
        x = int(keypoint.pt[0])
        blob_count[x // 200] += 1

    max_blob_index = np.argmax(blob_count)
    start_x = max_blob_index * 200
    end_x = start_x + 200

    return image[:, start_x:end_x]

def split_image(image, num_splits=16):
    height, width = image.shape
    split_height = height // 4
    split_width = width // 4

    split_images = []
    for i in range(4):
        for j in range(4):
            split_img = image[i*split_height:(i+1)*split_height, j*split_width:(j+1)*split_width]
            split_images.append(split_img)
    return split_images

def adjust_blobs(image, target_blob_count=12):
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 255
    params.filterByArea = True
    params.filterByCircularity = True
    params.filterByConvexity = True
    params.filterByInertia = True

    min_area_range = [5, 10, 20, 30, 50]
    max_area_range = [100, 200, 500, 1000, 2000]
    min_circularity_range = [0.1, 0.3, 0.5, 0.7, 0.9]
    min_convexity_range = [0.6, 0.7, 0.8, 0.9]
    min_inertia_range = [0.4, 0.5, 0.6, 0.7]

    for min_area in min_area_range:
        for max_area in max_area_range:
            for min_circularity in min_circularity_range:
                for min_convexity in min_convexity_range:
                    for min_inertia in min_inertia_range:
                        params.minArea = min_area
                        params.maxArea = max_area
                        params.minCircularity = min_circularity
                        params.minConvexity = min_convexity
                        params.minInertiaRatio = min_inertia

                        detector = cv2.SimpleBlobDetector_create(params)
                        keypoints = detector.detect(image)

                        if len(keypoints) == target_blob_count:
                            return keypoints, params

    return [], params

def map_blobs_to_point_cloud(keypoints, point_data, img_width=1024, img_height=64):
    points = point_data[:, :3]

    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()

    blob_points = []
    for kp in keypoints:
        x_img, y_img = int(kp.pt[0]), int(kp.pt[1])

        x_normalized = x_img / (img_width - 1)
        y_normalized = y_img / (img_height - 1)

        x_point = x_normalized * (x_max - x_min) + x_min
        y_point = y_normalized * (y_max - y_min) + y_min

        distances = np.sqrt((points[:, 0] - x_point) ** 2 + (points[:, 1] - y_point) ** 2)
        closest_point_index = np.argmin(distances)
        blob_points.append(points[closest_point_index])

    return np.array(blob_points)

def cluster_points(pcd, eps=0.025, min_samples=10):
    points = np.asarray(pcd.points)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    return labels

def find_most_similar_cluster(point_cloud, blob_points, labels):
    unique_labels = set(labels)
    min_distance_sum = float('inf')
    most_similar_label = None

    for label in unique_labels:
        if label == -1:
            continue
        cluster_indices = np.where(labels == label)[0]
        cluster_points = np.asarray(point_cloud.select_by_index(cluster_indices).points)

        distance_sum = 0
        for bp in blob_points:
            distances = np.sqrt((cluster_points[:, 0] - bp[0]) ** 2 +
                                (cluster_points[:, 1] - bp[1]) ** 2 +
                                (cluster_points[:, 2] - bp[2]) ** 2)
            distance_sum += np.min(distances)

        if distance_sum < min_distance_sum:
            min_distance_sum = distance_sum
            most_similar_label = label

    if most_similar_label is not None:
        similar_cluster_indices = np.where(labels == most_similar_label)[0]
        similar_cluster = point_cloud.select_by_index(similar_cluster_indices)
        return similar_cluster
    else:
        return None

def detect_holes_boundaries(projected_points, radius=0.01, min_neighbors=12):
    kdtree = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(projected_points)))
    boundary_points = []

    for i, point in enumerate(projected_points):
        [k, idx, _] = kdtree.search_radius_vector_3d(point, radius)
        if k < min_neighbors:
            boundary_points.append(point)

    boundary_points = np.array(boundary_points)
    return boundary_points

def cluster_boundary_points(boundary_points, eps=0.03, min_samples=5):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(boundary_points)
    labels = clustering.labels_
    return labels

def create_colored_point_cloud(boundary_points, labels):
    unique_labels = set(labels)
    colors = plt.cm.get_cmap("tab10", len(unique_labels))

    colored_points = []
    colored_cloud = o3d.geometry.PointCloud()

    for label in unique_labels:
        if label == -1:
            continue
        cluster_points = boundary_points[labels == label]
        color = colors(label)[:3]
        color_points = np.tile(color, (cluster_points.shape[0], 1))
        colored_points.append(np.hstack((cluster_points, color_points)))

    if colored_points:
        colored_points = np.vstack(colored_points)
        colored_cloud.points = o3d.utility.Vector3dVector(colored_points[:, :3])
        colored_cloud.colors = o3d.utility.Vector3dVector(colored_points[:, 3:])
    
    return colored_cloud

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

def visualize_results(pcd, boundary_cloud, projected_points, original_points):
    # 원래 포인트 클라우드
    pcd.paint_uniform_color([0.5, 0.5, 0.5])  # 회색으로 색칠

    # 경계 포인트 클라우드
    boundary_cloud.paint_uniform_color([1, 0, 0])  # 빨간색으로 색칠

    # 원래 포인트 클라우드와 평면에 사영된 포인트를 시각화
    projected_pcd = o3d.geometry.PointCloud()
    projected_pcd.points = o3d.utility.Vector3dVector(projected_points)
    projected_pcd.paint_uniform_color([0, 1, 0])  # 초록색으로 색칠

    original_pcd = o3d.geometry.PointCloud()
    original_pcd.points = o3d.utility.Vector3dVector(original_points)
    original_pcd.paint_uniform_color([0, 0, 1])  # 파란색으로 색칠

    o3d.visualization.draw_geometries([pcd, boundary_cloud, projected_pcd, original_pcd])

def main():
    rospy.init_node('velodyne_blob_detector', anonymous=True)
    detector = VelodyneBlobDetector()
    
    rospy.spin()
    
    detector.process_point_cloud()

if __name__ == "__main__":
    main()

