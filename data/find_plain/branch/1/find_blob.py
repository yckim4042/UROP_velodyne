#!/usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import cv2
from cv_bridge import CvBridge

class VelodyneBlobAnalyzer:
    def __init__(self):
        self.bridge = CvBridge()
        self.point_cloud_sub = rospy.Subscriber("/velodyne_points", PointCloud2, self.point_cloud_callback)
        self.reflec_image = None
        self.point_cloud_data = None
        self.img_width, self.img_height = 1024, 64

    def point_cloud_callback(self, msg):
        rospy.loginfo("Received point cloud")
        self.reflec_image, self.point_cloud_data = self.point_cloud_to_image_and_data(msg)
        if self.reflec_image is not None:
            self.find_and_draw_blobs(self.reflec_image)

    def point_cloud_to_image_and_data(self, msg):
        points = []
        reflectivity = []
        point_data = []

        for point in pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True):
            x, y, z, intensity = point
            points.append((x, y, z))
            reflectivity.append(intensity)
            point_data.append((x, y, z, intensity))

        points = np.array(points)
        reflectivity = np.array(reflectivity)
        point_data = np.array(point_data)

        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()

        x_normalized = (points[:, 0] - x_min) / (x_max - x_min)
        y_normalized = (points[:, 1] - y_min) / (y_max - y_min)

        x_img = (x_normalized * (self.img_width - 1)).astype(np.int32)
        y_img = (y_normalized * (self.img_height - 1)).astype(np.int32)

        reflec_image = np.zeros((self.img_height, self.img_width), dtype=np.float32)

        reflec_image[y_img, x_img] = reflectivity

        return reflec_image, point_data

    def find_and_draw_blobs(self, image):
        # 블랍 탐지 파라미터 설정
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

        if keypoints:
            self.draw_blob_boundaries(image, keypoints)

    def draw_blob_boundaries(self, image, keypoints):
        for keypoint in keypoints:
            center = keypoint.pt
            radius = keypoint.size / 2
            boundary_points = self.find_blob_boundary_points(center, radius, image.shape)

            for point in boundary_points:
                x, y = point
                if 0 <= x < self.img_width and 0 <= y < self.img_height:
                    # Find 3D coordinates of the boundary point
                    x_3d, y_3d, z_3d = self.find_3d_coordinates(x, y)
                    rospy.loginfo(f"Boundary Point: ({x}, {y}), 3D Coordinates: ({x_3d}, {y_3d}, {z_3d})")
                image[point[1], point[0]] = 255

        im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0, 0, 255),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        output_file_with_blobs = 'output_image_with_blobs.png'
        cv2.imwrite(output_file_with_blobs, im_with_keypoints)
        rospy.loginfo(f'Blob boundary image saved as {output_file_with_blobs}')

    def find_blob_boundary_points(self, blob_center, blob_radius, image_shape):
        boundary_points = []
        x_center, y_center = int(blob_center[0]), int(blob_center[1])
        for angle in range(0, 360):
            theta = np.deg2rad(angle)
            x = x_center + int(blob_radius * np.cos(theta))
            y = y_center + int(blob_radius * np.sin(theta))
            if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
                boundary_points.append((x, y))
        return boundary_points

    def find_3d_coordinates(self, x, y):
        # x와 y는 이미지 좌표 (픽셀 좌표)
        normalized_x = x / (self.img_width - 1)
        normalized_y = y / (self.img_height - 1)

        x_min, x_max = self.point_cloud_data[:, 0].min(), self.point_cloud_data[:, 0].max()
        y_min, y_max = self.point_cloud_data[:, 1].min(), self.point_cloud_data[:, 1].max()

        x_3d = normalized_x * (x_max - x_min) + x_min
        y_3d = normalized_y * (y_max - y_min) + y_min

        # 해당 (x, y) 픽셀에 해당하는 가장 가까운 3D 포인트 찾기
        distances = np.sqrt((self.point_cloud_data[:, 0] - x_3d) ** 2 +
                            (self.point_cloud_data[:, 1] - y_3d) ** 2)
        closest_point_index = np.argmin(distances)
        z_3d = self.point_cloud_data[closest_point_index, 2]

        return x_3d, y_3d, z_3d

if __name__ == "__main__":
    rospy.init_node("velodyne_blob_analyzer")
    analyzer = VelodyneBlobAnalyzer()

    rospy.spin()

