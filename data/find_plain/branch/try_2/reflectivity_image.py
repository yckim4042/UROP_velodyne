#!/usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import cv2
from cv_bridge import CvBridge
import time

class VelodyneProjection:
    def __init__(self):
        rospy.init_node('velodyne_projection_node', anonymous=True)
        self.point_cloud_sub = rospy.Subscriber("/velodyne_points", PointCloud2, self.point_cloud_callback)
        self.bridge = CvBridge()
        
        self.img_height = 64  # 이미지 높이를 64로 설정
        self.img_width = 1024  # 이미지 너비를 1024로 설정
        self.accumulated_points = []
        self.start_time = None
        self.duration = 3  # 데이터 축적 시간 (초)

        # LiDAR의 최대 측정 범위와 수직 시야각
        measurement_range = 100  # meters
        vertical_fov = 15  # degrees

        # z_max와 z_min 계산
        self.z_max = measurement_range * np.sin(np.radians(vertical_fov))
        self.z_min = measurement_range * np.sin(np.radians(-vertical_fov))

        rospy.loginfo(f"z_max: {self.z_max}, z_min: {self.z_min}")

    def point_cloud_callback(self, msg):
        if self.start_time is None:
            self.start_time = time.time()

        current_time = time.time()
        elapsed_time = current_time - self.start_time

        if elapsed_time <= self.duration:
            for point in pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True):
                self.accumulated_points.append([point[0], point[1], point[2], point[3]])
        else:
            self.point_cloud_sub.unregister()  # 데이터 축적이 완료되면 구독 해제
            rospy.loginfo("Data accumulation complete")
            self.create_and_save_image()
            rospy.signal_shutdown("Data processing complete")

    def create_and_save_image(self):
        if not self.accumulated_points:
            rospy.logwarn("No point cloud data accumulated")
            return

        point_cloud_data = np.array(self.accumulated_points)
        reflec_image = self.create_reflectivity_image(point_cloud_data)
        
        # 이미지를 PNG 파일로 저장
        cv2.imwrite("reflectivity_image.png", reflec_image)

        rospy.loginfo("Reflectivity image saved as 'reflectivity_image.png'")

    def create_reflectivity_image(self, point_data):
        points = point_data[:, :3]
        reflectivity = point_data[:, 3]

        z_normalized = (points[:, 2] - self.z_min) / (self.z_max - self.z_min)
        z_img = (z_normalized * (self.img_height - 1)).astype(np.int32)

        theta = np.arctan2(points[:, 1], points[:, 0])
        theta = np.where(theta < 0, theta + 2 * np.pi, theta)  # [0, 2π]로 변환
        theta_normalized = theta / (2 * np.pi)
        x_img = (theta_normalized * (self.img_width - 1)).astype(np.int32)

        reflec_image = np.zeros((self.img_height, self.img_width), dtype=np.uint8)
        reflec_image[z_img, x_img] = (reflectivity / reflectivity.max() * 255).astype(np.uint8)

        return reflec_image

if __name__ == "__main__":
    try:
        VelodyneProjection()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

