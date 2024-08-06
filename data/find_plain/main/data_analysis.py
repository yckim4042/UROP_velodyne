#!/usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import cv2
import open3d as o3d

class VelodyneDataSaver:
    def __init__(self):
        self.point_cloud_sub = rospy.Subscriber("/velodyne_points", PointCloud2, self.point_cloud_callback)
        self.accumulated_points = []
        self.start_time = None
        self.duration = 3  # 데이터 축적 시간 (초)

    def point_cloud_callback(self, msg):
        if self.start_time is None:
            self.start_time = rospy.Time.now()
        
        current_time = rospy.Time.now()
        elapsed_time = (current_time - self.start_time).to_sec()

        if elapsed_time <= self.duration:
            points = []
            for point in pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True):
                points.append([point[0], point[1], point[2], point[3]])
            self.accumulated_points.extend(points)
        else:
            rospy.loginfo("Data accumulation complete")
            self.save_data()
            rospy.signal_shutdown("Data accumulation complete")

    def save_data(self):
        if not self.accumulated_points:
            rospy.logwarn("No point cloud data accumulated")
            return

        # 포인트 클라우드 데이터 저장
        point_cloud_data = np.array(self.accumulated_points)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud_data[:, :3])
        o3d.io.write_point_cloud("accumulated_points.ply", pcd)
        rospy.loginfo("Point cloud saved as 'accumulated_points.ply'")

        # 반사율 이미지 생성 및 저장
        reflec_image = self.create_reflectivity_image(point_cloud_data)
        cv2.imwrite("reflectivity_image.png", reflec_image)
        rospy.loginfo("Reflectivity image saved as 'reflectivity_image.png'")

        # 반사율 이미지의 각 포인트의 3차원 좌표 저장
        self.save_reflectivity_coordinates(point_cloud_data, reflec_image)

    def create_reflectivity_image(self, point_data, img_width=1024, img_height=64):
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

        # 8비트로 변환
        reflec_image = cv2.normalize(reflec_image, None, 0, 255, cv2.NORM_MINMAX)
        reflec_image = reflec_image.astype(np.uint8)

        return reflec_image

    def save_reflectivity_coordinates(self, point_data, reflec_image, img_width=1024, img_height=64):
        points = point_data[:, :3]

        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()

        x_normalized = (points[:, 0] - x_min) / (x_max - x_min)
        y_normalized = (points[:, 1] - y_min) / (y_max - y_min)

        x_img = (x_normalized * (img_width - 1)).astype(np.int32)
        y_img = (y_normalized * (img_height - 1)).astype(np.int32)

        with open("reflectivity_coordinates.txt", "w") as f:
            for i in range(len(x_img)):
                f.write(f"{x_img[i]},{y_img[i]},{points[i, 0]},{points[i, 1]},{points[i, 2]}\n")
        rospy.loginfo("Reflectivity coordinates saved as 'reflectivity_coordinates.txt'")

def main():
    rospy.init_node('velodyne_data_saver', anonymous=True)
    saver = VelodyneDataSaver()
    
    rospy.spin()

if __name__ == "__main__":
    main()

