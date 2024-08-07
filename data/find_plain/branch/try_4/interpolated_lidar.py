#!/usr/bin/env python

import rospy
import math
import numpy as np
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from std_msgs.msg import Header
from scipy.interpolate import griddata
from geometry_msgs.msg import Point32

maxlen = 100.0
minlen = 0.01
max_FOV = 3.0
min_FOV = 0.4

angular_resolution_x = 0.5
angular_resolution_y = 2.1
max_angle_width = 360.0
max_angle_height = 180.0

interpol_value = 20.0
ang_x_lidar = 0.6 * math.pi / 180.0
max_var = 50.0
f_pc = True

pcTopic = "/velodyne_points"

def callback(msg_pointCloud):
    if not msg_pointCloud:
        return

    points_list = []
    for point in point_cloud2.read_points(msg_pointCloud, field_names=("x", "y", "z", "intensity"), skip_nans=True):
        points_list.append([point[0], point[1], point[2], point[3]])
    
    points_array = np.array(points_list)
    cloud_in = points_array[:, :3]
    intensities = points_array[:, 3]

    max_z, min_z = 0, float('inf')
    max_dis, min_dis = 0, float('inf')

    filtered_points = []
    for point in points_array:
        distance = np.linalg.norm(point[:2])
        if distance < minlen or distance > maxlen:
            continue

        filtered_points.append(point)
        max_z = max(max_z, point[2])
        min_z = min(min_z, point[2])
        max_dis = max(max_dis, distance)
        min_dis = min(min_dis, distance)

    filtered_points = np.array(filtered_points)
    if filtered_points.shape[0] == 0:
        return

    cloud_out = filtered_points[:, :3]
    intensities_out = filtered_points[:, 3]

    # 포인트 클라우드에서 reflectivity 이미지 생성
    rows_img = int(max_angle_height / angular_resolution_y)
    cols_img = int(max_angle_width / angular_resolution_x)
    
    x = np.linspace(0, cols_img - 1, cols_img)
    y = np.linspace(0, rows_img - 1, rows_img)
    XI, YI = np.meshgrid(x, y)

    points = np.zeros((cloud_out.shape[0], 2))
    points[:, 0] = ((np.arctan2(cloud_out[:, 1], cloud_out[:, 0]) + np.pi) / (2 * np.pi)) * cols_img
    points[:, 1] = ((np.arctan2(cloud_out[:, 2], np.linalg.norm(cloud_out[:, :2], axis=1)) + (np.pi / 2)) / np.pi) * rows_img

    Z = griddata(points, intensities_out, (XI, YI), method='linear', fill_value=0)

    # 보간된 이미지를 사용하여 포인트 클라우드 재구성
    point_cloud = []
    intensities_list = []  # intensity 정보를 저장할 리스트

    for i in range(rows_img):
        for j in range(cols_img):
            if Z[i, j] == 0:
                continue

            theta = (j / cols_img) * (2 * np.pi) - np.pi
            phi = (i / rows_img) * np.pi - (np.pi / 2)
            r = Z[i, j]

            x = r * np.cos(phi) * np.cos(theta)
            y = r * np.cos(phi) * np.sin(theta)
            z = r * np.sin(phi)

            Lidar_matrix = np.array([[np.cos(ang_x_lidar), 0, np.sin(ang_x_lidar)],
                                 [0, 1, 0],
                                 [-np.sin(ang_x_lidar), 0, np.cos(ang_x_lidar)]])
            result = np.dot(Lidar_matrix, np.array([x, y, z]))

            point_cloud.append([result[0], result[1], result[2]])
            intensities_list.append(r)  # intensity 정보 추가

# 이제 intensity 정보를 포함하여 포인트 클라우드를 퍼블리시
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "velodyne"
    pc_with_intensity = point_cloud2.create_cloud(header, fields=[
    point_cloud2.PointField('x', 0, point_cloud2.PointField.FLOAT32, 1),
    point_cloud2.PointField('y', 4, point_cloud2.PointField.FLOAT32, 1),
    point_cloud2.PointField('z', 8, point_cloud2.PointField.FLOAT32, 1),
    point_cloud2.PointField('intensity', 12, point_cloud2.PointField.FLOAT32, 1)
], points=[(p[0], p[1], p[2], i) for p, i in zip(point_cloud, intensities_list)])

    pc_pub.publish(pc_with_intensity)


def main():
    global pc_pub

    rospy.init_node("InterpolatedPointCloud")
    
    global maxlen, minlen, pcTopic, angular_resolution_x, interpol_value
    global angular_resolution_y, ang_x_lidar, max_var, f_pc

    maxlen = rospy.get_param("~maxlen", maxlen)
    minlen = rospy.get_param("~minlen", minlen)
    pcTopic = rospy.get_param("~pcTopic", pcTopic)
    angular_resolution_x = rospy.get_param("~x_resolution", angular_resolution_x)
    interpol_value = rospy.get_param("~y_interpolation", interpol_value)
    angular_resolution_y = rospy.get_param("~ang_Y_resolution", angular_resolution_y)
    ang_x_lidar = rospy.get_param("~ang_ground", ang_x_lidar)
    max_var = rospy.get_param("~max_var", max_var)
    f_pc = rospy.get_param("~filter_output_pc", f_pc)

    pc_pub = rospy.Publisher("/pc_interpoled", PointCloud2, queue_size=10)
    rospy.Subscriber(pcTopic, PointCloud2, callback)
    rospy.spin()

if __name__ == "__main__":
    main()

