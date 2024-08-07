#!/usr/bin/env python

import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2, Image
from sensor_msgs import point_cloud2
from cv_bridge import CvBridge
import cv2

# 글로벌 변수 설정
maxlen = 500
minlen = 0.1
angular_resolution_x = 0.25
angular_resolution_y = 0.85
max_angle_width = 360.0
max_angle_height = 360.0
pcTopic = "/pc_interpoled"

# 이미지 퍼블리셔와 CvBridge 초기화
imgI_pub = None
bridge = CvBridge()

def callback(msg_pointCloud):
    global imgI_pub, bridge

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

    intensity_image = np.zeros((rows_img, cols_img), dtype=np.uint8)

    for i in range(points.shape[0]):
        col = int(points[i, 0])
        row = int(points[i, 1])
        if 0 <= col < cols_img and 0 <= row < rows_img:
            intensity_image[row, col] = intensities_out[i]

    image_msg = bridge.cv2_to_imgmsg(intensity_image, encoding="mono8")
    imgI_pub.publish(image_msg)

def main():
    global imgI_pub
    global maxlen, minlen, pcTopic, angular_resolution_x, angular_resolution_y, max_angle_width, max_angle_height


    rospy.init_node("pointCloud2intensityImage")
	
    maxlen = rospy.get_param("~maxlen", maxlen)
    minlen = rospy.get_param("~minlen", minlen)
    pcTopic = rospy.get_param("~pcTopic", pcTopic)
    angular_resolution_x = rospy.get_param("~angular_resolution_x", angular_resolution_x)
    angular_resolution_y = rospy.get_param("~angular_resolution_y", angular_resolution_y)
    max_angle_width = rospy.get_param("~max_angle_width", max_angle_width)
    max_angle_height = rospy.get_param("~max_angle_height", max_angle_height)

    imgI_pub = rospy.Publisher("/intensity_image", Image, queue_size=10)
    rospy.Subscriber(pcTopic, PointCloud2, callback)
    rospy.spin()

if __name__ == "__main__":
    main()

