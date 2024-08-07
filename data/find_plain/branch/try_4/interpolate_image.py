#!/usr/bin/env python

import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2, Image
from sensor_msgs import point_cloud2
from cv_bridge import CvBridge
import math
from scipy.interpolate import griddata
import cv2

class VelodyneImageProcessor:
    def __init__(self, frame_id):
        self.frame = frame_id
        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher("/reflectivity_image", Image, queue_size=10)
        
        self.maxlen = rospy.get_param("/maxlen", 100.0)
        self.minlen = rospy.get_param("/minlen", 0.01)
        self.angular_resolution_x = rospy.get_param("/x_resolution", 0.5)
        self.angular_resolution_y = rospy.get_param("/ang_Y_resolution", 2.1)
        self.max_angle_width = rospy.get_param("/max_angle_width", 360.0)
        self.max_angle_height = rospy.get_param("/max_angle_height", 180.0)
        self.interpol_value = rospy.get_param("/y_interpolation", 20.0)
        self.ang_x_lidar = rospy.get_param("/ang_ground", 0.6 * math.pi / 180.0)
        self.max_var = rospy.get_param("/max_var", 50.0)
        self.f_pc = rospy.get_param("/filter_output_pc", True)

    def callback(self, msg):
        cloud_points = []
        for point in point_cloud2.read_points(msg, skip_nans=True, field_names=("x", "y", "z", "intensity")):
            cloud_points.append([point[0], point[1], point[2], point[3]])

        cloud_points = np.array(cloud_points, dtype=np.float32)

        # Filter points based on distance
        filtered_points = []
        for point in cloud_points:
            distance = math.sqrt(point[0]**2 + point[1]**2)
            if self.minlen <= distance <= self.maxlen:
                filtered_points.append(point)
        
        if len(filtered_points) == 0:
            return
        
        filtered_points = np.array(filtered_points)

        # Convert to point cloud
        H = int(self.max_angle_height / self.angular_resolution_y)
        W = int(self.max_angle_width / self.angular_resolution_x)
        reflectivity_image = np.zeros((H, W), dtype=np.float32)

        for point in filtered_points:
            x, y, z, intensity = point
            r = math.sqrt(x**2 + y**2 + z**2)
            azimuth = (math.atan2(y, x) + np.pi) * (W / (2 * np.pi))
            elevation = (math.asin(z / r) + (np.pi / 2)) * (H / np.pi)
            col = int(azimuth)
            row = int(elevation)
            if 0 <= row < H and 0 <= col < W:
                reflectivity_image[row, col] = intensity

        # Interpolation
        points = np.indices(reflectivity_image.shape).reshape(2, -1).T
        values = reflectivity_image.flatten()
        valid = values > 0
        points_valid = points[valid]
        values_valid = values[valid]

        grid_x, grid_y = np.mgrid[0:H, 0:W]
        interpolated_image = griddata(points_valid, values_valid, (grid_x, grid_y), method='linear', fill_value=0)

        # 180도 회전 및 좌우 반전
        interpolated_image = np.flipud(interpolated_image)
        interpolated_image = np.fliplr(interpolated_image)

        image_msg = self.bridge.cv2_to_imgmsg(interpolated_image.astype(np.uint16), encoding="mono16")
        image_msg.header.frame_id = self.frame
        image_msg.header.stamp = rospy.Time.now()
        self.image_pub.publish(image_msg)

def main():
    rospy.init_node('velodyne_image_processor', anonymous=True)

    frame_id = rospy.get_param("~frame_id", "velodyne")
    processor = VelodyneImageProcessor(frame_id)

    pc_topic = rospy.get_param("/pcTopic", "/velodyne_points")
    rospy.Subscriber(pc_topic, PointCloud2, processor.callback)
    rospy.spin()

if __name__ == "__main__":
    main()

