#!/usr/bin/env python

import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2, Image
from sensor_msgs import point_cloud2
from cv_bridge import CvBridge
import cv2
import math

class VelodyneImageProcessor:
    def __init__(self, frame_id):
        self.frame = frame_id
        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher("/reflectivity_image", Image, queue_size=10)
        
        self.maxlen = rospy.get_param("/maxlen", 500)
        self.minlen = rospy.get_param("/minlen", 0.1)
        self.angular_resolution_x = rospy.get_param("/angular_resolution_x", 0.25)
        self.angular_resolution_y = rospy.get_param("/angular_resolution_y", 0.85)
        self.max_angle_width = rospy.get_param("/max_angle_width", 360.0)
        self.max_angle_height = rospy.get_param("/max_angle_height", 360.0)

    def callback(self, msg):
        cloud_in = []
        for point in point_cloud2.read_points(msg, skip_nans=True, field_names=("x", "y", "z", "intensity")):
            cloud_in.append(point)

        cloud_out = []
        for point in cloud_in:
            distance = math.sqrt(point[0]**2 + point[1]**2)
            if distance < self.minlen or distance > self.maxlen:
                continue
            cloud_out.append(point)

        if len(cloud_out) == 0:
            return

        # Convert point cloud to reflectivity image
        H = int(self.max_angle_height / self.angular_resolution_y)
        W = int(self.max_angle_width / self.angular_resolution_x)
        reflectivity_image = np.zeros((H, W), dtype=np.uint16)

        for point in cloud_out:
            x, y, z, intensity = point[:4]
            r = math.sqrt(x**2 + y**2 + z**2)
            azimuth = (math.atan2(y, x) + np.pi) * (W / (2 * np.pi))
            elevation = (math.asin(z / r) + (np.pi / 2)) * (H / np.pi)
            col = int(azimuth)
            row = int(elevation)
            if 0 <= row < H and 0 <= col < W:
                reflectivity_value = int(intensity)
                reflectivity_image[row, col] = reflectivity_value

        # 180도 회전
        reflectivity_image = np.flipud(reflectivity_image)  # 상하 반전
        reflectivity_image = np.fliplr(reflectivity_image)  # 좌우 반전

        image_msg = self.bridge.cv2_to_imgmsg(reflectivity_image, encoding="mono16")
        image_msg.header.frame_id = self.frame
        image_msg.header.stamp = rospy.Time.now()
        self.image_pub.publish(image_msg)

def main():
    rospy.init_node('velodyne_image_processor', anonymous=True)

    frame_id = rospy.get_param("~frame_id", "velodyne")
    processor = VelodyneImageProcessor(frame_id)

    pc_topic = rospy.get_param("/pcTopic", "/pc_interpoled")
    rospy.Subscriber(pc_topic, PointCloud2, processor.callback)
    rospy.spin()

if __name__ == "__main__":
    main()
