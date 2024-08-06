#!/usr/bin/env python

import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, Image
from cv_bridge import CvBridge
import numpy as np

class PointCloudToReflectImage:
    def __init__(self):
        rospy.init_node('pointCloud2ReflectImage', anonymous=True)

        self.maxlen = rospy.get_param('~maxlen', 500.0)
        self.minlen = rospy.get_param('~minlen', 0.1)
        self.pcTopic = rospy.get_param('~pcTopic', '/velodyne_points')

        self.imgR_pub = rospy.Publisher('/reflectivity_image', Image, queue_size=10)
        self.bridge = CvBridge()
        self.sub = rospy.Subscriber(self.pcTopic, PointCloud2, self.callback)

    def callback(self, msg_pointCloud):
        points_list = []

        for point in pc2.read_points(msg_pointCloud, field_names=("x", "y", "z", "intensity"), skip_nans=True):
            x, y, z, intensity = point
            distance = np.sqrt(x * x + y * y)
            if self.minlen < distance < self.maxlen:
                points_list.append((x, y, intensity))

        if not points_list:
            return

        points_array = np.array(points_list)
        x_coords = points_array[:, 0]
        y_coords = points_array[:, 1]
        intensities = points_array[:, 2]

        # Create a 2D histogram to map points to image grid
        x_edges = np.linspace(np.min(x_coords), np.max(x_coords), 512)
        y_edges = np.linspace(np.min(y_coords), np.max(y_coords), 32)
        
        reflect_image, _, _ = np.histogram2d(x_coords, y_coords, bins=(x_edges, y_edges), weights=intensities)

        # Normalize intensity values to 0-65535 for mono16 encoding
        reflect_image = np.nan_to_num(reflect_image)  # Replace NaNs with 0
        normalized_reflect_image = np.uint16(reflect_image / np.max(reflect_image) * 65535)

        # Publish reflectivity image
        image_msg = self.bridge.cv2_to_imgmsg(normalized_reflect_image, encoding="mono16")
        self.imgR_pub.publish(image_msg)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    converter = PointCloudToReflectImage()
    converter.run()

