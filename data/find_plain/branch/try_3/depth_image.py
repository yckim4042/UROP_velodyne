#!/usr/bin/env python
import rospy
import numpy as np
import pcl
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import PointCloud2, Image
from pcl import PointCloud
import sensor_msgs.point_cloud2 as pc2
import math

# Publisher
imgD_pub = None
rngSpheric = None
coordinate_frame = pcl.RangeImage.LASER_FRAME

# Parameters
maxlen = 500
minlen = 0.1
angular_resolution_x = 0.25
angular_resolution_y = 0.85
max_angle_width = 360.0
max_angle_height = 360.0
pcTopic = "/velodyne_points"

bridge = CvBridge()

def callback(msg_pointCloud):
    global rngSpheric
    
    cloud_in = []
    cloud_out = []

    for point in pc2.read_points(msg_pointCloud, skip_nans=True):
        cloud_in.append(point)

    for point in cloud_in:
        distance = math.sqrt(point[0]**2 + point[1]**2)
        if distance < minlen or distance > maxlen:
            continue
        cloud_out.append(point)

    cloud_out = np.array(cloud_out, dtype=np.float32)
    cloud_out = pcl.PointCloud(cloud_out)

    rngSpheric.create_from_point_cloud(cloud_out, np.deg2rad(angular_resolution_x), np.deg2rad(angular_resolution_y),
                                       np.deg2rad(max_angle_height), np.deg2rad(max_angle_height),
                                       pcl.Affine3f.Identity(), coordinate_frame, 0.0, 0.0, 0)

    rngSpheric.header.frame_id = msg_pointCloud.header.frame_id
    rngSpheric.header.stamp = msg_pointCloud.header.stamp

    cols = rngSpheric.width
    rows = rngSpheric.height
    depth_image = np.zeros((rows, cols), dtype=np.uint16)

    for i in range(cols):
        for j in range(rows):
            r = rngSpheric.get_point(i, j).range
            if np.isinf(r) or r < minlen or r > maxlen:
                continue
            range_value = 1 - (pow(2, 16) / (maxlen - minlen)) * (r - minlen)
            depth_image[j, i] = range_value

    image_msg = bridge.cv2_to_imgmsg(depth_image, encoding="mono16")
    imgD_pub.publish(image_msg)

def main():
    global imgD_pub, rngSpheric
    rospy.init_node('pointCloud2depthImage', anonymous=True)

    rospy.get_param("/maxlen", maxlen)
    rospy.get_param("/minlen", minlen)
    rospy.get_param("/angular_resolution_x", angular_resolution_x)
    rospy.get_param("/angular_resolution_y", angular_resolution_y)
    rospy.get_param("/max_angle_width", max_angle_width)
    rospy.get_param("/max_angle_height", max_angle_height)
    rospy.get_param("/pcTopic", pcTopic)

    rngSpheric = pcl.RangeImageSpherical()
    imgD_pub = rospy.Publisher("/depth_image", Image, queue_size=10)
    rospy.Subscriber(pcTopic, PointCloud2, callback)
    rospy.spin()

if __name__ == "__main__":
    main()

