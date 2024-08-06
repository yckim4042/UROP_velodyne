#!/usr/bin/env python

import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
import numpy as np
import cv2

def callback(point_cloud_msg):
    # Convert PointCloud2 to a list of points
    points_list = []
    for point in pc2.read_points(point_cloud_msg, skip_nans=True):
        points_list.append([point[0], point[1], point[2], point[3]])  # x, y, z, intensity

    # Extract intensity values
    intensities = np.array([point[3] for point in points_list])

    # Normalize intensity values to the range 0-255 for visualization
    intensity_image = cv2.normalize(intensities, None, 0, 255, cv2.NORM_MINMAX)
    intensity_image = intensity_image.astype(np.uint8)

    # Create an image of size (height, width) to visualize the intensities
    height = 64  # Choose an appropriate height for visualization
    width = int(len(intensities) / height)  # Calculate width based on number of points and chosen height

    # If the number of points is not divisible by the chosen height, pad the intensities array
    if len(intensities) % height != 0:
        padded_length = (height * (width + 1)) - len(intensities)
        intensity_image = np.pad(intensity_image, (0, padded_length), mode='constant', constant_values=0)
        width += 1

    # Reshape the intensity values into an image
    intensity_image = intensity_image.reshape((height, width))

    # Display the intensity image
    cv2.imshow('Intensity Image', intensity_image)
    cv2.waitKey(1)

if __name__ == '__main__':
    rospy.init_node('velodyne_intensity_image_node', anonymous=True)
    rospy.Subscriber('/velodyne_points', PointCloud2, callback)
    rospy.spin()

