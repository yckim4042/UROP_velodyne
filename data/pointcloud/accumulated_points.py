#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d
import numpy as np

# Global variable to store accumulated points
accumulated_points = []

def callback(data):
    global accumulated_points
    try:
        # Convert PointCloud2 message to a list of points
        points = []
        for point in pc2.read_points(data, field_names=("x", "y", "z"), skip_nans=True):
            points.append([point[0], point[1], point[2]])

        # Append new points to the accumulated points list
        accumulated_points.extend(points)

        rospy.loginfo("Accumulated {} points".format(len(accumulated_points)))

    except Exception as e:
        rospy.logerr(f"Error in callback: {e}")

def save_point_cloud():
    global accumulated_points
    # Convert to numpy array
    np_points = np.array(accumulated_points, dtype=np.float64)

    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_points)

    # Save the point cloud to a PCD file
    o3d.io.write_point_cloud("./accumulated_pointcloud.ply", pcd)

    rospy.loginfo("Saved point cloud to accumulated_pointcloud.pcd")
    rospy.signal_shutdown("Point cloud saved")  # Shutdown the node after saving the data

def listener():
    rospy.init_node('save_filtered_pointcloud', anonymous=True)
    rospy.Subscriber("/velodyne_points", PointCloud2, callback)

    # Set the duration for how long you want to accumulate points before saving
    save_duration = rospy.Duration(1)  # For example, accumulate for 30 seconds
    rospy.sleep(save_duration)

    save_point_cloud()
    rospy.spin()

if __name__ == '__main__':
    listener()

