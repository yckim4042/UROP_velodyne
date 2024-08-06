#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d
import numpy as np

def callback(data):
    # Convert PointCloud2 message to a list of points
  try:
    points = []
    for point in pc2.read_points(data, field_names=("x", "y", "z"), skip_nans=True):
        points.append([point[0], point[1], point[2]])

    # Convert to numpy array
    np_points = np.array(points, dtype=np.float64)
    

    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_points)
    
    
    # Save the point cloud to a PCD file
    o3d.io.write_point_cloud("./example_pointcloud_livox.ply", pcd)
    
    rospy.loginfo("Saved point cloud to filtered_pointcloud.pcd")
    rospy.signal_shutdown("Point cloud saved")  # Shutdown the node after saving the data
    
  except Exception as e:
    rospy.logerr(f"Error in callback: {e}")
    
def listener():
    rospy.init_node('save_filtered_pointcloud', anonymous=True)
    rospy.Subscriber("/livox/lidar", PointCloud2, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()

