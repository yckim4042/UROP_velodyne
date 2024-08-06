#!/usr/bin/env python
import rospy
import numpy as np
import pcl
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from pcl import PointCloud
import math
from scipy import interpolate
from sklearn.neighbors import KDTree

# Publisher
pc_pub = None

# Parameters
maxlen = 100.0
minlen = 0.01
angular_resolution_x = 0.5
angular_resolution_y = 2.1
max_angle_width = 360.0
max_angle_height = 180.0
interpol_value = 20.0
ang_x_lidar = 0.6 * np.pi / 180.0
max_var = 50.0
f_pc = True

# Topics
pcTopic = "/velodyne_points"

def callback(msg_pointCloud):
    if msg_pointCloud is None:
        return

    cloud_in = []
    cloud_out = []
    
    for point in pc2.read_points(msg_pointCloud, skip_nans=True):
        cloud_in.append(point)

    max_z = -float('inf')
    min_z = float('inf')
    max_dis = -float('inf')
    min_dis = float('inf')

    for point in cloud_in:
        distance = math.sqrt(point[0]**2 + point[1]**2)
        if distance < minlen or distance > maxlen:
            continue

        cloud_out.append(point)
        if point[2] > max_z:
            max_z = point[2]
        if point[2] < min_z:
            min_z = point[2]
        if distance > max_dis:
            max_dis = distance
        if distance < min_dis:
            min_dis = distance

    if len(cloud_out) == 0:
        return

    cloud_out = np.array(cloud_out)
    
    width = int(360 / angular_resolution_x)
    height = int(180 / angular_resolution_y)
    
    Z = np.zeros((height, width))
    Zz = np.zeros((height, width))
    ZZei = np.zeros((height, width))

    max_depth = 0.0
    min_depth = -999.0

    for point in cloud_out:
        x, y, z = point[:3]
        r = math.sqrt(x**2 + y**2 + z**2)
        if np.isinf(r) or r < minlen or r > maxlen or np.isnan(z):
            continue
        azimuth = math.atan2(y, x)
        elevation = math.asin(z / r)
        col = int((azimuth + np.pi) / (2 * np.pi) * width)
        row = int((elevation + (np.pi / 2)) / np.pi * height)

        Z[row, col] = r
        Zz[row, col] = z
        ZZei[row, col] = z

        if r > max_depth:
            max_depth = r
        if r < min_depth:
            min_depth = r

    X = np.arange(1, Z.shape[1] + 1)
    Y = np.arange(1, Z.shape[0] + 1)
    XI = np.linspace(X.min(), X.max(), num=int(X.max()))
    YI = np.linspace(Y.min(), Y.max(), num=int(Y.max() * interpol_value))

    interp_Z = interpolate.interp2d(X, Y, Z, kind='linear')
    interp_Zz = interpolate.interp2d(X, Y, Zz, kind='linear')

    ZI = interp_Z(XI, YI)
    ZzI = interp_Zz(XI, YI)

    Zout = ZI.copy()

    for i in range(ZI.shape[0]):
        for j in range(ZI.shape[1]):
            if ZI[i, j] == 0:
                if i + interpol_value < ZI.shape[0]:
                    for k in range(1, int(interpol_value) + 1):
                        Zout[i + k, j] = 0
                if i > interpol_value:
                    for k in range(1, int(interpol_value) + 1):
                        Zout[i - k, j] = 0

    if f_pc:
        for i in range(0, ZI.shape[0] - 1, int(interpol_value)):
            for j in range(ZI.shape[1] - 5):
                promedio = np.mean(ZI[i:i + int(interpol_value), j])
                varianza = np.var(ZI[i:i + int(interpol_value), j])

                if varianza > max_var:
                    Zout[i:i + int(interpol_value), j] = 0

    point_cloud = []
    num_pc = 0

    for i in range(ZI.shape[0] - int(interpol_value)):
        for j in range(ZI.shape[1]):
            ang = np.pi - (2.0 * np.pi * j / ZI.shape[1])
            if Zout[i, j] != 0:
                pc_modulo = Zout[i, j]
                pc_x = np.sqrt(pc_modulo**2 - ZzI[i, j]**2) * np.cos(ang)
                pc_y = np.sqrt(pc_modulo**2 - ZzI[i, j]**2) * np.sin(ang)

                Lidar_matrix = np.array([
                    [np.cos(ang_x_lidar), 0, np.sin(ang_x_lidar)],
                    [0, 1, 0],
                    [-np.sin(ang_x_lidar), 0, np.cos(ang_x_lidar)]
                ])

                result = np.dot(Lidar_matrix, np.array([pc_x, pc_y, ZzI[i, j]]))

                point_cloud.append([result[0], result[1], result[2]])
                num_pc += 1

    header = msg_pointCloud.header
    pc2_out = pc2.create_cloud_xyz32(header, point_cloud)
    pc_pub.publish(pc2_out)

def main():
    global pc_pub
    rospy.init_node('InterpolatedPointCloud', anonymous=True)

    rospy.get_param("/maxlen", maxlen)
    rospy.get_param("/minlen", minlen)
    rospy.get_param("/pcTopic", pcTopic)
    rospy.get_param("/x_resolution", angular_resolution_x)
    rospy.get_param("/y_interpolation", interpol_value)
    rospy.get_param("/ang_Y_resolution", angular_resolution_y)
    rospy.get_param("/ang_ground", ang_x_lidar)
    rospy.get_param("/max_var", max_var)
    rospy.get_param("/filter_output_pc", f_pc)

    rospy.Subscriber(pcTopic, PointCloud2, callback)
    pc_pub = rospy.Publisher("/pc_interpoled", PointCloud2, queue_size=10)

    rospy.spin()

if __name__ == "__main__":
    main()

