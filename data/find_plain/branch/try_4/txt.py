import rospy
from sensor_msgs.msg import Image
import numpy as np
from cv_bridge import CvBridge
import os

bridge = CvBridge()

def image_to_text(image_msg):
    height = image_msg.height
    width = image_msg.width
    encoding = image_msg.encoding

    if encoding == "mono8":
        np_arr = np.frombuffer(image_msg.data, dtype=np.uint8).reshape(height, width)
    elif encoding == "mono16":
        np_arr = np.frombuffer(image_msg.data, dtype=np.uint16).reshape(height, width)
    elif encoding == "rgb8":
        np_arr = np.frombuffer(image_msg.data, dtype=np.uint8).reshape(height, width, 3)
    elif encoding == "rgba8":
        np_arr = np.frombuffer(image_msg.data, dtype=np.uint8).reshape(height, width, 4)
    else:
        raise ValueError(f"Unsupported encoding: {encoding}")

    np_arr = np.flipud(np_arr)
    return np_arr

def save_to_text_file(filename, data):
    with open(filename, 'w') as file:
        for row in data:
            file.write(' '.join(map(str, row)) + '\n')

def range_image_callback(msg):
    range_image = image_to_text(msg)
    save_to_text_file('range_image.txt', range_image)

def reflectivity_image_callback(msg):
    reflectivity_image = image_to_text(msg)
    save_to_text_file('reflectivity_image.txt', reflectivity_image)

def main():
    rospy.init_node('image_saver', anonymous=True)

    rospy.Subscriber("/range_image", Image, range_image_callback)
    rospy.Subscriber("/reflectivity_image", Image, reflectivity_image_callback)
    
    rospy.spin()

if __name__ == "__main__":
    main()

