import rosbag
import rospy
from sensor_msgs.msg import Image
import numpy as np

def image_to_text(image_msg):
    # Convert ROS Image message to numpy array
    height = image_msg.height
    width = image_msg.width
    encoding = image_msg.encoding

    print(f"Height: {height}, Width: {width}, Encoding: {encoding}")

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

    return np_arr

def save_to_text_file(filename, data):
    with open(filename, 'w') as file:
        for line in data:
            file.write(' '.join(map(str, line)) + '\n')

def extract_images_from_bag(bag_file, range_image_file, reflec_image_file):
    bag = rosbag.Bag(bag_file, 'r')
    
    range_image_data = []
    reflec_image_data = []

    for topic, msg, t in bag.read_messages(topics=['/ouster/range_image', '/ouster/reflec_image']):
        if topic == '/ouster/range_image':
            range_image = image_to_text(msg)
            range_image_data.extend(range_image)

        if topic == '/ouster/reflec_image':
            reflec_image = image_to_text(msg)
            reflec_image_data.extend(reflec_image)

    save_to_text_file(range_image_file, range_image_data)
    save_to_text_file(reflec_image_file, reflec_image_data)

    bag.close()

if __name__ == "__main__":
    bag_file = 'your_rosbag_file.bag'
    range_image_file = 'range_image.txt'
    reflec_image_file = 'reflec_image.txt'
    extract_images_from_bag(bag_file, range_image_file, reflec_image_file)

