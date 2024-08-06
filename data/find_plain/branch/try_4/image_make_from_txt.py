import numpy as np
import cv2

def load_data(filename):
    with open(filename, 'r') as file:
        data = file.readlines()

    data = [list(map(int, line.split())) for line in data]
    return np.array(data)

def create_image(data):

    
    # 유효한 줄 찾기 (0이 아닌 값이 있는 줄)
    valid_rows = [i for i in range(data.shape[0]) if np.any(data[i] != 0)]
    print(valid_rows)
    valid_data = data[valid_rows]

    # 이미지 크기 키우기
    scale_factor = 2  # 원하는 비율로 크기 조정
    scaled_data = np.zeros((valid_data.shape[0] * scale_factor, valid_data.shape[1]), dtype=np.uint16)

    for i in range(valid_data.shape[0]):
        scaled_data[i * scale_factor : (i + 1) * scale_factor] = np.tile(valid_data[i], (scale_factor, 1))

    return scaled_data

def save_image(image, filename):
    cv2.imwrite(filename, image)

if __name__ == "__main__":
    data = load_data('reflectivity_image.txt')
    image = create_image(data)
    save_image(image, 'fixed_reflectivity_image.png')

