import numpy as np
import cv2

def load_data(filename):
    with open(filename, 'r') as file:
        data = file.readlines()

    data = [list(map(int, line.split())) for line in data]
    return np.array(data)

def create_image(data):
    # 모든 데이터를 사용하여 이미지 생성
    num_rows = data.shape[0]
    num_cols = 144  # 주어진 가로 길이
    image_data = np.zeros((num_rows, num_cols), dtype=np.uint16)

    for i in range(num_rows):
        image_data[i, :] = data[i][:num_cols]

    # 이미지 값을 0~255 범위로 정규화 (8비트 이미지로 저장하기 위함)
    normalized_image = cv2.normalize(image_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return normalized_image

def save_image(image, filename):
    cv2.imwrite(filename, image)

if __name__ == "__main__":
    data = load_data('./reflectivity_image.txt')
    image = create_image(data)
    save_image(image, './fixed_reflectivity_image.png')

