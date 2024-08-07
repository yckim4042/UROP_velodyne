import numpy as np
import cv2

def load_data(filename):
    with open(filename, 'r') as file:
        data = file.readlines()

    data = [list(map(int, line.split())) for line in data]
    return np.array(data)

def create_image(data):
    # 유효한 줄 찾기 (0이 아닌 값이 있는 줄)
    valid_rows = [i + 1 for i in range(data.shape[0]) if np.any(data[i] != 0)]
    print(valid_rows)
    
    # 유효한 행들의 데이터 추출
    valid_data = data[valid_rows]
    
    # 각 유효한 행 사이의 빈 행 수 계산
    row_gaps = np.diff(valid_rows)
    print(row_gaps)
    
    # 새로운 이미지의 높이 계산 (유효한 행들 사이의 평균 빈 행 수를 사용하여 이미지 확장)
    new_height = int(np.sum(row_gaps) / len(row_gaps) * len(valid_rows))
    new_data = np.zeros((new_height, data.shape[1]), dtype=np.uint16)
    
    # 행 사이에 비례하여 값 채우기
    current_row = 0
    for i in range(len(valid_rows) - 1):
        start_row = current_row
        end_row = current_row + row_gaps[i]
        
        for j in range(row_gaps[i]):
            interpolation_factor = j / row_gaps[i]
            new_data[start_row + j] = (1 - interpolation_factor) * valid_data[i] + interpolation_factor * valid_data[i + 1]
        
        current_row = end_row

    # 마지막 유효 행 복사
    new_data[current_row:] = valid_data[-1]
    
    return new_data

def save_image(image, filename):
    cv2.imwrite(filename, image)

if __name__ == "__main__":
    data = load_data('reflectivity_image.txt')
    image = create_image(data)
    save_image(image, 'fixed_reflectivity_image.png')

