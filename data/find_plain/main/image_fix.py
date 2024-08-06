import numpy as np
import cv2
import open3d as o3d

def load_reflectivity_coordinates(filename):
    coordinates = []
    with open(filename, "r") as f:
        for line in f:
            x_img, y_img, x, y, z = map(float, line.strip().split(","))
            coordinates.append([x_img, y_img, x, y, z])
    return np.array(coordinates)

def align_reflectivity_image(point_data, img_width=1024, img_height=64):
    points = point_data[:, 2:5]  # x, y, z 좌표만 사용
    reflectivity = point_data[:, 1]

    # 포인트 정렬
    points_sorted = points[np.argsort(points[:, 0])]

    x_min, x_max = points_sorted[:, 0].min(), points_sorted[:, 0].max()
    y_min, y_max = points_sorted[:, 1].min(), points_sorted[:, 1].max()

    x_normalized = (points_sorted[:, 0] - x_min) / (x_max - x_min)
    y_normalized = (points_sorted[:, 1] - y_min) / (y_max - y_min)

    x_img = (x_normalized * (img_width - 1)).astype(np.int32)
    y_img = (y_normalized * (img_height - 1)).astype(np.int32)

    reflec_image = np.zeros((img_height, img_width), dtype=np.float32)
    reflec_image[y_img, x_img] = reflectivity

    # 8비트로 변환
    reflec_image = cv2.normalize(reflec_image, None, 0, 255, cv2.NORM_MINMAX)
    reflec_image = reflec_image.astype(np.uint8)

    return reflec_image

def save_aligned_image(reflectivity_coords_file, output_image_file):
    # 반사율 좌표 로드
    reflectivity_coords = load_reflectivity_coordinates(reflectivity_coords_file)

    # 반사율 이미지 정렬
    reflec_image = align_reflectivity_image(reflectivity_coords)

    # 정렬된 반사율 이미지 저장
    cv2.imwrite(output_image_file, reflec_image)

def main():
    # 반사율 좌표 파일 경로
    reflectivity_coords_file = "reflectivity_coordinates.txt"

    # 출력 이미지 파일 경로
    output_image_file = "aligned_reflectivity_image.png"

    # 이미지 보정 및 저장
    save_aligned_image(reflectivity_coords_file, output_image_file)

if __name__ == "__main__":
    main()

