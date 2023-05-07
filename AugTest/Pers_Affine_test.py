import torch
import torchvision.transforms.functional as F
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import os
import numpy as np

def perspect_transform(img, vertices, transformation_scale=1.0):
    # 원본 이미지의 크기를 얻어옵니다.
    h, w = img.shape[:2]

    # vertices를 numpy 배열로 변환합니다.
    src_vertices = np.array(vertices, dtype=np.float32).reshape(-1, 1, 2)

    # 변환을 위한 세 개의 대응점을 설정합니다.
    pers_ratio = np.random.randint(9, 31)
    direction_num = np.random.randint(2)

    if direction_num == 1:
        src_points = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float32)
        dst_points = np.array([[0, 0], [w - w/pers_ratio, h/pers_ratio], [0, h], [w - w/pers_ratio, h - h/pers_ratio]], dtype=np.float32)
    else:
        src_points = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float32)
        dst_points = np.array([[w/pers_ratio, h/pers_ratio], [w, 0], [w/pers_ratio, h - h/pers_ratio], [w, h]], dtype=np.float32)        

    # 변환 행렬을 계산합니다.
    perspect_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # 이미지를 변환합니다.
    transformed_img = cv2.warpPerspective(img, perspect_matrix, (int(w * transformation_scale), int(h * transformation_scale)))
    
    # 입력된 vertices를 변환합니다.
    transformed_vertices = cv2.perspectiveTransform(src_vertices, perspect_matrix)
    # 변환된 이미지와 vertices를 반환합니다.
    return transformed_img, transformed_vertices.reshape(-1).tolist()


def affine_transform(img, vertices):
    # 원본 이미지의 크기를 얻어옵니다.
    h, w = img.shape[:2]

    # vertices를 numpy 배열로 변환합니다.
    src_vertices = np.array(vertices, dtype=np.float32).reshape(-1, 1, 2)

    # Affine 변환을 위한 세 개의 대응점을 설정합니다.
    # src_points = np.array([[0,0], [w,0], [w, h]], dtype=np.float32)
    # dst_points = np.array([[0, 0], [w - w/20, w/20], [w, h]], dtype=np.float32)

    affine_ratio = np.random.randint(9, 31)
    direction_num = np.random.randint(2)

    if direction_num == 1:
        src_points = np.array([[0,0], [w,0], [w, h]], dtype=np.float32)
        dst_points = np.array([[0, 0], [w - w/affine_ratio, w/affine_ratio], [w, h]], dtype=np.float32)
    
    else:
        src_points = np.array([[0,0], [w,0], [w, h]], dtype=np.float32)
        dst_points = np.array([[w/affine_ratio, h/affine_ratio], [w, 0], [w - w/affine_ratio, h - h/affine_ratio]], dtype=np.float32)

    # Affine 변환 행렬을 계산합니다.
    affine_matrix = cv2.getAffineTransform(src_points, dst_points)

    # 이미지를 변환합니다.
    transformed_img = cv2.warpAffine(img, affine_matrix, (int(w), int(h)))

    # 입력된 vertices를 변환합니다.
    transformed_vertices = cv2.transform(src_vertices, affine_matrix)

    # 변환된 이미지와 vertices를 반환합니다.
    return transformed_img, transformed_vertices.reshape(-1).tolist()

    
def show(img, vertices):
    vertices_array = np.array(vertices, dtype=np.float32).reshape(-1, 2)

    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Polygon 객체를 생성하고 소수점 좌표를 사용합니다.
    poly = Polygon(vertices_array, edgecolor=(1, 0, 0), fill=None, linewidth=1)
    ax.add_patch(poly)

    plt.title(f"Image with BBox")
    plt.savefig('image_pers_affin.jpg', dpi=2000)
    plt.show()

img = cv2.imread('test_00001.jpg')
vertice = [638, 576, 950, 576, 950, 623, 638, 623]  # 638, 576, 950, 623
img, vertice = perspect_transform(img, vertice)
img, vertice = affine_transform(img, vertice)
# print(f"img : {img}, vertice : {vertice}")
show(img, vertice)