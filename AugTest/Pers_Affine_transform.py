import torch
import torchvision.transforms.functional as F
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import math
import os
import numpy as np

def affine_transform(img, vertices):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    src_vertices = np.array(vertices, dtype=np.float32).reshape(-1, 1, 2)
    
    affine_ratio = np.random.randint(9, 31)
    direction_num = np.random.randint(2)

    if direction_num == 1:
        src_points = np.array([[0,0], [w,0], [w, h]], dtype=np.float32)
        dst_points = np.array([[0, 0], [w - w/affine_ratio, w/affine_ratio], [w, h]], dtype=np.float32)
    
    else:
        src_points = np.array([[0,0], [h,0], [w, h]], dtype=np.float32)
        dst_points = np.array([[0, 0], [h - h/affine_ratio, h/affine_ratio], [w, h]], dtype=np.float32)

    affine_matrix = cv2.getAffineTransform(src_points, dst_points)

    transformed_img = cv2.warpAffine(img, affine_matrix, (int(w), int(h)))
    transformed_vertices = cv2.transform(src_vertices, affine_matrix)

    transformed_img_pil = Image.fromarray(cv2.cvtColor(transformed_img, cv2.COLOR_BGR2RGB))
    return transformed_img_pil, transformed_vertices.reshape(-1).tolist()

def perspect_transform(img, vertices):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    src_vertices = np.array(vertices, dtype=np.float32).reshape(-1, 1, 2)

    pers_ratio = np.random.randint(8, 31)
    direction_num = np.random.randint(2)

    if direction_num == 1:
        src_points = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float32)
        dst_points = np.array([[0, 0], [w - w/pers_ratio, h/pers_ratio], [0, h], [w - w/pers_ratio, h - h/pers_ratio]], dtype=np.float32)
    else:
        src_points = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float32)
        dst_points = np.array([[w/pers_ratio, h/pers_ratio], [w, 0], [w/pers_ratio, h - h/pers_ratio], [w, h]], dtype=np.float32)        

    perspect_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    transformed_img = cv2.warpPerspective(img, perspect_matrix, (int(w * transformation_scale), int(h * transformation_scale)))
    transformed_img_pil = Image.fromarray(cv2.cvtColor(transformed_img, cv2.COLOR_BGR2RGB))
    transformed_vertices = cv2.perspectiveTransform(src_vertices, perspect_matrix)

    return transformed_img, transformed_vertices.reshape(-1).tolist()