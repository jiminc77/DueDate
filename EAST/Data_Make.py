from dataset import extract_vertices
from dataset import affine_transform
from dataset import perspect_transform
from dataset import adjust_height
from dataset import rotate_img
from dataset import crop_img_maker
from dataset import rotate_img_and_resize
from dataset import get_score_geo
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
from PIL import Image
import os

img_path = '/home/jovyan/DueDate/Dataset/Products-RS/images'
gt_path = '/home/jovyan/DueDate/Dataset/Products-RS/annotations(txt)'
gt_files = [os.path.join(gt_path, gt_file) for gt_file in sorted(os.listdir(gt_path))]
img_files = [os.path.join(img_path, img_file) for img_file in sorted(os.listdir(img_path))]
num_files = len([f for f in os.listdir(img_path)])
scale=0.25
length=512

for i in range(num_files):
    print(f"idx : {i}")
    with open(gt_files[i], 'r') as f:
        lines = f.readlines()
    vertices, labels = extract_vertices(lines)
    angle_range = (-175, 175)
    img = Image.open(img_files[i])

    img, vertices, cx, cy, rotation_matrix = rotate_img_and_resize(img, vertices, angle_range)
    img, vertices = affine_transform(img, vertices)
    img, vertices = perspect_transform(img, vertices)

    img.save(f'/home/jovyan/DueDate/Dataset/Products-All/images/img_{str(i+1).zfill(5)}.jpg')

    with open(f'/home/jovyan/DueDate/Dataset/Products-All/annotations(txt)/gt_img_{str(i+1).zfill(5)}.txt', 'w') as f:
        for vertex, label in zip(vertices, labels):
            line = ','.join(map(str, vertex))
            if label == 1:
                line += ',1'
            else:
                line += ',###'
            f.write(line + '\n')
            
            # [labels==1,:]