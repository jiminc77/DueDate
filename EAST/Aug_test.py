from dataset import extract_vertices
from dataset import affine_transform
from dataset import perspect_transform
from dataset import adjust_height
from dataset import rotate_img
from dataset import crop_img
from dataset import rotate_img_and_resize
from dataset import get_score_geo
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
from PIL import Image
import os

img_path = '/home/jovyan/DueDate/AugTest/Test_Img'
gt_path = '/home/jovyan/DueDate/AugTest/Test_GT'
gt_files = [os.path.join(gt_path, gt_file) for gt_file in sorted(os.listdir(gt_path))]
img_files = [os.path.join(img_path, img_file) for img_file in sorted(os.listdir(img_path))]
scale=0.25
length=512

for i in range(10):
    print(f"idx : {i}")
    with open(gt_files[i], 'r') as f:
        lines = f.readlines()
    vertices, labels = extract_vertices(lines)
    angle_range = (-170, 170)
    img = Image.open(img_files[i])

    img, vertices, cx, cy, rotation_matrix = rotate_img_and_resize(img, vertices, angle_range)
    img, vertices = affine_transform(img, vertices)
    img, vertices = perspect_transform(img, vertices)
    img, vertices = adjust_height(img, vertices)
    img, vertices = crop_img(img, vertices, labels, length, cx, cy, rotation_matrix)
    # img, vertices = rotate_img(img, vertices)
    get_score_geo(img, vertices, labels, scale, length)

    img.save(f'/home/jovyan/DueDate/AugTest/Aug_test/sample{i+1}.png', 'png')

    # OpenCV는 BGR 채널 순서를 사용하므로, 이를 matplotlib의 RGB 순서로 변환합니다.
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 이미지를 표시합니다.
    fig, ax = plt.subplots(1)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ax.imshow(img)
    # crop_box = np.array([[crop_start_w, crop_start_h], [crop_start_w + length, crop_start_h], [crop_start_w + length, crop_start_h + length], [crop_start_w, crop_start_h + length]])

    # bbox 좌표를 이용해 사각형을 생성합니다.
    vertices = vertices[0].reshape(-1, 2) # 좌표를 (x, y) 쌍으로 변환합니다.
    # crop_box = crop_box.reshape(-1, 2)
    # print(crop_box)
    rect = patches.Polygon(vertices, linewidth=1, edgecolor='r', facecolor='none')
    # crop_box = patches.Polygon(crop_box, linewidth=1, edgecolor='g', facecolor='none')

    # 생성한 사각형을 이미지에 추가합니다.
    ax.add_patch(rect)
    # ax.add_patch(crop_box)
    plt.savefig(f'/home/jovyan/DueDate/AugTest/Aug_test/show{i + 1}.png')
