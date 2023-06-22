from dataset import extract_vertices
from dataset import affine_transform
from dataset import perspect_transform
from dataset import adjust_height
from dataset import rotate_img
from dataset import crop_img
from dataset import rotate_img_and_resize
from dataset import get_score_geo
from dataset import randomly_scale_image
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

for i in range(20):
    print(f"idx : {i}")
    with open(gt_files[i], 'r') as f:
        lines = f.readlines()
    vertices, labels = extract_vertices(lines)
    img = Image.open(img_files[i])
    img, vertices = randomly_scale_image(img, vertices)
    img, vertices = adjust_height(img, vertices)
    print(f'image {i+1} : labels : {labels}')
    img, vertices = crop_img(img, vertices, labels, length)
    get_score_geo(img, vertices, labels, scale, length)

    img.save(f'/home/jovyan/DueDate/AugTest/Aug_test/sample{i+1}.png', 'png')

    # OpenCV는 BGR 채널 순서를 사용하므로, 이를 matplotlib의 RGB 순서로 변환
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ax.imshow(img) 
    vertices = vertices[labels==1,:].reshape(-1, 2)
    rect = patches.Polygon(vertices, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.savefig(f'/home/jovyan/DueDate/AugTest/Aug_test/show{i + 1}.png')
