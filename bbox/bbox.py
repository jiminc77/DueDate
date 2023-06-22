import cv2
import argparse
import json
import os
import matplotlib.pyplot as plt


def main(args):

    # 폴더 경로
    root = "/home/jovyan/DueDate/Dataset/Products-RS"

    # 이미지 파일 경로
    selected_image_name = args.image_name

    # json 파일 읽기
    with open(os.path.join(root, 'annotations.json'), 'r') as json_file:
        data = json.load(json_file)

    if selected_image_name in data:
        img_info = data[selected_image_name]
        image_path = os.path.join(root, 'images', selected_image_name)
        image = cv2.imread(image_path)
        
        for ann in img_info['ann']:
            cls = ann['cls']
            # if cls == 'date':
            bbox = ann['bbox']
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 바운딩 박스가 그려진 이미지 보기
    plt.imshow(image_rgb)
    plt.title(f"Image with BBox: {selected_image_name}")
    plt.savefig(os.path.join('bbox_show', selected_image_name), dpi=2000)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Draw a bounding box around the image.')
    parser.add_argument('--image_name', type=str, required=True, help='Ex) img_00001.jpg')
    args = parser.parse_args()
    main(args)