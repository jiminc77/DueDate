import sys
sys.path.append('/home/jovyan/DueDate/EAST')
sys.path.append('/home/jovyan/DueDate/Text_Recognition')
from torchvision import transforms
from PIL import Image, ImageDraw
from model import EAST
from EAST.dataset import get_rotate_mat
from skimage.transform import rotate
from utils import CTCLabelConverter, AttnLabelConverter
from Text_Recognition.dataset import RawDataset, AlignCollate
from Text_Recognition.model import Model
from dateutil.relativedelta import relativedelta
from collections import defaultdict
import skimage
import torch
import os
import numpy as np
import cv2
import lanms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as mtransforms
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import string
import argparse
import datetime
import torchvision


def resize_img(img):
	'''resize image to be divisible by 32
	'''
	w, h = img.size
	resize_w = w
	resize_h = h

	resize_h = resize_h if resize_h % 32 == 0 else int(resize_h / 32) * 32
	resize_w = resize_w if resize_w % 32 == 0 else int(resize_w / 32) * 32
	img = img.resize((resize_w, resize_h), Image.BILINEAR)
	ratio_h = resize_h / h
	ratio_w = resize_w / w

	return img, ratio_h, ratio_w


def load_pil(img):
	'''convert PIL Image to torch.Tensor
	'''
	t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))])
	return t(img).unsqueeze(0)


def is_valid_poly(res, score_shape, scale):
	'''check if the poly in image scope
	Input:
		res        : restored poly in original image
		score_shape: score map shape
		scale      : feature map -> image
	Output:
		True if valid
	'''
	cnt = 0
	for i in range(res.shape[1]):
		if res[0,i] < 0 or res[0,i] >= score_shape[1] * scale or \
           res[1,i] < 0 or res[1,i] >= score_shape[0] * scale:
			cnt += 1
	return True if cnt <= 1 else False


def restore_polys(valid_pos, valid_geo, score_shape, scale=4):
	'''restore polys from feature maps in given positions
	Input:
		valid_pos  : potential text positions <numpy.ndarray, (n,2)>
		valid_geo  : geometry in valid_pos <numpy.ndarray, (5,n)>
		score_shape: shape of score map
		scale      : image / feature map
	Output:
		restored polys <numpy.ndarray, (n,8)>, index
	'''
	polys = []
	index = []
	valid_pos *= scale
	d = valid_geo[:4, :] # 4 x N
	angle = valid_geo[4, :] # N,

	for i in range(valid_pos.shape[0]):
		x = valid_pos[i, 0]
		y = valid_pos[i, 1]
		y_min = y - d[0, i]
		y_max = y + d[1, i]
		x_min = x - d[2, i]
		x_max = x + d[3, i]
		rotate_mat = get_rotate_mat(-angle[i])
		
		temp_x = np.array([[x_min, x_max, x_max, x_min]]) - x
		temp_y = np.array([[y_min, y_min, y_max, y_max]]) - y
		coordidates = np.concatenate((temp_x, temp_y), axis=0)
		res = np.dot(rotate_mat, coordidates)
		res[0,:] += x
		res[1,:] += y
		
		if is_valid_poly(res, score_shape, scale):
			index.append(i)
			polys.append([res[0,0], res[1,0], res[0,1], res[1,1], res[0,2], res[1,2],res[0,3], res[1,3]])
	return np.array(polys), index


def get_boxes(score, geo, score_thresh=0.95, nms_thresh=0.5):
	'''get boxes from feature map
	Input:
		score       : score map from model <numpy.ndarray, (1,row,col)>
		geo         : geo map from model <numpy.ndarray, (5,row,col)>
		score_thresh: threshold to segment score map
		nms_thresh  : threshold in nms
	Output:
		boxes       : final polys <numpy.ndarray, (n,9)>
	'''
	score = score[0,:,:]
	xy_text = np.argwhere(score > score_thresh) # n x 2, format is [r, c]
	if xy_text.size == 0:
		return None

	xy_text = xy_text[np.argsort(xy_text[:, 0])]
	valid_pos = xy_text[:, ::-1].copy() # n x 2, [x, y]
	valid_geo = geo[:, xy_text[:, 0], xy_text[:, 1]] # 5 x n
	polys_restored, index = restore_polys(valid_pos, valid_geo, score.shape) 
	if polys_restored.size == 0:
		return None

	boxes = np.zeros((polys_restored.shape[0], 9), dtype=np.float32)
	boxes[:, :8] = polys_restored
	boxes[:, 8] = score[xy_text[index, 0], xy_text[index, 1]]
	boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thresh)
	return boxes


def adjust_ratio(boxes, ratio_w, ratio_h):
	'''refine boxes
	Input:
		boxes  : detected polys <numpy.ndarray, (n,9)>
		ratio_w: ratio of width
		ratio_h: ratio of height
	Output:
		refined boxes
	'''
	if boxes is None or boxes.size == 0:
		return None
	boxes[:,[0,2,4,6]] /= ratio_w
	boxes[:,[1,3,5,7]] /= ratio_h
	return np.around(boxes)
	
	
def detect(img, model, device):
	'''detect text regions of img using model
	Input:
		img   : PIL Image
		model : detection model
		device: gpu if gpu is available
	Output:
		detected polys
	'''
	img, ratio_h, ratio_w = resize_img(img)
	with torch.no_grad():
		score, geo = model(load_pil(img).to(device))
	boxes = get_boxes(score.squeeze(0).cpu().numpy(), geo.squeeze(0).cpu().numpy())
	return adjust_ratio(boxes, ratio_w, ratio_h)


def plot_boxes(img, boxes):

	if boxes is None:
		return img
	
	draw = ImageDraw.Draw(img)
	for box in boxes:
		draw.polygon([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]], outline=(0,255,255))
	return img


def rotate_parallelogram(img, box, index):
    
    img = np.array(img)
    pts = np.array([[box[0], box[1]], [box[2], box[3]], [box[4], box[5]], [box[6], box[7]]], np.int32)

    lengths = [np.linalg.norm(pts[i]-pts[(i+1)%4]) for i in range(4)]
    max_index = np.argmax(lengths)

    dx = pts[(max_index+1)%4, 0] - pts[max_index, 0]
    dy = pts[(max_index+1)%4, 1] - pts[max_index, 1]
    angle = np.degrees(np.arctan2(dy, dx))

    center = np.mean(pts, axis=0)

    M = cv2.getRotationMatrix2D((center[0], center[1]), angle, 1)
    img_rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, 
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    pts_rotated = cv2.transform(np.array([pts]), M)[0]

    min_x = np.min(pts_rotated[:, 0])
    max_x = np.max(pts_rotated[:, 0])
    min_y = np.min(pts_rotated[:, 1])
    max_y = np.max(pts_rotated[:, 1])

    img_cropped = img_rotated[int(min_y):int(max_y), int(min_x):int(max_x)]
    img_cropped_rotate = cv2.rotate(img_cropped, cv2.ROTATE_180)

    # if img_cropped is not None:
    #     cv2.imwrite(f'cropped_{index}.png', img_cropped)
    #     cv2.imwrite(f'rotated_and_cropped_{index}.png', img_cropped_rotate)

    return [img_cropped, img_cropped_rotate]


def Text_Recog_single_image(opt, model_text_recog, converter, image, device):

    cv2.imwrite('before.png', image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    image[:,:,0] = clahe.apply(image[:,:,0])
    image[:,:,1] = clahe.apply(image[:,:,1])
    image[:,:,2] = clahe.apply(image[:,:,2])
    image = cv2.resize(image, (opt.imgW, opt.imgH), interpolation=cv2.INTER_LINEAR)
    image = Image.fromarray(image)
    image = torchvision.transforms.ToTensor()(image).unsqueeze(0)
    
    batch_size = image.size(0)
    image = image.to(device)

    length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
    text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

    preds = model_text_recog(image, text_for_pred)

    preds_size = torch.IntTensor([preds.size(1)] * batch_size)
    _, preds_index = preds.max(2)
    preds_str = converter.decode(preds_index, preds_size)

    preds_prob = F.softmax(preds, dim=2)
    preds_max_prob, _ = preds_prob.max(dim=2)

    confidence_score = preds_max_prob.cumprod(dim=0)[-1]
    
    valid_preds_str = []
    valid_confidence_score = []
    
    for pred, score in zip(preds_str, confidence_score):
        date_pred = get_date(pred)
        if date_pred is not None:
            valid_preds_str.append(date_pred)
            valid_confidence_score.append(score.item())
            # print(f'{date_pred:32s}\t{score.item():0.4f}')

    return valid_preds_str, valid_confidence_score


def get_date(input_date):
    
    formats = {
    "%Y%m%d": ["%Y%m%d", "%y%m%d", "%d%m%Y", "%d%m%y"],
    "%m%d": ["%m%d"]
    }

    month_names = {
        "jan": "01", "feb": "02", "mar": "03", "apr": "04", "may": "05", "jun": "06",
        "jul": "07", "aug": "08", "sep": "09", "oct": "10", "nov": "11", "dec": "12"
    }
    
    input_date = input_date.lower()

    for name, num in month_names.items():
        input_date = input_date.replace(name, num)

    possible_dates = []
    for output_format, input_formats in formats.items():
        for input_format in input_formats:
            try:
                if len(input_date) != len(datetime.datetime.now().strftime(input_format)):
                    continue
                date = datetime.datetime.strptime(input_date, input_format)
                if input_format in formats["%m%d"]:
                    date = date.replace(year=2023)
                if date.year < 100:
                    if date.year < 22 or date.year > 26:
                        continue
                    date = date.replace(year=date.year+2000)
                elif date.year < 2022 or date.year > 2026:
                    continue
                possible_dates.append(date.strftime("%Y%m%d"))
            except ValueError:
                continue

    current_date = datetime.datetime.now()
    min_date = current_date - relativedelta(years=1)
    max_date = current_date + relativedelta(years=2)
    possible_dates = [datetime.datetime.strptime(date, "%Y%m%d") for date in possible_dates if min_date <= datetime.datetime.strptime(date, "%Y%m%d") <= max_date]

    if possible_dates:
        max_diff_date = max(possible_dates, key=lambda date: abs((date - current_date).days))
        return max_diff_date.strftime("%Y%m%d")
    else:
        return None


def main(opt):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # EAST
    model = EAST().to(device)
    model.load_state_dict(torch.load(opt.model_path)['model_state_dict'])
    model.eval()
    
    # TPS-ResNet-BiLSTM-CTC
    converter = CTCLabelConverter(opt.character)
    opt.num_class = len(converter.character)
    model_text_recog = Model(opt)
    model_text_recog = torch.nn.DataParallel(model_text_recog).to(device)
    model_text_recog.load_state_dict(torch.load(opt.saved_model, map_location=device))
    model_text_recog.eval()
    
    cap = cv2.VideoCapture(opt.video_path)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(opt.output_name, fourcc, fps, (width, height))

    pred_freq = defaultdict(int)
    max_freq_pred = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if hasattr(img, '_getexif'): 
            orientation = 0x0112
            exif = img._getexif()
            if exif is not None:
                orientation = exif[orientation]
                if orientation in [3, 6, 8]:
                    if orientation == 3:
                        img = img.transpose(Image.ROTATE_180)
                    elif orientation == 6:
                        img = img.transpose(Image.ROTATE_270)
                    elif orientation == 8:
                        img = img.transpose(Image.ROTATE_90)

        boxes = detect(img, model, device)
        plot_img = plot_boxes(img, boxes)
        
        if boxes is not None:
            for i, box in enumerate(boxes):
                crop_images = rotate_parallelogram(img, box, i)
                
                for crop_img in crop_images:
                    if crop_img is not None:
                        h, w = crop_img.shape[:2]
                        if h < 10 or w < 10:
                            continue
                        valid_preds_str, valid_confidence_score = Text_Recog_single_image(opt, model_text_recog, converter, crop_img, device)
                        for pred in valid_preds_str:
                            pred_freq[pred] += 1  # 빈도수 누적        
        
        if pred_freq:
            max_freq_pred = max(pred_freq, key=pred_freq.get)
        
        if max_freq_pred:
            print("\n" + "=" * 30)
            print(f'  Expiration Date : {str(max_freq_pred)}  ')
            print("=" * 30 + "\n")

            frame = cv2.cvtColor(np.array(plot_img), cv2.COLOR_RGB2BGR)
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f" Expiration Date : {max_freq_pred}"
            cv2.rectangle(frame, (10, 30), (550, 70), (0, 255, 0), -1)
            cv2.putText(frame, text, (10, 50), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
        
        out.write(frame)

    cap.release()
    out.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    """ Recognitor Setting """
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=95)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--saved_model', default = '/home/jovyan/DueDate/Text_Recognition/saved_models/TPS-ResNet-BiLSTM-CTC-Seed1111/best_accuracy.pth', help="path to recognition model")
    parser.add_argument('--batch_max_length', type=int, default=8, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', default = True, help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--Transformation', type=str, default = 'TPS', help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, default = 'ResNet', help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, default = 'BiLSTM', help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, default = 'CTC', help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=3, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    """ Detector Setting """
    parser.add_argument('--video_path', default = '/home/jovyan/DueDate/Dataset/Video_Demo/Demo_8.mp4', help='path to video which contains exp date object')
    parser.add_argument('--model_path', default = '/home/jovyan/DueDate/EAST/pths/best_model.pth', help='path to text detection model')
    """ Output Setting """
    parser.add_argument('--output_name', default = 'output_8.mp4')
    opt = parser.parse_args()
    
    cudnn.benchmark = True
    cudnn.deterministic = True
    
    main(opt)

