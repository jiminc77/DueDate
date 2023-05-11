import time
import json
import torch
import subprocess
import os
import matplotlib.pyplot as plt
from model import EAST
from detect import detect_dataset
import numpy as np
import shutil


def eval_model(model_name, test_img_path, submit_path, save_flag=True):
	if os.path.exists(submit_path):
		shutil.rmtree(submit_path) 
	os.mkdir(submit_path)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = EAST(False).to(device)
	model.load_state_dict(torch.load(model_name))
	model.eval()

	start_time = time.time()
	detect_dataset(model, device, test_img_path, submit_path)
	os.chdir(submit_path)
	res = subprocess.getoutput('zip -q submit.zip *.txt')
	res = subprocess.getoutput('mv submit.zip ../')
	os.chdir('../')
	res = subprocess.getoutput('python ./evaluate/script.py –g=./evaluate/gt.zip –s=./submit.zip')
	print(res)
	os.remove('./submit.zip')
	print('eval time is {}'.format(time.time()-start_time))	

	if not save_flag:
		shutil.rmtree(submit_path)
	return res

def epoch_num(model_name):
    return int(model_name.split('_')[-1].split('.')[0])

def plot_metrics(epochs, precisions, recalls, hmeans):
	plt.figure(figsize=(12, 6))
	plt.plot(epochs, precisions, label='Precision')
	plt.plot(epochs, recalls, label='Recall')
	plt.plot(epochs, hmeans, label='H-mean')
	plt.xlabel('Epoch')
	plt.ylabel('Metric Value')
	plt.legend(loc='best')
	plt.title('Performance Metrics per Epoch')
	plt.savefig(save_path)


if __name__ == '__main__': 
	model_dir = '/home/jovyan/DueDate/EAST/pths/Test_2'
	test_img_path = '/home/jovyan/DueDate/Dataset/Products-Real/evaluation/images'
	submit_path = '/home/jovyan/DueDate/EAST/submit'

	precisions = []
	recalls = []
	hmeans = []
	epochs = []

	for model_name in sorted(os.listdir(model_dir), key=epoch_num):
		if model_name.endswith('.pth'):
			epoch = int(model_name.split('_')[-1].split('.')[0])
			epochs.append(epoch)
			model_path = os.path.join(model_dir, model_name)
			print(f'Evaluating model: {model_name}')
			result = eval_model(model_path, test_img_path, submit_path)
			if not result.startswith('Calculated!'):  # 변경된 부분
				print(f'No result for {model_name}\n')
				continue
			result = result[len('Calculated!'):].strip()
			result_dict = json.loads(result)
			precision = result_dict['precision']
			recall = result_dict['recall']
			hmean = result_dict['hmean']
			precisions.append(precision)
			recalls.append(recall)
			hmeans.append(hmean)
			print(f'Precision: {precision}, Recall: {recall}, H-mean: {hmean}\n')

	save_path = '/home/jovyan/DueDate/EAST/pths/Test_2/performance_metrics_per_epoch.png'
	plot_metrics(epochs, precisions, recalls, hmeans)