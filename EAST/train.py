import torch
from torch.utils import data
from torch import nn
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import custom_dataset
from model import EAST
from loss import Loss
import os
import time
import numpy as np


def train(train_img_path, train_gt_path, pths_path, batch_size, lr, num_workers, epoch_iter, interval, resume_epoch=None):
	file_num = len(os.listdir(train_img_path))
	trainset = custom_dataset(train_img_path, train_gt_path)
	train_loader = data.DataLoader(trainset, batch_size=batch_size, \
                                   shuffle=True, num_workers=num_workers, drop_last=True)
	
	criterion = Loss()
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = EAST()
	data_parallel = False
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
		data_parallel = True
	model.to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150, 200, 250], gamma=0.5)

	if resume_epoch is not None:
		checkpoint_path = os.path.join(pths_path, f'model_epoch_{resume_epoch}.pth')
		checkpoint = torch.load(checkpoint_path, map_location=device)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		start_epoch = checkpoint['epoch']
	else:
		start_epoch = 0

	for epoch in range(start_epoch, epoch_iter):
		model.train()
		epoch_loss = 0
		epoch_time = time.time()
		for i, (img, gt_score, gt_geo, ignored_map) in enumerate(train_loader):
			start_time = time.time()
			img, gt_score, gt_geo, ignored_map = img.to(device), gt_score.to(device), gt_geo.to(device), ignored_map.to(device)
			pred_score, pred_geo = model(img)
			loss = criterion(gt_score, pred_score, gt_geo, pred_geo, ignored_map)
			
			epoch_loss += loss.item()
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			print('Epoch is [{}/{}], mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}'.format(\
				epoch+1, epoch_iter, i+1, int(file_num/batch_size), time.time()-start_time, loss.item()))
		
		scheduler.step()
		current_lr = optimizer.param_groups[0]['lr']
		print('epoch_loss is {:.8f}, epoch_time is {:.8f}, lr : {:.8f}'.format(epoch_loss/int(file_num/batch_size), time.time()-epoch_time, current_lr))
		print(time.asctime(time.localtime(time.time())))
		print('='*50)
		if (epoch + 1) % interval == 0:
			state = {
				'model_state_dict': model.module.state_dict() if data_parallel else model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'scheduler_state_dict': scheduler.state_dict(),
				'epoch': epoch + 1
			}
			torch.save(state, os.path.join(pths_path, 'model_epoch_{}.pth'.format(epoch+1)))



if __name__ == '__main__':
	train_img_path = os.path.abspath('/home/jovyan/DueDate/Dataset/Products-RS/images')
	train_gt_path  = os.path.abspath('/home/jovyan/DueDate/Dataset/Products-RS/annotations(txt)')
	pths_path      = './pths'
	batch_size     = 32 
	lr             = 1e-3
	num_workers    = 4
	epoch_iter     = 300
	save_interval  = 5
	resume_epoch   = None # Default = None
	train(train_img_path, train_gt_path, pths_path, batch_size, lr, num_workers, epoch_iter, save_interval, resume_epoch)
	
