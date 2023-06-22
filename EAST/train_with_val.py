import torch
from torch.utils import data
from torch import nn
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import custom_dataset
from model import EAST
from loss import Loss
import matplotlib.pyplot as plt
import os
import time
import numpy as np

def train(train_img_path, train_gt_path, pths_path, batch_size, lr, num_workers, epoch_iter, interval, best_loss=np.inf, resume_epoch=None):
    file_num = len(os.listdir(train_img_path))
    trainset = custom_dataset(train_img_path, train_gt_path)
    valset = custom_dataset(val_img_path, val_gt_path) 
    train_loader = data.DataLoader(trainset, batch_size=batch_size, \
                                   shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = data.DataLoader(valset, batch_size=batch_size, \
                                   shuffle=False, num_workers=num_workers, drop_last=True)

    train_losses = []
    val_losses = []

    criterion = Loss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    data_parallel = False

    patience = 20
    counter = 0
    best_loss = np.inf

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        data_parallel = True
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.1)

    # scheduler = CosineAnnealingLR(optimizer, T_max=epoch_iter, eta_min=0)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)


    if resume_epoch is not None:
        checkpoint_path = os.path.join(pths_path, f'model_epoch_{resume_epoch}.pth')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"start with : {start_epoch}")
    else:
        start_epoch = 0
        print("start with 0")

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

            print('Epoch is [{}/{}], mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}'.format(
                epoch+1, epoch_iter, i+1, int(file_num/batch_size), time.time()-start_time, loss.item()))
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print('epoch_loss is {:.8f}, epoch_time is {:.8f}, lr : {:.8f}'.format(epoch_loss/int(file_num/batch_size), time.time()-epoch_time, current_lr))
        print(time.asctime(time.localtime(time.time())))
        print('='*50)

        # validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, (img, gt_score, gt_geo, ignored_map) in enumerate(val_loader):
                img, gt_score, gt_geo, ignored_map = img.to(device), gt_score.to(device), gt_geo.to(device), ignored_map.to(device)
                pred_score, pred_geo = model(img)
                loss = criterion(gt_score, pred_score, gt_geo, pred_geo, ignored_map)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print('Validation loss:', val_loss)

        train_losses.append(epoch_loss/int(file_num/batch_size))

        val_losses.append(val_loss)
        
        plt.figure(figsize=(10, 5))
        plt.title("Training and Validation Loss")
        plt.plot(train_losses,label="train")
        plt.plot(val_losses,label="val")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig('/home/jovyan/DueDate/EAST/results/loss_graph.png')

        # save the best model
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            state = {
                'model_state_dict': model.module.state_dict() if data_parallel else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch + 1
            }
            torch.save(state, os.path.join(pths_path, 'best_model.pth'))
            print('Best model saved')
        else:   
            counter += 1
            print(f'EarlyStopping counter: {counter} out of {patience}')
            if counter >= patience:
                print('Early stopping...')
                return  # or 'break' if you want to just exit the loop

        if (epoch + 1) % interval == 0:
            state = {
                'model_state_dict': model.module.state_dict() if data_parallel else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch + 1
            }
            torch.save(state, os.path.join(pths_path, 'model_epoch_{}.pth'.format(epoch+1)))


if __name__ == '__main__':
    train_img_path = os.path.abspath('/home/jovyan/DueDate/Dataset/Products-All/images')
    train_gt_path  = os.path.abspath('/home/jovyan/DueDate/Dataset/Products-All/annotations(txt)')
    val_img_path   = os.path.abspath('/home/jovyan/DueDate/Dataset/Products-All/val/images')
    val_gt_path    = os.path.abspath('/home/jovyan/DueDate/Dataset/Products-All/val/annotations(txt)')
    pths_path      = './pths'
    batch_size     = 32 
    lr             = 1e-3
    num_workers    = 90
    epoch_iter     = 150
    save_interval  = 5
    resume_epoch   = 50 # Default = None
    train(train_img_path, train_gt_path, pths_path, batch_size, lr, num_workers, epoch_iter, save_interval, resume_epoch = 50)
