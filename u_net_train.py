import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torchvision.transforms.functional as TF
from torch.optim.lr_scheduler import StepLR
import argparse
from tqdm import tqdm

from model import UNet
from dataset import EyeDataset, onehot2img, label2img
from loss import *
"""
ref:
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
https://github.com/taey16/pix2pixBEGAN.pytorch
https://github.com/gokriznastic/SegAN/blob/master/train.py
https://arxiv.org/pdf/1706.01805.pdf
https://github.com/advaitsave/Multiclass-Semantic-Segmentation-CamVid/blob/master/.ipynb_checkpoints/Multiclass%20Semantic%20Segmentation%20using%20U-Net-checkpoint.ipynb
https://github.com/usuyama/pytorch-unet
https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/loss.py
https://github.com/JunMa11/SegLoss/tree/master/losses_pytorch
"""

pretrainedModel = ''
lr = 1e-3
batchSize = 32
epoch_num = 1500

torch.manual_seed(1000)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = UNet().to(device)

#criterion = nn.CrossEntropyLoss().to(device)
#criterion = LovaszSoftmax().to(device)
criterion = FocalLoss(nn.Softmax(1), gamma=1).to(device)
optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0001)
scheduler = StepLR(optimizer, step_size=300, gamma=0.5)
loss_track = []

if pretrainedModel != '':
    model = torch.load(pretrainedModel)
    net.load_state_dict(model['state_dict'])
    loss_track = torch.load('loss.pth')
    if lr == 1e-3:
        optimizer.load_state_dict(model['optimizer'])
        scheduler.load_state_dict(model['scheduler'])
    epoch0 = model['epoch']
else: 
    epoch0 = 0

train_path = "./dataset/train"
val_path = "./dataset/test"
train_dataset = EyeDataset(train_path, aug=True)
val_dataset = EyeDataset(val_path)
train_dataloader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True, num_workers=8, pin_memory=False, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=batchSize, shuffle=False, num_workers=8, pin_memory=False)
trainSize = len(train_dataset)
valSize = len(val_dataset)

for epoch in range(epoch0, epoch_num):
    train_loss, val_loss = 0.0, 0.0
    
    net.train()
    for i, batch in enumerate(train_dataloader):
        img, label = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()
        output = net(img)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()/trainSize
    
    net.eval()
    with torch.no_grad():
        for batch in val_dataloader:
            img_v, label_v = batch[0].to(device), batch[1].to(device)
            
            output_v = net(img_v)
            loss = criterion(output_v, label_v)
            
            val_loss += loss.item()/valSize
    
    scheduler.step()
    loss_track.append((train_loss, val_loss))
    torch.save(loss_track, 'loss.pth')
    
    print('[{:4d}/{}] lr: {:.5f}, train_loss: {:.5f}, test_loss: {:.5f}'.format(epoch+1, epoch_num, optimizer.param_groups[0]['lr'], train_loss, val_loss))
    
    if epoch % 50 == 0:
        torch.save({
            'epoch': epoch+1,
            'state_dict':net.state_dict(),
            'optimizer':optimizer.state_dict(),
            'scheduler':scheduler.state_dict(),
        }, 'checkpoint/epoch_{}.pth'.format(epoch+1))
    

    
    if epoch % 5 == 0:
        img_out, img_gt = [], []
        sampleNum = 4
        for i in np.random.choice(32, sampleNum, replace=False):
            img_out.append(TF.to_tensor(onehot2img(output[i].detach().to('cpu').numpy())))
            img_gt.append(TF.to_tensor(label2img(label[i].detach().to('cpu').numpy())))
        for i in np.random.choice(17, sampleNum, replace=False):
            img_out.append(TF.to_tensor(onehot2img(output_v[i].detach().to('cpu').numpy())))
            img_gt.append(TF.to_tensor(label2img(label_v[i].detach().to('cpu').numpy())))
        vutils.save_image(torch.stack(img_out + img_gt), 'result_U/epoch_{}.png'.format(epoch+1), nrow=sampleNum*2)