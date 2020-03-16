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
import os

from model import UNet, Discriminator
from dataset import EyeDataset, onehot2img, label2img
from loss import *

pretrainedModel = ''
lr = 1e-4
batchSize = 32
epoch_num = 1500

torch.manual_seed(1000)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = UNet().to(device)
D = Discriminator().to(device)
print("G_params: {}".format(sum(p.numel() for p in G.parameters() if p.requires_grad)))
print("D_params: {}".format(sum(p.numel() for p in D.parameters() if p.requires_grad)))

criterion = FocalLoss(nn.Softmax(1), gamma=1).to(device)

if not os.path.exists('./result_GAN'):
    os.makedirs('./result_GAN', exist_ok=True)
if not os.path.exists('./checkpoint_GAN'):
    os.makedirs('./checkpoint_GAN', exist_ok=True)

optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0001)
optimizer_D = optim.RMSprop(D.parameters(), lr=5e-5)

loss_track = []

if pretrainedModel != '':
    model = torch.load(pretrainedModel)
    G.load_state_dict(model['state_dict_G'])
    D.load_state_dict(model['state_dict_D'])
    loss_track = torch.load('loss.pth')
    if lr == 1e-4:
        optimizer_G.load_state_dict(model['optimizer_G'])
        optimizer_D.load_state_dict(model['optimizer_D'])
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
    
    G.train(), D.train()
    for i, batch in enumerate(train_dataloader):
        for d_iter in range(1):
            optimizer_D.zero_grad()
    
            img, label, onehot = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            output = G(img)

            real_imgs = torch.cat((img, onehot), 1)
            fake_imgs = torch.cat((img, output.detach()), 1)
    
            loss_D = -torch.mean(D(real_imgs)) + torch.mean(D(fake_imgs))
    
            loss_D.backward()
            optimizer_D.step()
    
            for p in D.parameters():
                p.data.clamp_(-0.01, 0.01)

        for g_iter in range(1):
            # generator
            optimizer_G.zero_grad()
            gen_imgs = torch.cat((img, output), 1)
            
            loss_G = -torch.mean(D(gen_imgs))
            loss_focal = criterion(output, label)
            loss = loss_focal + loss_G
    
            loss.backward()
            optimizer_G.step()
    
            train_loss += loss_focal.item() / trainSize
    
    G.eval(), D.eval()
    with torch.no_grad():
        for batch in val_dataloader:
            img_v, label_v = batch[0].to(device), batch[1].to(device)
            
            output_v = G(img_v)
            loss = criterion(output_v, label_v)
            
            val_loss += loss.item() / valSize
    
    loss_track.append((train_loss, loss_G, loss_D, val_loss))
    torch.save(loss_track, 'checkpoint_GAN/loss.pth')
    
    print('[{:4d}/{}], tr_ls: {:.5f}, G_ls: {:.5f}, D_ls: {:.5f}, te_ls: {:.5f}'.format(epoch+1, epoch_num, train_loss, loss_G, loss_D, val_loss))
    
    if epoch % 50 == 0:
        torch.save({
            'epoch': epoch+1,
            'state_dict_G':G.state_dict(),
            'state_dict_D':D.state_dict(),
            'optimizer_G':optimizer_G.state_dict(),
            'optimizer_D':optimizer_D.state_dict(),
        }, 'checkpoint_GAN/model.pth')
    

    
    if epoch % 5 == 0:
        img_out, img_gt = [], []
        sampleNum = 4
        for i in np.random.choice(32, sampleNum, replace=False):
            img_out.append(TF.to_tensor(onehot2img(output[i].detach().to('cpu').numpy())))
            img_gt.append(TF.to_tensor(label2img(label[i].detach().to('cpu').numpy())))
        for i in np.random.choice(17, sampleNum, replace=False):
            img_out.append(TF.to_tensor(onehot2img(output_v[i].detach().to('cpu').numpy())))
            img_gt.append(TF.to_tensor(label2img(label_v[i].detach().to('cpu').numpy())))
        vutils.save_image(torch.stack(img_out + img_gt), 'result_GAN/epoch_{}.png'.format(epoch+1), nrow=sampleNum*2)