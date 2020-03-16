import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torchvision.transforms.functional as TF
import torchvision.transforms as TT
from torch.optim.lr_scheduler import StepLR
import argparse
from tqdm import tqdm

from model import UNet
from dataset import *
from loss import *
from metrics import pixel_accuracy, mean_IOU, f1_score

model_unet = 'checkpoint/epoch_{}.pth'.format(151)
model_GAN = 'checkpoint_GAN/epoch_{}.pth'.format(151)


torch.manual_seed(1000)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net1 = UNet().to(device)
net2 = UNet().to(device)

net1.load_state_dict(torch.load(model_unet)['state_dict'])
net2.load_state_dict(torch.load(model_GAN)['state_dict_G'])

val_path = "./dataset/test"
val_dataset = EyeDataset(val_path)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=False)
valSize = len(val_dataset)

PA_unet = 0
PA_GAN = 0
IOU_unet = 0
IOU_GAN = 0
F1_unet = 0
F1_GAN = 0
with torch.no_grad():
    for batch in tqdm(val_dataloader):
        img, label, onehot = batch[0].to(device), batch[1].numpy(), batch[2].numpy()
        
        output_unet = net1(img).to('cpu').numpy()
        output_GAN = net2(img).to('cpu').numpy()
        
        for i in range(img.size()[0]):
            eval_unet = img2label(onehot2img(output_unet[i]))
            eval_GAN = img2label(onehot2img(output_GAN[i]))

            PA_unet += pixel_accuracy(eval_unet, label[i])
            PA_GAN += pixel_accuracy(eval_GAN, label[i])
            IOU_unet += mean_IOU(eval_unet, label[i])
            IOU_GAN += mean_IOU(eval_GAN, label[i])
            F1_unet += f1_score(eval_unet, label[i])
            F1_GAN += f1_score(eval_GAN, label[i])
        
PA_unet /= valSize
PA_GAN /= valSize
IOU_unet /= valSize
IOU_GAN /= valSize
F1_unet /= valSize
F1_GAN /= valSize

print("PA_unet: {}, PA_GAN: {}, IOU_unet: {}, IOU_GAN: {}".format(PA_unet, PA_GAN, IOU_unet, IOU_GAN))
print("F1_unet: {}, F1_GAN: {}".format(F1_unet, F1_GAN))
"""
with torch.no_grad():
    img_ori, img_gt, img_unet, img_GAN = [], [], [], []
    sampleNum = 4
    for i in np.random.choice(17, sampleNum, replace=False):
        img_ori.append(TF.to_tensor(TT.ToPILImage()(img[i].to('cpu')).convert("RGB")))
        img_gt.append(TF.to_tensor(label2img(label[i])))
        img_unet.append(TF.to_tensor(onehot2img(output_unet[i])))
        img_GAN.append(TF.to_tensor(onehot2img(output_GAN[i])))
    vutils.save_image(torch.stack(img_ori + img_gt + img_unet + img_GAN), 'result.png', nrow=sampleNum)
"""