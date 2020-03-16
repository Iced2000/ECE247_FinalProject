import os
import numpy as np

import torch
import random
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import scipy
from PIL import Image, ImageOps
from tqdm import tqdm

def labelPath(dataName):
    return dataName[:-4] + '_label' + dataName[-4:]

def processImg(img):
    img = img.crop((0, 0, 400, 800))
    img = img.resize((128, 256))
    return img

def img2onehot(rawImg):
    img = np.asarray(rawImg)
    h, w, c = img.shape
    img = img.reshape(-1, 3)
    output = np.zeros((5, img.shape[0]))
    for i in range(img.shape[0]):
        if (img[i, 0] < 128 and img[i, 1] < 128 and img[i, 2] < 128) or (img[i, 0] >= 128 and img[i, 1] >= 128 and img[i, 2] >= 128):
            output[0, i] = 1
        elif img[i, 0] < 128 and img[i, 1] >= 128 and img[i, 2] < 128:
            output[1, i] = 1
        elif img[i, 0] < 128 and img[i, 1] < 128 and img[i, 2] >= 128:
            output[2, i] = 1
        elif img[i, 0] >= 128 and img[i, 1] >= 128 and img[i, 2] < 128:
            output[3, i] = 1
        elif img[i, 0] >= 128 and img[i, 1] < 128 and img[i, 2] < 128:
            output[4, i] = 1
        else:
            raise
    return output.reshape(5, h, w)

def img2label(rawImg):
    img = np.asarray(rawImg)
    h, w, c = img.shape
    img = img.reshape(-1, 3)
    output = np.zeros((img.shape[0]))
    for i in range(img.shape[0]):
        if (img[i, 0] < 128 and img[i, 1] < 128 and img[i, 2] < 128) or (img[i, 0] >= 128 and img[i, 1] >= 128 and img[i, 2] >= 128):
            output[i] = 0
        elif img[i, 0] < 128 and img[i, 1] >= 128 and img[i, 2] < 128:
            output[i] = 1
        elif img[i, 0] < 128 and img[i, 1] < 128 and img[i, 2] >= 128:
            output[i] = 2
        elif img[i, 0] >= 128 and img[i, 1] >= 128 and img[i, 2] < 128:
            output[i] = 3
        elif img[i, 0] >= 128 and img[i, 1] < 128 and img[i, 2] < 128:
            output[i] = 4
        else:
            raise
    return output.reshape(h, w)

def onehot2img(vec):
    c, h, w = vec.shape
    vec = vec.reshape(5, -1)
    img = np.zeros((vec.shape[1], 3))
    for i in range(vec.shape[1]):
        lb = np.argmax(vec[:, i])
        if lb == 0:
            img[i] = np.array([0, 0, 0])
        elif lb == 1:
            img[i] = np.array([0, 255, 0])
        elif lb == 2:
            img[i] = np.array([0, 0, 255])
        elif lb == 3:
            img[i] = np.array([255, 255, 0])
        elif lb == 4:
            img[i] = np.array([255, 0, 0])
        else:
            raise
    img = img.reshape(h, w, 3)
    img = Image.fromarray(np.uint8(img))
    #img.show()
    return img

def label2img(vec):
    h, w = vec.shape
    vec = vec.reshape(-1)
    img = np.zeros((vec.shape[0], 3))
    for i in range(vec.shape[0]):
        if vec[i] == 0:
            img[i] = np.array([0, 0, 0])
        elif vec[i] == 1:
            img[i] = np.array([0, 255, 0])
        elif vec[i] == 2:
            img[i] = np.array([0, 0, 255])
        elif vec[i] == 3:
            img[i] = np.array([255, 255, 0])
        elif vec[i] == 4:
            img[i] = np.array([255, 0, 0])
        else:
            raise
    img = img.reshape(h, w, 3)
    img = Image.fromarray(np.uint8(img))
    #img.show()
    return img

class EyeDataset(Dataset):
    def __init__(self, path, aug=False):
        """
        dataset structure:
        dataset - train - data
                        - label
                - test  - data
                        - label

        """

        if os.path.isfile('{}.pkl'.format(path.split('/')[-1])):
            self.images, self.labels, self.onehot = torch.load('{}.pkl'.format(path.split('/')[-1]))

        else:
            self.dataPath = os.path.join(path, 'data')
            self.labelPath = os.path.join(path, 'label')
    
            self.images = []
            self.labels = []
            self.onehot = []
    
            for _path, _, fns in os.walk(self.dataPath):
                for fn in tqdm(fns):
                    rawImg = Image.open(os.path.join(self.dataPath, fn)).convert('L')
                    rawLabel = Image.open(os.path.join(self.labelPath, labelPath(fn)))

                    imgTensor = transforms.functional.to_tensor(processImg(rawImg))
                    labelTensor = torch.from_numpy(img2label(processImg(rawLabel))).long()
                    ohTensor = torch.from_numpy(img2onehot(processImg(rawLabel))).float()

                    if labelTensor.sum() == 0:
                        continue
                    
                    self.images.append(imgTensor)
                    self.labels.append(labelTensor)
                    self.onehot.append(ohTensor)

                    if aug:
                        flipImg = ImageOps.mirror(rawImg)
                        flipLabel = ImageOps.mirror(rawLabel)

                        flipImgTensor = transforms.functional.to_tensor(processImg(flipImg))
                        flipLabelTensor = torch.from_numpy(img2label(processImg(flipLabel))).long()
                        flipOhTensor = torch.from_numpy(img2onehot(processImg(flipLabel))).float()
                    
                        self.images.append(flipImgTensor)
                        self.labels.append(flipLabelTensor)
                        self.onehot.append(flipOhTensor)
            
            torch.save((self.images, self.labels, self.onehot), '{}.pkl'.format(path.split('/')[-1]))
        
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx], self.onehot[idx]

if __name__ == '__main__':
    path = './dataset/test'
    dset = EyeDataset(path)
    loader = DataLoader(dset,
                       batch_size=32,
                       shuffle=True,
                       num_workers=8,
                       pin_memory=False,
                       drop_last=False)

    for i, (image, label, onehot) in enumerate(loader):
        print(image.size(), label.size(), onehot.size())