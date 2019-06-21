from __future__ import print_function, division
import os,io
import torch
import numpy as np
import PIL
from PIL import Image
import cv2
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import random
from skimage.transform import resize

class retDataset(Dataset):

    def __init__(self, path_file, trnFlg=0, transform=None):
        f = open(path_file,"r")
        self.alines = f.readlines()
        self.transform = transform
        self.trnFlg = trnFlg
        self.allImgs = []
        self.allgt = []
        self.allsn = []

        widthHM = 96
        heightHM = 96
        for idx in range(0,len(self.alines)):
                line = self.alines[idx]
                line = line.split()
                nm = line[0].split("/")[-1]
                seq_no = int(line[0].split("/")[-2]) 
                lbl = np.float32(np.asarray([-float(line[1]), float(line[3]), -float(line[2])]))
                prts = nm.split("_")
                namehm = prts[0] + '_' + prts[1] + '_rgb_c' + line[0].split("/")[-2] + '_heatmaps.png'
                isPrs = True
                img = Image.open('/home/aryaman.g/projects/cscFcPs/allImgOut/'+namehm)
                i=0
                hmroi = (widthHM*i,0,widthHM*(i+1),heightHM)
                hm = [] 
                tmp = img.crop(hmroi)
                hm.append(tmp)	
                for i in range(1,5):	
                    hmroi = (widthHM*(i+13),0,widthHM*(i+14),heightHM)
                    tmp = img.crop(hmroi)
                    hm.append(tmp)	
                self.allImgs.append(hm)
                self.allgt.append(lbl)
                self.allsn.append(seq_no)
    
    def __getitem__(self, idx):
        hm = self.Imgs[idx]
        lbl = self.gt[idx]
        
        margin = 12
        x = random.randint(0,margin)
        y = random.randint(0,margin)
        xd = 96-random.randint(0,margin)    
        yd = 96-random.randint(0,margin)   
        nhm = [] 
        for i in range(0,5):
            nhm.append(hm[i].crop((x,y,xd,yd)))
        fhm = []       
        for i in range(0,5):
            fhm.append(np.array(nhm[i].resize((96, 96), PIL.Image.ANTIALIAS)))
        
        hmc = np.zeros((5,96,96))
        for i in range(0,5):	
            hmc[i,:,:] = fhm[i]
      
        img = hmc
        img = img.astype('float32')
        img /= 255
        img = torch.from_numpy(img)
        lbl += 90
        lbl /= 180
        
        func_choice = random.choice([True, False])
        if func_choice and self.trnFlg==1:
            t = torch.zeros((5,96,96))
            for j in range(96):
                t[:,:,j] = img[:,:,96-1-j]
            lbl[0] = 1-(lbl[0])
            lbl[2] = 1-(lbl[2])
            img = t
        
        lbl =  torch.from_numpy(lbl)
        if self.transform:
            img = self.transform(img)
        return img, lbl
	
    def __len__(self):
        return len(self.Imgs)

    def to_categorical(self, y, num_classes):
        return np.eye(num_classes, dtype='uint8')[y]

    def kfldShuff(self, ks=[]):
        self.Imgs = []
        self.gt = []
        self.lines = []
        if len(ks)==0:
            for idx in range(0,len(self.alines)):    
                self.Imgs.append(self.allImgs[idx])
                self.gt.append(self.allgt[idx])   
        else:
            for idx in range(0,len(self.alines)):
                if self.allsn[idx] in ks:
                    self.Imgs.append(self.allImgs[idx])
                    self.gt.append(self.allgt[idx]) 
                    self.lines.append(self.alines[idx])  
 
#train_dataset = retDataset('biwiGT',1,0)
#train_dataset = retDataset('/home/aryaman.g/pyTorchLearn/biwiTrain.txt',1)
#train_dataset = biwiDataset('/ssd_scratch/cvit/aryaman.g/biwiHighResHM/allHM',1)
#print(train_dataset.__getitem__(2))
