import sys
import os
import random
import math
import time
import torch; torch.utils.backcompat.broadcast_warning.enabled = True
from torchvision import transforms, datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.backends.cudnn as cudnn; cudnn.benchmark = True
import numpy as np
    
class Model(nn.Module):

    def __init__(self,args):
        super(Model, self).__init__()
        args_defaults=dict(num_channels=182, num_classes=10, verbose=False)
        for arg,default in args_defaults.items():
            setattr(self, arg, args[arg] if arg in args and args[arg] is not None else default)
    
        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, (1, self.num_channels), padding = 0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)
        
        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, 32))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)
        
        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))
        
        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 1280 timepoints. 
        # self.fc1 = nn.Linear(640, self.num_classes)
        self.fc1 = nn.Linear(640, 64)

        # Audio feature pathway
        self.fc_audio1 = nn.Linear(11, 32)  # 11 audio features (danceability, energy, key, ...)
        self.fc_audio2 = nn.Linear(32, 64)

        self.fc_final = nn.Linear(704, self.num_classes) # adjust


    def forward(self, x, song_features):
        if self.verbose: 
            print(f"[INPUT] {x.shape}")
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        # Layer 1
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.25)
        x = x.permute(0, 3, 1, 2)
        if self.verbose:
            print(f"[CONV1] {x.shape}")
        
        # Layer 2
        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.25)
        x = self.pooling2(x)
        if self.verbose:
            print(f"[CONV2] {x.shape}")
        
        # Layer 3
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.25)
        x = self.pooling3(x)
        if self.verbose:
            print(f"[CONV3] {x.shape}")

        # FC Layer
        x = x.contiguous().view(x.shape[0], -1)
        if self.verbose:
            print(f"[VIEW] {x.shape}")
        # x = torch.sigmoid(self.fc1(x))
        x_eeg = F.elu(self.fc1(x))
        if self.verbose:
            print(f"[FC1] {x_eeg.shape}")

        # audio
        print(song_features.shape)
        x_audio = F.elu(self.fc_audio1(song_features))
        x_audio = F.elu(self.fc_audio2(x_audio))
        if self.verbose:
            print(f"[FC2] {x_audio.shape}")
        
        # concatenate EEG and song features
        x_concat = torch.cat((x, x_audio), dim=1)
        out = self.fc_final(x_concat)

        return out


if __name__ == "__main__":
    net = Model().cuda(0)
    print (net.forward(Variable(torch.Tensor(np.random.rand(64, 1, 120, 64)).cuda(0))))
   