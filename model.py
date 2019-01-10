import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Pretained model as feature maps
class ConvNet(nn.Module):
    """
    Convolution Net that classifies MNIST images
    """
    def __init__(self, dim_hid=8, dim_embed=50, n_class=10):
        super(ConvNet, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(1, dim_hid, 3,1,0),
            nn.BatchNorm2d(dim_hid),
            nn.ReLU(),
            nn.Conv2d(dim_hid, dim_hid, 3,1,1),
            nn.BatchNorm2d(dim_hid),
            nn.ReLU(),
            nn.MaxPool2d(2,2,1)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim_hid, 2*dim_hid, 3,1,0),
            nn.BatchNorm2d(2*dim_hid),
            nn.ReLU(),
            nn.Conv2d(2*dim_hid, 2*dim_hid, 3,1,1),
            nn.BatchNorm2d(2*dim_hid),
            nn.ReLU(),
            nn.MaxPool2d(2,2,0)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(2*dim_hid, 4*dim_hid, 3,1,0),
            nn.BatchNorm2d(4*dim_hid),
            nn.ReLU(),
            nn.Conv2d(4*dim_hid, 4*dim_hid, 3,1,1),
            nn.BatchNorm2d(4*dim_hid),
            nn.ReLU(),
            nn.MaxPool2d(2,2,0)
        )
        
        self.out = nn.Sequential(
            nn.Linear(2*2*4*dim_hid, dim_embed*8),
            nn.LeakyReLU(),
            nn.Linear(dim_embed*8, dim_embed*4),
            nn.LeakyReLU(),
            nn.Linear(dim_embed*4, dim_embed),
            nn.LeakyReLU(), 
            nn.Linear(dim_embed, n_class)
        )
        
    def forward(self, img):
        
        out = self.conv0(img)
        out = self.conv1(out)
        out = self.conv2(out)
        batch_size, _, _, _ = out.size()

        out = out.view(batch_size, -1)

        out = self.out(out)
        
        return out