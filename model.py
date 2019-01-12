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

# Load pretrained model
# feature_cnn = ConvNet().to(device)
# feature_cnn.load_state_dict(torch.load("fashionCNN.weight"))

class FeatureMaps(nn.Module):
    """Get pretrained CNN feature maps"""
    def __init__(self):
        super(FeatureMaps, self).__init__()
        cnn = ConvNet()
        cnn.load_state_dict(torch.load("fashionCNN.weight"))
        
        self.conv0 = cnn.conv0
        self.conv1 = cnn.conv1
        self.conv2 = cnn.conv2
        
    def forward(self, img):
        
        f0 = self.conv0(img).detach()
        f1 = self.conv1(f0).detach()
        f2 = self.conv2(f1).detach()
        
        return f0, f1, f2

# Embed the images using pretrained feature maps
class ConvEmbeddingNet(nn.Module):
    """
    Convolution Net that classifies MNIST images and output embedding map
    """
    def __init__(self, dim_hid=8, dim_embed=50, n_class=10):
        super(ConvEmbeddingNet, self).__init__()
        
        self.cnn = FeatureMaps()
        
        self.embed = nn.Sequential(
            nn.Dropout(0.05),
            nn.Linear(88*dim_hid, dim_embed*8),
            nn.LeakyReLU(),
            nn.Linear(dim_embed*8, dim_embed*4),
            nn.LeakyReLU(),
            nn.Linear(dim_embed*4, dim_embed)
        )
        self.output = nn.Sequential(
            nn.LeakyReLU(), 
            nn.Linear(dim_embed, n_class)   
                                   )
        self.cat2vec = nn.Embedding(n_class, dim_embed, max_norm=10.0)
        
    def forward(self, img, label):
        # max_norm for embedded items
        max_norm = 25.0
        
        batch, _, _, _ = img.size()
        
        f0, f1, f2 = self.cnn(img)
        f_ = torch.cat((f1.view(batch,-1), f2.view(batch,-1)), dim=1)
        v_x = self.embed(f_)
        v_norm = (torch.sqrt((v_x**2).sum(1)) + 1e-6).detach()
        v_norm[v_norm < max_norm] = 1.0
        v_x = v_x / v_norm.unsqueeze(1)
        logit = self.output(v_x)
        v_y = self.cat2vec(label)
        assert v_x.size() == v_y.size()
        
        return (v_x, v_y), logit, (f0, f1, f2)