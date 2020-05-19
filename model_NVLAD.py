import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import numpy as np

import torchvision.models as models

# based on https://github.com/Nanne/pytorch-NetVlad/netvlad.py
class NetVLAD(nn.Module) :
    def __init__(self) :
        self.num_clusters = 64
        self.alpha = 0
        self.dim = 128
        self.normalize_input = True
        self.vladv2 = False

        super(NetVLAD, self).__init__()
        self.conv = nn.Conv2d(self.dim, self.num_clusters, kernel_size=(1,1), bias=self.vladv2)
        self.centroids = nn.Parameter(torch.rand(self.num_clusters, self.dim))

        # using pretrained vgg16
        vgg16 = models.vgg16(pretrained=True)
        # remove last relu and maxpool
        vgg16_layers = list(self.vgg16.features.children())[:-2]
        
        for layer in vgg16_layers[:-5] :
            for parameter in layer.parameters() :
                parameter.requires_grad = False

        self.vgg16 = nn.Sequential(*vgg16_layers)
        