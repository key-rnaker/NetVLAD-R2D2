import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import numpy as np

import torchvision.models as models

# based on https://github.com/Nanne/pytorch-NetVlad/netvlad.py
class NetVLAD(nn.Module) :
    def __init__(self) :
        super(NetVLAD, self).__init__()
        self.num_clusters = 64
        self.alpha = 0
        self.dim = 512
        self.normalize_input = True
        # convolution operation for soft assignment
        self.conv = nn.Conv2d(self.dim, self.num_clusters, kernel_size=(1,1), bias=True)

    def init_param(self, clsts, traindescs) :

        ############ Convolution Layer ############
        # using pretrained vgg16
        vgg16 = models.vgg16(pretrained=True)
        # remove last relu and maxpool
        vgg16_layers = list(vgg16.features.children())[:-2]
        
        for layer in vgg16_layers[:-5] :
            for parameter in layer.parameters() :
                parameter.requires_grad = False

        self.vgg16 = nn.Sequential(*vgg16_layers)
        del vgg16

        ############ NetVLAD layer ############
        knn = NearestNeighbors(n_jobs=-1)
        traindescs = traindescs.reshape(-1, self.dim)
        knn.fit(traindescs)
        dsSq = np.square(knn.kneighbors(clsts, 2)[0])
        # initialize weights and bias of conv
        self.alpha = (-np.log(0.01) / np.mean(dsSq[:,1] - dsSq[:,0])).item()
        self.centroids = nn.Parameter(torch.from_numpy(clsts))
        self.conv.weight = nn.Parameter( (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1))
        self.conv.bias = nn.Parameter( - self.alpha * self.centroids.norm(dim=1))

        del traindescs, knn, clsts, dsSq

    def forward(self, image) :

        # local descriptor x of image
        x = self.vgg16(image)
        
        num_batches, num_channels = x.shape[:2]
        #print(x.shape)

        if self.normalize_input :
            x = F.normalize(x, p=2, dim=1)

        # soft assignment = [48, 64, 256]
        soft_assign = self.conv(x).view(num_batches, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)
        #print(soft_assign.shape)

        # local descriptor x flatten
        #x_flatten = [4, 256, 512]
        x_flatten = x.view(num_batches, -1, num_channels)

        # vlad = [48, 64, 512]
        vlad = torch.zeros([num_batches, self.num_clusters, num_channels], dtype=x.dtype, layout=x.layout, device=x.device)
        
        for batch in range(num_batches) :
            for cluster in range(self.num_clusters) :
                #residual = [256, 512]
                # 256개로 이루어져 있고 각각은 512개
                residual = x_flatten[batch] - self.centroids[cluster,:].expand(x_flatten.size(1), -1)
                residual *= soft_assign[batch,cluster].unsqueeze(0).T
                vlad[batch, cluster, :] = residual.sum(dim=0)
                #for i in range(x_flatten.shape[1]) : 
                    #vlad[batch,cluster,:] += soft_assign[batch,cluster,i] * ( x_flatten[batch,i,:] - self.centroids[cluster,:])

        vlad = F.normalize(vlad, p=2, dim=2) # intra-normalization
        vlad = vlad.view(num_batches, -1)     # flatten
        vlad = F.normalize(vlad, p=2, dim=1) # L2-normalization

        # vlad.shape = [num_batches, num_clusters x num_channels]

        return vlad
                
