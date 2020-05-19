import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision.models as models

from math import ceil
from dataloader import image_dir, input_transform
from dataloader import NAVERIMGDataset
import os
import h5py
import numpy as np
import sklearn

if __name__ == "__main__" :
    print("Making Clusters")
    nDescriptors = 50000
    nPerImage = 100
    # 이미지 5000개에서 각각 100개의 descriptor를 추출하여 총 50000개의 descriptor 저장
    nImage = ceil(nDescriptors/nPerImage)
    feature_dimension = 512
    nCluster = 64

    # NAVER Data image Dataset
    image_set = NAVERIMGDataset(input_transform)

    sampler = SubsetRandomSampler(np.random.choice(len(image_set), nImage, replace=False))
    data_loader = DataLoader(dataset=image_set, num_workers=6, batch_size=2, shuffle=False, 
            pin_memory=True, sampler=sampler)

    # centroids 폴더 생성
    if not os.path.exists(os.path.join('/home/jhyeup/NetVLAD-R2D2', 'centroids')) :
        os.makedirs(os.path.join('/home/jhyeup/NetVLAD-R2D2', 'centroids'))

    cluster_file = os.path.join('/home/jhyeup/NetVLAD-R2D2', 'centroids', 'cluster.hdf5')
    
    # 입력 이미지들의 feature vector를 얻기위한 CNN 모델
    model = nn.Module()
    cnn = models.vgg16(pretrained=True)
        
    layers = list(cnn.features.children())[:-2]
    for layer in layers :
        for parameter in layer.parameters() :
            parameter.requires_grad = False

    cnn = nn.Sequential(*layers)
    model.add_module('cnn', cnn)

    cuda = False
    device = torch.device("cuda" if cuda else "cpu")
    
    with h5py.File(cluster_file, mode='w') as h5 :
        with torch.no_grad() :
            model.eval()
            print('====> Extracting Descriptors')
            dbFeat = h5.create_dataset("descriptors",
                        [nImage, nPerImage, feature_dimension], dtype=np.float32)

            for iteration, (image, indices) in enumerate(data_loader, 1) :
                input = image.to(device)
                # input = ( 이미지 갯수 X Channel X Height X Width)
                # model.cnn(input) = ( 이미지 갯수 X feature_dimension X feature_height X feature_width)
                # view = ( 이미지 갯수 X feature_dimension X feature_heightxfeature_width)
                # permute = ( 이미지 갯수 x feature_heightxfeature_width X feature_dimension)
                image_descriptors = model.cnn(input).view(input.size(0), feature_dimension, -1).permute(0, 2, 1)
        
        
                for i in range(image_descriptors.size(0)) :
                    # 이미지 마다 feature_heightxfeature_width개의 descriptor 중에 100개 추출
                    sample = np.random.choice(image_descriptors.size(1), nPerImage, replace=False )
                    dbFeat[(iteration-1)*image_descriptors.size(0) + i, :] = image_descriptors[i,sample,:].detach().cpu().numpy()

                del input, image_descriptors

                print("===> ", iteration, " Done")

        #arrayFeat = np.array(dbFeat)
        print("===> Clustring")
        kmeans = sklearn.cluster.KMeans(n_cluster=nCluster, random_state=0).fit(dbFeat.reshape(-1,feature_dimension))
        centroids = kmeans.cluster_centers_

        print("===> Storing")
        h5.create_dataset('centroids', data=centroids)
            
        print("===> Done")
        
