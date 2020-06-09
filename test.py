import torch
from torch.utils.data import DataLoader

import argparse
from PIL import Image
import numpy as np
import os
import h5py
from sklearn.neighbors import NearestNeighbors

from dataloader import NAVERIMGDataset
from dataloader import input_transform
from model_NVLAD import NetVLAD


parser = argparse.ArgumentParser(description="NetVLAD-Test")
parser.add_argument('--checkpoint-path', type=str)
parser.add_argument('--cache-path', type=str)
parser.add_argument('--image-path', type=str)

root_dir = '/home/jhyeup/NetVLAD-R2D2'
image_dir = '/media/jhyeup/5666b044-8f1b-47ad-83f3-d0acf3c6ec52/NAVER/outdoor_dataset/pangyo/train/images/'

cacheBatchSize = 40
num_clusters = 64
encoder_dim = 512

if __name__ == "__main__" :

    args = parser.parse_args()

    print("===> Loading Model")
    netvlad = NetVLAD()
    device = 'cuda'

    with h5py.File(os.path.join(root_dir, 'centroids', 'cluster.hdf5'), 'r') as h5 :
        clsts = h5.get('centroids')[...]
        traindescs = h5.get('descriptors')[...]
        netvlad.init_param(clsts, traindescs)
        del clsts, traindescs

    checkpoint = torch.load(args.checkpoint_path)
    netvlad.load_state_dict(checkpoint['state_dict'])
    netvlad.to(device)
    print("Done")

    image = Image.open(args.image_path)

    print("===> Loading Dataset")
    test_data = NAVERIMGDataset(input_transform)
    test_data_loader = DataLoader(dataset=test_data, batch_size=cacheBatchSize, shuffle=False, num_workers=8, pin_memory=True)
    print("Done")
    vlad_size = num_clusters * encoder_dim

    print("===> Loading Cache")
    if args.cache_path == None :
        DBfeature = np.empty([len(test_data), vlad_size])
        with torch.no_grad() :
            for iteration, (input, indices) in enumerate(test_data_loader, 1) :
                input = input.to(device)
                vlad = netvlad(input)
                DBfeature[indices.detach().numpy(), :] = vlad.detach().cpu().numpy()
                del input, vlad

    else :
        h5 =  h5py.File(args.cache_path, mode='r')
        DBfeature = h5.get('features')
        del h5
    print("Done")

    print("===>Predicting")
    image_transform = input_transform()
    qimage = image_transform(image).unsqueeze_(0).to(device)
    qvlad = netvlad.forward(qimage).detach().cpu().numpy()

    knn = NearestNeighbors(n_jobs=-1)
    candidates = []
    knn.fit(DBfeature[:10000])
    candidates.append(knn.kneighbors(qvlad.reshape(1, -1), 1))

    knn.fit(DBfeature[10000:20000])
    candidates.append(knn.kneighbors(qvlad.reshape(1, -1), 1))

    knn.fit(DBfeature[20000:30000])
    candidates.append(knn.kneighbors(qvlad.reshape(1, -1), 1))

    knn.fit(DBfeature[30000:40000])
    candidates.append(knn.kneighbors(qvlad.reshape(1,-1), 1))

    knn.fit(DBfeature[40000:])
    candidates.append(knn.kneighbors(qvlad.reshape(1,-1), 1))

    closest_idx = np.argmin(np.array([candidates[i][0].item() for i in range(len(candidates)) ]))

    closest_idx = closest_idx * 10000 + candidates[closest_idx][1].item()

    #image.show()
    #result_image, _ = test_data.__getitem__(closest_idx)
    #result_image = Image.open(os.path.join(image_dir, 'left') + '/' + test_data.images[closest_idx])
    #result_image.show()

    for i in range(len(candidates)) :
        print("image name : ", candidates[i][1].item() + i*10000, "euclidean distance is ",candidates[i][0].item())