import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.nn.functional as F
import os

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np

from sklearn.neighbors import NearestNeighbors
import h5py

image_dir = '/media/jhyeup/5666b044-8f1b-47ad-83f3-d0acf3c6ec52/NAVER/outdoor_dataset/pangyo/train/images/'
root_dir = '/home/jhyeup/NetVLAD-R2D2/'

def input_transform():
    # pre trained VGG16 model expects input images normalized
    # mean and std of ImageNet
    return transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
    ])

def collate_fn(batch) :

    batch = list(filter(lambda x : x is not None, batch))
    if len(batch) == 0 : return None, None, None, None, None

    query, positive, negatives, indices = zip(*batch)
    """
    queary : tuple (batch_size, tensor(3, h, w))
    positive : tuple (batch_size, tensor(3, h, w))
    negatives : tuple (batch_size, tensor(n, 3, h, w))
    """

    query = data.dataloader.default_collate(query)
    positive = data.dataloader.default_collate(positive)
    negCounts = data.dataloader.default_collate([x.shape[0] for x in negatives])
    negatives = torch.cat(negatives, 0)
    import itertools
    indices = list(itertools.chain(*indices))

    return query, positive, negatives, negCounts, indices

class NAVERDataset(data.Dataset) :
    def __init__(self, input_transform ) :
        super().__init__()

        self.input_transform = input_transform()

        # Training Images
        self.images = os.listdir(os.path.join(image_dir, 'left'))
        self.images = np.sort(np.array(self.images))
        # XYZ Location of each Training Images
        # used to nontrivial_positives and potential negatives.
        self.centers = np.load(image_dir + 'centers.npy')
        self.nNegSample = 1000
        self.nNeg = 10
        self.margin = 0.1
        self.positive_threshold = 10
        self.negative_threshold = 25

        # number of total images
        numImage = len(self.images)
        # divide images into Q for query and DB for potential and negatives
        self.DBidx = np.arange(int(len(self.images)/2)) * 2
        self.Qidx = self.DBidx + 1

        np.random.shuffle(self.DBidx)
        np.random.shuffle(self.Qidx)

        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.centers[self.DBidx])

        # potential positives' indices of DBidx for each query within 10 meters
        self.potential_positives = list(knn.radius_neighbors(self.centers[self.Qidx],
                radius=self.positive_threshold, return_distance=False))

        # sort indecies of potential positives
        for i, positive_indices in enumerate(self.potential_positives) :
            self.potential_positives[i] = np.sort(positive_indices)

        # it's possible some queries don't have any non trivial potential positives
        self.queries = np.where(np.array([len(x) for x in self.potential_positives])>0)[0]
        
        # for potential negatives
        potential_unnegatives = knn.radius_neighbors(self.centers[self.Qidx], 
                radius=self.negative_threshold, return_distance=False)

        # potential negatives' indices of DBidx away then 25 meters
        self.potential_negatives = []
        for pos in potential_unnegatives :
            self.potential_negatives.append(np.setdiff1d(np.arange(self.DBidx.shape[0]), pos, assume_unique=True))

        self.cache = None
        self.negCache = [np.empty((0,)) for _ in range(self.Qidx.shape[0])]
        
    def __getitem__(self, index) :
        # index of centers[Qidx]
        index = self.queries[index]
        with h5py.File(self.cache, mode='r') as h5 :
            h5feat = h5.get("features")

            # vlad vector of query image from cache
            qFeat = h5feat[self.Qidx[index]]

            # vlad vector of potential positives assigned to query image
            posFeat = h5feat[sorted(self.DBidx[self.potential_positives[index]].tolist())]
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(posFeat)

            # positive's index of DBidx closest to query vlad vector
            dPos, posNN = knn.kneighbors(qFeat.reshape(1,-1), 1)
            dPos = dPos.item()
            posIndex = self.potential_positives[index][posNN[0]].item()

            # choose number of nNegSample potiential negatives' indices of DBidx
            negSample = np.random.choice(self.potential_negatives[index], self.nNegSample)
            negSample = np.unique(np.concatenate([self.negCache[index], negSample])).astype(np.int)

            negFeat = h5feat[sorted(self.DBidx[negSample].tolist())]
            knn.fit(negFeat)

            # choose number of nNeg x 10 negatives whose vlad vector is closest to query vlad vector
            dNeg, negNN = knn.kneighbors(qFeat.reshape(1,-1), self.nNeg*10)
            dNeg = dNeg.reshape(-1)
            negNN = negNN.reshape(-1)

            # dNeg is vlad vector of potential negatives
            # dPos is vlad vector of positive
            violatingNeg = dNeg < dPos + self.margin**0.5

            # skip if no violatingNeg
            if np.sum(violatingNeg) < 1 :
                return None

            # choose number of nNeg negatives in violatingNeg
            negNN = negNN[violatingNeg][:self.nNeg]
            # nNeg number of negative's indices of DBidx 
            negIndices = negSample[negNN].astype(np.int32)

            self.negCache[index] = negIndices
        
        query = Image.open(os.path.join(image_dir, 'left', self.images[self.Qidx][index]))
        positive = Image.open(os.path.join(image_dir, 'left', self.images[self.DBidx][posIndex]))

        if self.input_transform :
            query = self.input_transform(query)
            positive = self.input_transform(positive)

        negatives = []
        for negIndex in negIndices :
            negative = Image.open(os.path.join(image_dir, 'left') + '/' + self.images[self.DBidx][negIndex])
            if self.input_transform :
                negative = self.input_transform(negative)
            negatives.append(negative)

        negatives = torch.stack(negatives, 0)
            
        return query, positive, negatives, [index, posIndex] + negIndices.tolist()

    def __len__(self) :
        return len(self.queries)

class NAVERIMGDataset(data.Dataset) :
    def __init__(self, input_transform) :
        super().__init__()

        self.images = os.listdir(os.path.join(image_dir, 'left'))
        self.images = np.sort(np.array(self.images))
        self.input_transform = input_transform()

    def __getitem__(self, index) :

        img = Image.open(os.path.join(image_dir, 'left') +'/' + self.images[index])
        img = self.input_transform(img)
        
        return img, index

    def __len__(self) :
        return len(self.images)