import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os

from PIL import Image
import numpy as np

from sklearn.neighbors import NearestNeighbors
import h5py

image_dir = '/media/jhyeup/5666b044-8f1b-47ad-83f3-d0acf3c6ec52/NAVER/outdoor_dataset/pangyo/train/images/'

def input_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
    ])

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

        numImage = len(self.images)
        self.DBidx = np.arange(int(len(self.images)/2)) * 2
        self.Qidx = self.DBidx + 1

        np.random.shuffle(self.DBidx)
        np.random.shuffle(self.Qidx)

        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.centers[self.DBidx])

        # potential positive images within 10 meters
        self.potential_positives = list(knn.radius_neighbors(self.centers[self.Qidx],
                radius=self.positive_threshold, return_distance=False))

        # sort index of potential positions
        for i, posi in enumerate(self.potential_positives) :
            self.potential_positives[i] = np.sort(posi)

        # it's possible some queries don't have any non trivial potential positives
        self.queries = np.where(np.array([len(x) for x in self.potential_positives])>0)[0]
        
        # for potential negatives
        potential_unnegatives = knn.radius_neighbors(self.centers[self.Qidx], 
                radius=self.negative_threshold, return_distance=False)

        # potential negative images away then 25 meters
        self.potential_negatives = []
        for pos in potential_unnegatives :
            self.potential_negatives.append(np.setdiff1d(np.arange(self.DBidx.shape[0]), pos, assume_unique=True))

        self.cache = None
        self.negCache = [np.empty((0,)) for _ in range(self.Qidx.shape[0])]

        
    def __getitem__(self, index) :
        # index 번째 query
        index = self.queries[index]
        with h5py.File(self.cache, mode='r') as h5 :
            h5feat = h5.get("features")

            # qOffset = DB의 갯수
            qOffset = self.DBidx.shape[0]
            # h5feat은 DB features가 먼저 들어가 있고 뒤에 Q features가 들어가있음.
            # qFeat는 query image 의 feature vector
            qFeat = h5feat[index+qOffset]

            # query image에 해당하는 potential positives들의 feature들의 vecotor
            posFeat = h5feat[self.potential_positives[index].tolist()]
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(posFeat)

            # potential positives들의 feature vector들 중에서 query image의 feature vector와 가장 가까운 것의 index
            dPos, posNN = knn.kneighbors(qFeat.reshape(-1,1), 1)
            dPos = dPos.item()
            posIndex = self.potential_positives[index][posNN[0]].item()

            # query image index에 해당하는 potential negative들 중에 1000개를 뽑는다. DBidx기준의 index가 됨.
            negSample = np.random.choice(self.potential_negatives[index], self.nNegSample)
            negSample = np.unique(np.concatenate([self.negCache[index], negSample]))

            negFeat = h5feat[negSample.tolist()]
            knn.fit(negFeat)

            # query image index에 해당하는 potential negative들 중에서 1000개를 무작위로 뽑고 
            # 그것들의 feature vector와 query의 featue vector를 비교해서 가장 가까운 100개를 뽑는다.
            dNeg, negNN = knn.kneighbors(qFeat.reshape(-1,1), self.nNeg*10)
            # dNeg는 potential negative들(좌표상에서 query가 해당하는 위치와 25미터 이상 떨어진 data들)의 feature vector와 
            # query의 feature vector를 비교하여 가장 가까운 100개
            dNeg = dNeg.reshape(-1)
            negNN = negNN.reshape(-1)

            # 좌표상에서 거리로는 더 멀어서 potential negative가 되었으나
            # featue vector가 potential positive의 feature vector보다 query와 가깝다고 판별되면
            # 실제로 loss를 계산하고 학습되어할 negative sample이라고 할수있다.
            violatingNeg = dNeg < dPos + self.margin**0.5

            # featue vector를 기준으로 positive sample 보다 query sample에 가까운 negative sample이 없다면
            # 해당 batch는 network update가 이루어지지 않으므로 skip한다
            if np.sum(violatingNeg) < 1 :
                return None

            # feature vector상의 거리가 positive sample보다 가까운 negative sample중 nNeg개를 추출
            negNN = negNN[violatingNeg][:self.nNeg]
            negIndices = negSample[negNN].astype(np.int32)

            self.negCache[index] = negIndices
        
        query = Image.open(os.path.join(image_dir, 'left')+'/'+self.images[Qidx][index])
        positive = Image.open(os.path.join(image_dir, 'left')+'/'+self.images[DBidx][posIndex])

        if self.input_transform :
            query = self.input_transform(query)
            positive = self.input_transform(positive)

        negatives = []
        for negIndex in negIndices :
            negative = Image.open(os.path.join(image_dir, 'left') + '/' + self.images[DBidx][negIndex])
            if self.input_transform :
                negative = self.input_transform(negative)
            negatives.append(negative)

        negatives = torch.stack(negatives, 0)
            
        return query, positive, negatives, [index, posIndex] + negIndices.tolist()

    def __len__(self) :
        return len(self.images)

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