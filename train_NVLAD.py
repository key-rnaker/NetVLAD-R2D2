from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torch.autograd import Function

import argparse
import os
import h5py
import numpy as np
import json
import shutil
import time
from sklearn.neighbors import NearestNeighbors
from tensorboardX import SummaryWriter
from datetime import datetime
from math import ceil

from dataloader import input_transform
from dataloader import NAVERDataset
from dataloader import NAVERIMGDataset
from dataloader import collate_fn
from model_NVLAD import NetVLAD

parser = argparse.ArgumentParser(description='NetVLAD-R2D2')
parser.add_argument('--dataPath', type=str, default='/')
parser.add_argument('--savePath', type=str, default='checkpoints/')
parser.add_argument('--resume', type=str, default='', help='checkpoint path')

root_dir = '/home/jhyeup/NetVLAD-R2D2/'

batchSize = 4
nEpochs = 30
lr = 0.0001
weightDecay = 0.001
cacheBatchSize = 40
cacheRefreshRate = 1000
momentum = 0.9
margin = 0.1
start_epoch = 0
num_clusters = 64
encoder_dim = 512

class TripletMarginLoss(nn.Module) :
    def __init__(self, margin) :
        super().__init__()
        self.margin = margin

    def forward(self, query, positive, negatives) :     
        positive_distance = torch.dist(query, positive)
        loss = torch.zeros(negatives.shape[0])
        for i in range(loss.shape[0]) :
            loss[i] = positive_distance + self.margin - torch.dist(query, negatives[i])

        return loss.sum()

def train(epoch) :

    epoch_loss = 0
    startIter = 1

    # nSubset = 48117 / 1000 = 49
    # 총 49개 subset 
    nSubset = ceil(len(naver_set) / cacheRefreshRate)
    subsetIdx = np.array_split(np.arange(len(naver_set)), nSubset)
    nBatches = (len(naver_set) + batchSize - 1) // batchSize

    for subIter in range(nSubset) :
        if subIter == 0 and os.path.exists(os.path.join(root_dir, 'centroids', 'feat_cache.hdf5')) :
            naver_set.cache = os.path.join(root_dir, 'centroids', 'feat_cache.hdf5')
            print('cache exist')

        else :
            print('===> Building cache')
            netvlad.eval()
            naver_set.cache = os.path.join(root_dir, 'centroids', 'feat_cache.hdf5')
            with h5py.File(naver_set.cache, mode='w') as h5 :
                vlad_size = encoder_dim * num_clusters
                h5feat = h5.create_dataset('features', [len(naver_img_set), vlad_size], dtype=np.float32)

                print("Number of Images : {}".format(len(naver_img_set)))

                with torch.no_grad() :
                    for iteration, (input, indices) in enumerate(imgDataLoader, 1) :
                        start = time.time()
                        input = input.to(device)
                        vlad = netvlad(input)
                        h5feat[indices.detach().numpy(), :] = vlad.detach().cpu().numpy()
                        del input, vlad
                        print("takes {} seconds for one batch".format(time.time() - start))

            print("Done cache")

        train_subset = Subset(naver_set, indices=subsetIdx[subIter])
        train_loader = DataLoader(dataset=train_subset, batch_size=batchSize, shuffle=True, num_workers=8, pin_memory=True, collate_fn=collate_fn)

        print("Allocated:", torch.cuda.memory_allocated())
        print("Cached:", torch.cuda.memory_cached())

        netvlad.train()
        for iteration, (query, positive, negatives, negCounts, indices) in enumerate(train_loader, startIter) :

            if query is None : continue

            B, C, H, W = query.shape
            nNeg = torch.sum(negCounts)
            # query, positive, negatives를 한번에 넣기 위해 합침
            input = torch.cat([query, positive, negatives])
            input = input.to(device)
            vlad = netvlad(input)

            # 결과를 분리
            vladQ, vladP, vladNs = torch.split(vlad, [B, B, nNeg])
            
            optimizer.zero_grad()

            loss = 0
            for i in range(B) :
                loss += criterion(vladQ[i], vladP[i], vladNs[torch.sum(negCounts[:i]).item() : torch.sum(negCounts[:i+1]).item()])

            loss.to(device)
            loss.backward()
            optimizer.step()
            del input, vlad, vladQ, vladP, vladNs
            del query, positive, negatives

            batch_loss = loss.item()
            epoch_loss += batch_loss

            print("aaa")

            if iteration % 50 == 0 :
                print("==> Epoch[{}]({}/{}) : Loss : {:.4f}".format(epoch, iteration, nBatches, batch_loss))
                writer.add_scalar('Train/Loss', batch_loss, ((epoch-1) * nBatches) + iteration)
                writer.add_scalar('Train/nNeg', nNeg, ((epoch-1) *nBatches) + iteration)
                print('Allocated :', torch.cuda.memory_allocated())
                print('Cached :', torch.cuda.memory_cached())

        startIter += len(train_loader)
        del train_loader, loss
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        os.remove(naver_set.cache)

    avg_loss = epoch_loss / nBatches

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, avg_loss), flush=True)
    writer.add_scalar('Train/AvgLoss)', avg_loss, epoch)

def test(epoch) :
    return

def save_checkpoint(savePath, state, is_best, filename='checkpoint.pth.tar') :
    model_out_path = os.path.join(savePath, filename)
    torch.save(state, model_out_path)
    if is_best :
        shutil.copyfile(model_out_path, os.path.join(savePath, 'model_best.pth.tar'))

if __name__ == "__main__" :
    
    args = parser.parse_args()

    if not torch.cuda.is_available() :
        raise Exception("GPU failed")
    torch.cuda.empty_cache()

    device = torch.device("cuda")

    print('===> loading dataset')   #데이터셋 로딩
    naver_set = NAVERDataset(input_transform=input_transform)
    naver_img_set = NAVERIMGDataset(input_transform=input_transform)
    imgDataLoader = DataLoader(naver_img_set, num_workers=8, batch_size=cacheBatchSize, shuffle=False, pin_memory=True)
    print('Done load')

    print('===> Building model')    # 모델 생성
    
    # NetVLAD model
    netvlad = NetVLAD()
    with h5py.File(os.path.join(root_dir, 'centroids', 'cluster.hdf5')) as h5 :
        clsts = h5.get('centroids')[...]
        traindescs = h5.get('descriptors')[...]
        netvlad.init_param(clsts, traindescs)
        del clsts, traindescs

    netvlad.to(device)

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, netvlad.parameters()), lr=lr, momentum=momentum, weight_decay=weightDecay)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # The shape of all input tenseors should be (N, D)
    criterion = TripletMarginLoss(margin=margin).to(device)

    print('Done Build')

    print('===> Training')

    if not os.path.exists(os.path.join(root_dir, 'log')) :
        os.makedirs(os.path.join(root_dir, 'log'))

    writer = SummaryWriter(log_dir=os.path.join(root_dir, 'log', datetime.now().strftime('%b%d_%H-%M-%S')))
    logdir = writer.file_writer.get_logdir()
    savePath = os.path.join(logdir, 'checkpoints')

    if not args.resume :
        os.makedirs(savePath)

    with open(os.path.join(savePath, 'flags.json'), 'w') as f:
        f.write(json.dumps({ k:v for k,v in vars(args).items()}))
    print('===> Saving state to :' , logdir)


    best_score = 0
    not_improved = 0
    for epoch in range(start_epoch+1, nEpochs+1) :
        scheduler.step(epoch)
        train(epoch)

        recalls = test(epoch)
        if recalls[5] > best_score :
            best_score = recalls[5]
        else :
            not_improved += 1

        save_checkpoint(savePath, {
            'epoch' : epoch,
            'state_dict' : netvlad.state_dict(),
            'recalls' : recalls,
            'best_score' : best_score,
            'optimizer' : optimizer.state_dict(),
        }, recalls[5] > best_score)

        if not_improved > 10 :
            print("Performance don't improve untill 10 epochs")
            break

    
    print("==> Best Recall@5 : {:.4f}".format(best_score), flush=True)
    writer.close()







