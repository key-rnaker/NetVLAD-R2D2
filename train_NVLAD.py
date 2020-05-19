from __future__ import print_function
import argparse

parser = argparse.ArgumentParser(description='NetVLAD-R2D2')
parser.add_argument('--batchSize', type=int, default=4)
parser.add_argument('--nEpochs', type=int, default=30)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--dataPath', type=str, default='/')
parser.add_argument('--savePath', type=str, default='checkpoints/')
