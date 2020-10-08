#!/usr/bin/env python3
import argparse
import logging
import numpy as np
import os
import sys
import torch
import torch.autograd
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time

from dataset import *
import network_run
from networks.depth_completion import *
from networks.surface_normal import *
from networks.surface_normal_dorn import *

from plane_mask_detection.maskrcnn_benchmark.config import cfg
from plane_mask_detection.demo.predictor import COCODemo

def ParseCmdLineArguments():
    parser = argparse.ArgumentParser(description='MARS CNN Script')
    parser.add_argument('--dataset_pickle_file', type=str, 
            default='/home/nate/datasets/vi_depth_completion/triangulated_merged.pkl')
    return parser.parse_args()


if __name__ == '__main__':
    args = ParseCmdLineArguments()
    args.batch_size = 1

    test_dataset = KinectAzureDataset(usage='test',
                                      dataset_pickle_file=args.dataset_pickle_file,
                                      skip_every_n_image=1)

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=4,
                                 pin_memory=True)
    
    # Nx2 of [triang_depth, gt_depth]
    data = []
    ind = 0
    for i, sample_batched in enumerate(test_dataloader):
        sd = np.squeeze(sample_batched['sparse_depth'].numpy())
        dd = np.squeeze(sample_batched['depth'].numpy())
        print("Loading data ...[%d/%d]" % (i+1,len(test_dataloader)))
        assert sd.shape == dd.shape
        y,x = np.where(np.logical_and(dd > 1e-3,sd > 1e-3))
        for row,col in zip(y,x):
            data.append([sd[row,col],dd[row,col]])
            ind += 1
    data = np.array(data)

    # Formulate least squares problem to model the error as a linear
    # function of triangulated depth
    # argmin_{a,b} || |gt_d-d| - (a*d + b)||^2
    # Stack Ax=b ==> [d 1] . [a b] = gt_d-d
    A = np.ones((data.shape[0],2))
    A[:,0] = data[:,0]
    b = np.abs(data[:,0] - data[:,1])

    ATA = A.T @ A
    ATb = A.T @ b
    x_hat = np.squeeze(np.linalg.inv(ATA) @ ATb)
    print("RESULT: [a_hat, b_hat] = [%f, %f]" % (x_hat[0], x_hat[1]))

    # RESULT: [a_hat, b_hat] = [0.082136, 0.152236]
