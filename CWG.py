import sys, os
sys.path.append(os.path.abspath('..'))

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
from mask import gen_mask
from input import gen_input

from sklearn.cluster import KMeans

import networkx as nx
import random
import hashlib

class GradWhere(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, thrd, device):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        rst = torch.where(input>thrd, torch.tensor(1.0, device=device, requires_grad=True),
                                      torch.tensor(0.0, device=device, requires_grad=True))
        return rst

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        
        """
        Return results number should corresponding with .forward inputs (besides ctx),
        for each input, return a corresponding backward grad
        """
        return grad_input, None, None  

class Generator(nn.Module):
    def __init__(self, sq_dim, feat_dim, layernum, trigger_size, dropout=0.05):
        super(Generator, self).__init__()
        layers = []
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        for l in range(layernum-1):
            layers.append(nn.Linear(sq_dim, sq_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(sq_dim, sq_dim))

        layers_id = []
        if dropout > 0:
            layers_id.append(nn.Dropout(p=dropout))
        for l in range(layernum-1):
            layers_id.append(nn.Linear(sq_dim, sq_dim))
            layers_id.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers_id.append(nn.Dropout(p=dropout))
        layers_id.append(nn.Linear(sq_dim, sq_dim))
        
        self.sq_dim = sq_dim
        self.trigger_size = trigger_size
        self.layers = nn.Sequential(*layers)
        self.layers_id = nn.Sequential(*layers_id)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Linear(1, sq_dim*sq_dim)
        self.mlp1 = nn.Linear(1, sq_dim)
       
               
    def forward(self, args, id, graphs_train, bkd_dr, Ainput_trigger, topomask, bkd_nid_groups, bkd_gids_train, Ainput, Xinput, nodenums_id, 
                nodemax, is_Customized , is_test , trigger_size , device=torch.device('cpu'), binaryfeat=False): 
        seed_string = "my_seed"+str(id)
        seed_value = hash(seed_string) % (2**32)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)

        n = self.sq_dim 
        id_output_temp = torch.tensor(np.random.rand(n, n), dtype=torch.float32)
        id_output = self.layers_id(id_output_temp)
        id_output = torch.sigmoid(id_output)
        
        GW = GradWhere.apply

        edges_len = 0
        rst_bkdA = {}
        for gid in bkd_gids_train:
            rst_bkdA[gid] = torch.mul(Ainput_trigger[gid], id_output)
            rst_bkdA[gid] = self.layers(rst_bkdA[gid]) 

            if args.topo_activation=='relu':
                rst_bkdA[gid] = F.relu(rst_bkdA[gid])
            elif args.topo_activation=='sigmoid':
                rst_bkdA[gid] = torch.sigmoid(rst_bkdA[gid])    

            for_whom='topo'
            if for_whom == 'topo':  
                rst_bkdA[gid] = torch.div(torch.add(rst_bkdA[gid], rst_bkdA[gid].transpose(0, 1)), 2.0)
               
            if for_whom == 'topo' or (for_whom == 'feat' and binaryfeat):
                rst_bkdA[gid] = GW(rst_bkdA[gid], args.topo_thrd, device) 
            rst_bkdA[gid] = torch.mul(rst_bkdA[gid], topomask[gid]) #local_loss
         
        if len(bkd_gids_train) != 0:
            edges_len_avg = edges_len / len(bkd_gids_train)
        else :
            edges_len_avg = 0

        return bkd_dr, bkd_nid_groups, edges_len_avg, self.trigger_size, rst_bkdA
    
def SendtoCUDA(gid, items):
    """
    - items: a list of dict / full-graphs list, 
             used as item[gid] in items
    - gid: int
    """
    cuda = torch.device('cuda')
    for item in items:
        item[gid] = torch.as_tensor(item[gid], dtype=torch.float32).to(cuda)
        
        
def SendtoCPU(gid, items):
    """
    Used after SendtoCUDA, target object must be torch.tensor and already in cuda.
    
    - items: a list of dict / full-graphs list, 
             used as item[gid] in items
    - gid: int
    """
    
    cpu = torch.device('cpu')
    for item in items:
        item[gid] = item[gid].to(cpu)