import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.functional import kl_div
import numpy as np
import random
from tqdm import tqdm
import networkx as nx
import CWG as cwg
from mask import gen_mask
from input import gen_input
from util import *
from graphcnnt import Discriminatort
from graphcnnt1 import Discriminatort1
import pickle
import copy

import matplotlib
import matplotlib.pyplot as plt

criterion = nn.CrossEntropyLoss()

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg
def choose_sub_model(large_param_name):
    if large_param_name == 'eps':
        sub_i = 0
    elif large_param_name[0:4]=='mlps':
        if large_param_name[5:7] == '0.' or large_param_name[5:7] == '1.':
            sub_i = 0
        elif large_param_name[5:7] == '2.' or large_param_name[5:7] == '3.':
            sub_i = 0
        elif large_param_name[5:7] == '4.' or large_param_name[5:7] == '5.':
            sub_i = 1
        elif large_param_name[5:7] == '6.' or large_param_name[5:7] == '7.':
            sub_i = 1
        elif large_param_name[5:7] == '8.' or large_param_name[5:7] == '9.':
            sub_i = 2
        elif large_param_name[5:7] == '10' or large_param_name[5:7] == '11':
            sub_i = 2
        elif large_param_name[5:7] == '12' or large_param_name[5:7] == '13':
            sub_i = 3
        elif large_param_name[5:7] == '14' or large_param_name[5:7] == '15':
            sub_i = 3
    elif large_param_name[0:11] == 'batch_norms':
        if large_param_name[12:14] == '0.' or large_param_name[12:14] == '1.':
            sub_i = 0
        elif large_param_name[12:14] == '2.' or large_param_name[12:14] == '3.':
            sub_i = 0
        elif large_param_name[12:14] == '4.' or large_param_name[12:14] == '5.':
            sub_i = 1
        elif large_param_name[12:14] == '6.' or large_param_name[12:14] == '7.':
            sub_i = 1
        elif large_param_name[12:14] == '8.' or large_param_name[12:14] == '9.':
            sub_i = 2
        elif large_param_name[12:14] == '10' or large_param_name[12:14] == '11':
            sub_i = 2
        elif large_param_name[12:14] == '12' or large_param_name[12:14] == '13':
            sub_i = 3
        elif large_param_name[12:14] == '14' or large_param_name[12:14] == '15':
            sub_i = 3
    elif large_param_name[0:18] == 'linears_prediction':
        if large_param_name[19:21] == '0.' or large_param_name[19:21] == '1.':
            sub_i = 0
        elif large_param_name[19:21] == '2.' or large_param_name[19:21] == '3.':
            sub_i = 0
        elif large_param_name[19:21] == '4.': 
            sub_i = 0
        elif large_param_name[19:21] == '6.' or large_param_name[19:21] == '7.':
            sub_i = 1
        elif large_param_name[19:21] == '8.' or large_param_name[19:21] == '9.':
            sub_i = 1
        elif large_param_name[19:21] == '5.': 
            sub_i = 1
        elif large_param_name[19:21] == '12' or large_param_name[19:21] == '13':
            sub_i = 2
        elif large_param_name[19:21] == '14' or large_param_name[19:21] == '10':
            sub_i = 2
        elif large_param_name[19:21] == '11': 
            sub_i = 2
        elif large_param_name[19:21] == '18' or large_param_name[19:21] == '19':
            sub_i = 3
        elif large_param_name[19:21] == '17' or large_param_name[19:21] == '16':
            sub_i = 3
        elif large_param_name[19:21] == '15':
            sub_i = 3
    return sub_i

def train_sub(args, global_model, sub_model, optimizer, device, train_graphs, epoch, tag2index):
    total_iters = args.iters_per_epoch

    loss_accum = 0
    for pos in range(total_iters):
        selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]

        batch_graph = [train_graphs[idx] for idx in selected_idx]
        labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)
        loss = 0
        for i in range(len(sub_model)):
            sub_model[i].train()
            output = sub_model[i](batch_graph, i)
            loss = criterion(output, labels)

            if optimizer[i] is not None:
                optimizer[i].zero_grad()
                loss.backward()
                optimizer[i].step()                     
    return loss

def train_G(args, model, sub_model, generator, optimizer_G, id, device, train_graphs_trigger, epoch, tag2index, bkd_gids_train, Ainput_train, Xinput_train, nodenums_id, nodemax, binaryfeat=False):
    for i in range(len(sub_model)):
        sub_model[i].eval()
    generator.train()
    optimizer_G.zero_grad()
    
    total_iters = args.iters_per_epoch

    for pos in range(total_iters):
        selected_idx = bkd_gids_train 
        
        batch_graph = [train_graphs_trigger[idx] for idx in selected_idx]

        bkd_nid_groups = {}
        graphs = copy.deepcopy(train_graphs_trigger)
        for gid in bkd_gids_train:
            if nodenums_id[gid] >= args.triggersize:
                bkd_nid_groups[gid] = np.random.choice(nodenums_id[gid],args.triggersize,replace=False)
            else:
                bkd_nid_groups[gid] = np.random.choice(nodenums_id[gid],args.triggersize,replace=True)
            
        init_dr = init_trigger(
                        args, graphs, bkd_gids_train, bkd_nid_groups, 0.0)
        bkd_dr = copy.deepcopy(init_dr)
        topomask, featmask = gen_mask(
                        graphs[0].node_features.shape[1], nodemax, bkd_dr, bkd_gids_train, bkd_nid_groups)
        Ainput_trigger, Xinput_trigger = gen_input(bkd_dr, bkd_gids_train, nodemax)
    
        output_graph, trigger_group, edges_len, nodes_len, rst_bkdA  = generator(args, id, train_graphs_trigger, bkd_dr, Ainput_trigger, topomask, bkd_nid_groups, bkd_gids_train, Ainput_train, Xinput_train, nodenums_id, nodemax, args.is_Customized, args.is_test, args.triggersize, device=torch.device('cpu'), binaryfeat=False)

        for gid in bkd_gids_train:
            output_graph[gid].edge_mat = torch.add(init_dr[gid].edge_mat, rst_bkdA[gid][:nodenums_id[gid], :nodenums_id[gid]]) 
            for i in range(nodenums_id[gid]):
                for j in range(nodenums_id[gid]):
                    if rst_bkdA[gid][i][j] == 1:
                        output_graph[gid].g.add_edge(i, j)
            output_graph[gid].node_tags = list(dict(output_graph[gid].g.degree).values())   

        for i in range(len(sub_model)):
            if i == 0:
                output = sub_model[i](output_graph, i)
            else: 
                output = torch.add(output, sub_model[i](output_graph, i))
        output_graph_poison = torch.stack([output[idx] for idx in selected_idx])

        labels_poison = torch.LongTensor([args.target for idx in selected_idx]).to(device)
        loss_poison = criterion(output_graph_poison, labels_poison)    
    loss_poison.backward()
    optimizer_G.step()
    average_loss = 0
    aver_loss_poison = 0
    return average_loss, aver_loss_poison, edges_len, nodes_len

def train_D_sub(args, global_model, sub_model, generator, optimizer, id, device, train_graphs, epoch, tag2index, bkd_gids_train, Ainput_train, Xinput_train, nodenums_id, nodemax, binaryfeat=False):
    for i in range(len(sub_model)):
        sub_model[i].train()
    total_iters = args.iters_per_epoch
    generator.eval()

    selected_idx = bkd_gids_train
    batch_graph = [train_graphs[idx] for idx in selected_idx]

    bkd_nid_groups = {}
    graphs = copy.deepcopy(train_graphs)
    for gid in bkd_gids_train:
        if nodenums_id[gid] >= args.triggersize:
            bkd_nid_groups[gid] = np.random.choice(nodenums_id[gid],args.triggersize,replace=False)
        else:
            bkd_nid_groups[gid] = np.random.choice(nodenums_id[gid],args.triggersize,replace=True)
        
    init_dr = init_trigger(
                    args, graphs, bkd_gids_train, bkd_nid_groups, 0.0)
    bkd_dr = copy.deepcopy(init_dr)
    topomask, featmask = gen_mask(
                    graphs[0].node_features.shape[1], nodemax, bkd_dr, bkd_gids_train, bkd_nid_groups)
    Ainput_trigger, Xinput_trigger = gen_input(bkd_dr, bkd_gids_train, nodemax)
    
    output_graph, _, _, _, rst_bkdA = generator(args, id, train_graphs, bkd_dr, Ainput_trigger, topomask, bkd_nid_groups, bkd_gids_train, Ainput_train, Xinput_train, nodenums_id, nodemax, args.is_Customized, args.is_test, args.triggersize, device=torch.device('cpu'), binaryfeat=False)#.detach()
    
    for gid in bkd_gids_train:
        output_graph[gid].edge_mat = torch.add(init_dr[gid].edge_mat, rst_bkdA[gid][:nodenums_id[gid], :nodenums_id[gid]]).detach() 
        for i in range(nodenums_id[gid]):
            for j in range(nodenums_id[gid]):
                if rst_bkdA[gid][i][j] == 1:
                    output_graph[gid].g.add_edge(i, j)
        output_graph[gid].node_tags = list(dict(output_graph[gid].g.degree).values())

    loss_accum = 0
    #for pos in pbar:
    for pos in range(total_iters):
        for i in range(len(sub_model)):
            
            labels = torch.LongTensor([graph.label for graph in output_graph]).to(device)
            # loss = 0
            output = sub_model[i](output_graph, i)

            # compute loss
            loss = criterion(output, labels)

            if optimizer[i] is not None:
                optimizer[i].zero_grad()
                loss.backward()
                optimizer[i].step()     
    return loss
def pass_data_iteratively1(model, graphs, model_id, minibatch_size=1):
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i + minibatch_size]
        if len(sampled_idx) == 0:
            continue
        output.append(model([graphs[j] for j in sampled_idx], model_id).detach())
    return torch.cat(output, 0)   

def bkd_cdd_test(graphs, target_label):
    
    backdoor_graphs_indexes = []
    for graph_idx in range(len(graphs)):
        if graphs[graph_idx].label != target_label: # != target_label:
            backdoor_graphs_indexes.append(graph_idx)
        
    return backdoor_graphs_indexes

def bkd_cdd(graphs, target_label, dataset):

    if dataset == 'MUTAG':
        num_backdoor_train_graphs = 1 # value given by the authors of the paper
    elif dataset == 'PROTEINS':
        num_backdoor_train_graphs = 1 # value copied from MUTAG. TODO: verify if the value makes sense for this dataset
    elif dataset == 'DD':
        num_backdoor_train_graphs = 1 # value copied from MUTAG. TODO: verify if the value makes sense for this dataset
    elif dataset == 'COLLAB':
        num_backdoor_train_graphs = 1 # value copied from MUTAG. TODO: verify if the value makes sense for this dataset
    else:
        raise Exception("Undefined number of backdoors for this dataset")

    temp_n = 0
    backdoor_graphs_indexes = []
    for graph_idx in range(len(graphs)):
        if graphs[graph_idx].label != target_label and temp_n < num_backdoor_train_graphs:
            backdoor_graphs_indexes.append(graph_idx)
            temp_n += 1
           
    return backdoor_graphs_indexes

def test_ensemble(args, model, device, test_graphs, tag2index):
    if args.dataset == 'MUTAG':
        num_labels = 2 # value given by the authors of the paper
    if args.dataset == 'PROTEINS':
        num_labels = 2 # value given in data_preprocessing or loading_data (i.e. num of classes)
    if args.dataset == 'DD':
        num_labels = 2 # value given in data_preprocessing or loading_data (i.e. num of classes)
    if args.dataset == 'COLLAB':
        num_labels = 3 # value given in data_preprocessing or loading_data (i.e. num of classes)
    output = {}
    pred = {}
    model[0].eval()
    pred_ens_g_tmp = pass_data_iteratively1(model[0], test_graphs, 0)
    for i in range(len(model)):
        model[i].eval()
        output[i] = pass_data_iteratively1(model[i], test_graphs, i)
        if i > 0 :
            pred_ens_g_tmp += output[i]
        pred[i] = output[i].max(1, keepdim=True)[1]
    pred_ens = pred[0]
    sub_test_label = {}
    for i in range(len(pred[i])):
        for k in range(num_labels):
            sub_test_label[k] = 0
        for j in range(len(model)):
            sub_test_label[int(pred[j][i])] += 1
        pred_ens[i] = max(sub_test_label, key=sub_test_label.get)

    labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
    correct = pred_ens.eq(labels.view_as(pred_ens)).sum().cpu().item()
    acc_test = correct / float(len(test_graphs))

    

    return acc_test

def init_trigger(args, x, bkd_gids: list, bkd_nid_groups: list, init_feat: float):
    if init_feat == None:
        init_feat = - 1
        print('init feat == None, transferred into -1')

    graphs = copy.deepcopy(x)   
    for idx in bkd_gids:
        for i in bkd_nid_groups[idx]:  
            for j in bkd_nid_groups[idx]:
                if graphs[idx].edge_mat[i, j] == 1:
                    # edges.remove([i, j])
                    graphs[idx].edge_mat[i, j] = 0
                if (i, j) in graphs[idx].g.edges():
                    graphs[idx].g.remove_edge(i, j)
        
        assert args.target is not None
        graphs[idx].label = args.target
        graphs[idx].node_tags = list(dict(graphs[idx].g.degree).values()) 
        # change features in-place
        featdim = graphs[idx].node_features.shape[1]
        a = np.array(graphs[idx].node_features)
        a[bkd_nid_groups[idx]] = np.ones((len(bkd_nid_groups[idx]), featdim)) * init_feat
        graphs[idx].node_features = torch.Tensor(a.tolist())
            
    return graphs  

# transfer the parameters from global model to submodels
def global_to_sub(global_model,sub_model):
    global_model_state_dict = global_model.state_dict()
    sub_model_state_dict = {}
    for i in range(len(sub_model)):
        sub_model_state_dict[i] = sub_model[i].state_dict()

    for large_param_name, large_param in global_model.named_parameters():
        sub_i = choose_sub_model(large_param_name)        
        sub_model_state_dict[sub_i][large_param_name].data.copy_(global_model_state_dict[large_param_name].data)
    
    for i in range(len(sub_model)):
        sub_model[i].load_state_dict(sub_model_state_dict[i])

# transfer the parameters from submodels to global model
def sub_to_global(global_model,sub_model):
    global_model_state_dict = global_model.state_dict()
    sub_model_state_dict = {}
    for i in range(len(sub_model)):
        sub_model_state_dict[i] = sub_model[i].state_dict()

    for large_param_name, large_param in global_model.named_parameters():
        sub_i = choose_sub_model(large_param_name)
        global_model_state_dict[large_param_name].data.copy_(sub_model_state_dict[sub_i][large_param_name].data)
    
    global_model.load_state_dict(global_model_state_dict)

def main():
    # Training settings
    # Note: Hyper-parameters need to be tuned in order to obtain results reported in the paper.
    parser = argparse.ArgumentParser(
        description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--port', type=str, default="acm4",
                        help='name of sever')
    parser.add_argument('--dataset', type=str, default="MUTAG",
                        help='name of dataset (default: MUTAG)') # COLLAB and DD, can't be trained with a free studio. You need more than 16GB RAM
    parser.add_argument('--backdoor', action='store_true', default=True,
                        help='Backdoor GNN')
    parser.add_argument('--attack', type=str, default="none",
                        help='Type of attacks. Possible values are: {none, distillation, finetuning, layerperturb} where "none" conducts no attack. They may be comma concatenated to apply multiple attacks against the global model.')
    parser.add_argument('--graphtype', type=str, default='ER',
                        help='type of graph generation')
    parser.add_argument('--prob', type=float, default=1.0,
                        help='probability for edge creation/rewiring each edge')
    parser.add_argument('--K', type=int, default=4,
                        help='Each node is connected to k nearest neighbors in ring topology')
    parser.add_argument('--num_agents', type=int, default=20,
                        help="number of agents:n")
    parser.add_argument('--T', type=int, default=4,
                        help='number of sub models')
    parser.add_argument('--num_corrupt', type=int, default=5,
                        help="number of corrupt agents") # clients that will watermark their local models
    parser.add_argument('--frac', type=float, default=0.1, 
                        help='fraction of training graphs are backdoored')
    parser.add_argument('--frac_epoch', type=float, default=0.5,
                        help='fraction of users are chosen') 
    parser.add_argument('--is_Customized', type=int, default=0,
                        help='is_Customized') 
    parser.add_argument('--is_test', type=int, default=0,
                        help='is_test')   
    parser.add_argument('--n_epoch', type=int, default=5,
                        help='n_epoch') 
    parser.add_argument('--n_test', type=int, default=1,
                        help='n_test')                                         
    parser.add_argument('--triggersize', type=int, default=4,
                        help='number of nodes in a clique (trigger size)')
    parser.add_argument('--target', type=int, default=0,
                        help='targe class (default: 0)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=128,  
                        help='input batch size for training (default: 32)')
    parser.add_argument('--iters_per_epoch', type=int, default=1,
                        help='number of iterations per each epoch (default: 50)') # no, I didn't change the default... The authors were just not consistent.
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 350)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--n_train_D', type=int, default=1,
                        help='training rounds')
    parser.add_argument('--n_train_G', type=int, default=1,
                        help='training rounds')   
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument('--fold_idx', type=int, default=0,
                        help='the index of fold in 10-fold validation. Should be less then 10.')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "average"],
                        help='Pooling for over nodes in a graph: sum or average')
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average", "max"],
                        help='Pooling for over neighboring nodes: sum, average or max')
    parser.add_argument('--learn_eps', action="store_true", default=False,
                        help='Whether to learn the epsilon weighting for the center nodes. Does not affect training accuracy though.')
    parser.add_argument('--degree_as_tag', action="store_true",
                        help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
    parser.add_argument('--topo_thrd', type=float, default=0.5, 
                       help="threshold for topology generator")
    parser.add_argument('--gtn_layernum', type=int, default=3, 
                        help="layer number of CWG")
    parser.add_argument('--topo_activation', type=str, default='sigmoid', 
                        help="activation function for topology generator")
    parser.add_argument('--feat_activation', type=str, default='relu', 
                       help="activation function for feature generator")
    parser.add_argument('--feat_thrd', type=float, default=0, 
                       help="threshold for feature generator (only useful for binary feature)")
    parser.add_argument('--filename', type=str, default="output",
                        help='output file')
    parser.add_argument('--filenamebd', type=str, default="output_bd",
                        help='output backdoor file')
    args = parser.parse_args()

    cpu = torch.device('cpu')
    # set up seeds and gpu device
    torch.manual_seed(0)  
    np.random.seed(0) 
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    # graphs, num_classes, tag2index = load_data(args.dataset, args.degree_as_tag)
    graphs, num_classes, tag2index = load_data(args.dataset, args.degree_as_tag) # build list of networkx graphs based on the dataset (i.e. if we want to test with other datasets, we need them to have the same format as MUTAG.txt)

    train_graphs, test_graphs, test_idx = separate_data(graphs, args.seed, args.fold_idx) # split list into training and test sets.
    # train_graphs1 = copy.deepcopy(train_graphs) # NEVER USED?
    print('#train_graphs:', len(train_graphs), '#test_graphs:', len(test_graphs))

    test_cleangraph_backdoor_labels = [graph for graph in test_graphs if graph.label == 1] # test_cleangraph_backdoor_labels is never used...
    print('#test clean graphs:', len(test_cleangraph_backdoor_labels))

    print('input dim:', train_graphs[0].node_features.shape[1])
    
    train_data_size = len(train_graphs) # Total number of training graphs
    client_data_size=int(train_data_size/(args.num_agents)) # Number of training graphs per client
    split_data_size = [client_data_size for i in range(args.num_agents-1)]
    split_data_size.append(train_data_size-client_data_size*(args.num_agents-1))
    train_graphs = torch.utils.data.random_split(train_graphs,split_data_size) # Assign training graphs to clients randomly
                  
    global_model = Discriminatort(args, args.num_layers, args.num_mlp_layers, train_graphs[0][0].node_features.shape[1],
                        args.hidden_dim, \
                        num_classes, args.final_dropout, args.learn_eps, args.graph_pooling_type,
                        args.neighbor_pooling_type, device).to(device) # Create global model
    optimizer_D = optim.Adam(global_model.parameters(), lr=args.lr)
    scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=50, gamma=0.1)

    test_graphs_trigger = copy.deepcopy(test_graphs)
    test_backdoor = bkd_cdd_test(test_graphs_trigger, args.target)
    nodenums = [len(graphs[idx].g.adj) for idx in range(len(graphs))]
    nodemax = max(nodenums)
    featdim = train_graphs[0][0].node_features.shape[1]
    
    Ainput_test, Xinput_test = gen_input(test_graphs_trigger, test_backdoor, nodemax) 
    
    with open(args.filenamebd, 'w+') as f: # Useless file, it is not used
        f.write("acc\n")
        bkd_gids_train = {}
        Ainput_train = {}
        Xinput_train = {}
        nodenums_id = {}
        train_graphs_trigger = {}
        
        for id in range(args.num_corrupt): # For every client taking part in the watermwarking, ...? 
            train_graphs_trigger[id] = copy.deepcopy(train_graphs[id])
            nodenums_id[id] = [len(train_graphs_trigger[id][idx].g.adj) for idx in range(len(train_graphs_trigger[id]))]
            bkd_gids_train[id] = bkd_cdd(train_graphs_trigger[id], args.target, args.dataset)
            Ainput_train[id], Xinput_train[id] = gen_input(train_graphs_trigger[id], bkd_gids_train[id], nodemax)
        global_weights = global_model.state_dict()
        generator = {}
        optimizer_G = {}
        for cwg_id in range(args.num_corrupt): # For every client taking part in the watermarking, we create a Customized Watermark Generator (CWG)
            generator[cwg_id]=cwg.Generator(nodemax, featdim, args.gtn_layernum, args.triggersize)
            optimizer_G[cwg_id]= optim.Adam(generator[cwg_id].parameters(), lr=args.lr)

        sub_model = {}
        sub_hidden_dim = args.hidden_dim
        optimizer_sub = {}
        scheduler_sub = {}
        for i in range(args.T): # Create each submodel (each client as multiple submodels), 
            sub_model[i] = Discriminatort1(args, args.num_layers, args.num_mlp_layers, train_graphs[0][0].node_features.shape[1],
                        sub_hidden_dim, \
                        num_classes, args.final_dropout, args.learn_eps, args.graph_pooling_type,
                        args.neighbor_pooling_type, device).to(device)
            optimizer_sub[i] = optim.Adam(sub_model[i].parameters(), lr=args.lr)
            scheduler_sub[i] = optim.lr_scheduler.StepLR(optimizer_sub[i], step_size=50, gamma=0.1)
        for epoch in tqdm(range(1, args.epochs + 1)): # Train global model (as well as the submodels)
            global_weights = global_model.state_dict() 
            local_weights, local_losses = [], []
            m = max(int(args.frac_epoch * args.num_agents), 1)
            idxs_users = np.random.choice(range(args.num_agents), m, replace=False) # Select random clients for each epoch (authors don't say why? for simulation purposes?)
            print("idxs_users:", idxs_users)
            for id in idxs_users: # For each select client
                global_model.load_state_dict(copy.deepcopy(global_weights))
                global_to_sub(global_model, sub_model)
                if id < args.num_corrupt: # If the given client is taking part to the watermarking
                    for kk in range(args.n_train_D): # For n_train_D training rounds, trains the submodels on watermarked graphs 
                        loss_sub = train_D_sub(args, global_model, sub_model, generator[id], optimizer_sub, id, device, train_graphs_trigger[id], 
                                        epoch, tag2index, bkd_gids_train[id], Ainput_train[id], 
                                        Xinput_train[id], nodenums_id[id], nodemax, 
                                        binaryfeat=False) 
                    if epoch % args.n_epoch == 0: # Every n_epoch epoch, trains the generator model to produce watermarked graphs (??? isn't it too late? For the first epoc, the generator is not trained?)
                        for kk in range(args.n_train_G):
                            loss, loss_poison, edges_len, nodes_len = train_G(args, global_model, sub_model, generator[id], optimizer_G[id], id, device, train_graphs_trigger[id], 
                                            epoch, tag2index, bkd_gids_train[id], Ainput_train[id], 
                                            Xinput_train[id], nodenums_id[id], nodemax, 
                                            binaryfeat=False) 
                else:
                    loss_sub = train_sub(args, global_model, sub_model, optimizer_sub, device, train_graphs[id], epoch, tag2index) # Traing submodels on clean (non-watermarked) graphs
                sub_to_global(global_model, sub_model) 
                l_weights = global_model.state_dict()
                local_weights.append(l_weights)
            
            for sch_i in range(args.T):
                scheduler_sub[sch_i].step() 
            global_weights = average_weights(local_weights)   
            global_model.load_state_dict(global_weights) # Update global model with the average weights of the submodels
            #----------------- Evaluation -----------------#
            if epoch % args.n_test == 0: # Each n_test epoch, we print the current accuracies (MA + WA)
                global_to_sub(global_model, sub_model)
                id = 0
                args.is_test = 1
                nodenums_test = [len(test_graphs[idx].g.adj) for idx in range(len(test_graphs))]
                generator[id].eval() #
                bkd_nid_groups = {}
                graphs = copy.deepcopy(test_graphs) # Clean (not watermarked) graphs
                for gid in test_backdoor: # Select some random watermarked graphs
                    if nodenums_test[gid] >= args.triggersize:
                        bkd_nid_groups[gid] = np.random.choice(nodenums_test[gid],args.triggersize,replace=False)
                    else:
                        bkd_nid_groups[gid] = np.random.choice(nodenums_test[gid],args.triggersize,replace=True)
                    
                init_dr = init_trigger(
                                args, graphs, test_backdoor, bkd_nid_groups, 0.0)
                bkd_dr = copy.deepcopy(init_dr)
                topomask, featmask = gen_mask(
                                graphs[0].node_features.shape[1], nodemax, bkd_dr, test_backdoor, bkd_nid_groups)
                Ainput_trigger, Xinput_trigger = gen_input(bkd_dr, test_backdoor, nodemax)

                bkd_dr_test, bkd_nid_groups_test, _, _, rst_bkdA= generator[id](args, id, graphs, bkd_dr, Ainput_trigger, topomask, bkd_nid_groups, test_backdoor, Ainput_test, Xinput_test, nodenums_test, nodemax, args.is_Customized, args.is_test, args.triggersize, device=torch.device('cpu'), binaryfeat=False)
                
                for gid in test_backdoor:
                    bkd_dr_test[gid].edge_mat = torch.add(init_dr[gid].edge_mat, rst_bkdA[gid][:nodenums_test[gid], :nodenums_test[gid]])
                    for i in range(nodenums_test[gid]):
                        for j in range(nodenums_test[gid]):
                            if rst_bkdA[gid][i][j] == 1:
                                bkd_dr_test[gid].g.add_edge(i, j)
                    bkd_dr_test[gid].node_tags = list(dict(bkd_dr_test[gid].g.degree).values())
                for gid in test_backdoor: 
                    for i in bkd_nid_groups_test[gid]:
                        for j in bkd_nid_groups_test[gid]:
                            if i != j:
                                bkd_dr_test[gid].edge_mat[i][j] = 1
                                if (i,j) not in bkd_dr_test[gid].g.edges():
                                    bkd_dr_test[gid].g.add_edge(i, j)
                                                        
                    bkd_dr_test[gid].node_tags = list(dict(bkd_dr_test[gid].g.degree).values())

                acc_test_clean = test_ensemble(args, sub_model, device, test_graphs, tag2index)
                print("accuracy test clean (MA): %f" % acc_test_clean)
                bkd_dr_ = [bkd_dr_test[idx] for idx in test_backdoor]
                test_watermark = test_ensemble(args, sub_model, device, bkd_dr_, tag2index)
                print("accuracy test watermark (WA): %f" % test_watermark)

                f.flush()
            #scheduler.step()  

    f = open('./saved_model/' + str(args.graphtype) + '_' + str(args.dataset) + '_' + str(
            args.frac) + '_triggersize_' + str(args.triggersize), 'wb')

    pickle.dump(global_model, f)
    f.close()


    #----------------- Evaluation under attack -----------------#
    
    print("Evaluating model under attacks...")
    
    # Create a copy of the global model for attack testing
    attack_model = copy.deepcopy(global_model)
    
    attacks = args.attack.split(",")
    # Test different types of attacks based on args.attack parameter
    if "distillation" in attacks:
        # Knowledge distillation attack
        print("Performing distillation attack...")
        
        # Create a smaller student model with fewer parameters
        student_model = Discriminatort(args, max(2, args.num_layers-2), args.num_mlp_layers, 
                                      train_graphs[0][0].node_features.shape[1],
                                      args.hidden_dim // 2, num_classes, args.final_dropout,
                                      args.learn_eps, args.graph_pooling_type,
                                      args.neighbor_pooling_type, device).to(device)
        
        # Train student model to mimic the teacher (global_model)
        optimizer_student = optim.Adam(student_model.parameters(), lr=args.lr)
        
        # Distillation temperature
        temp = 4.0
        
        for distill_epoch in range(30):  # Usually fewer epochs needed for distillation
            # Get logits from teacher model
            global_to_sub(global_model, sub_model)
            
            # Distillation on clean training data
            for id in range(args.num_agents):
                if id >= args.num_corrupt:  # Only use clean data for distillation
                    for batch_idx in range(args.iters_per_epoch):
                        selected_idx = np.random.permutation(len(train_graphs[id]))[:args.batch_size]
                        batch_graph = [train_graphs[id][idx] for idx in selected_idx]
                        
                        # Get teacher outputs
                        with torch.no_grad():
                            teacher_outputs = []
                            for i in range(len(sub_model)):
                                if i == 0:
                                    teacher_output = sub_model[i](batch_graph, i)
                                else:
                                    teacher_output = torch.add(teacher_output, sub_model[i](batch_graph, i))
                            
                            # Apply temperature scaling to soften probabilities
                            teacher_outputs = F.softmax(teacher_output / temp, dim=1)
                        
                        # Train student to match teacher outputs
                        student_model.train()
                        optimizer_student.zero_grad()
                        student_output = student_model(batch_graph)
                        student_output_soft = F.log_softmax(student_output / temp, dim=1)
                        
                        # Compute distillation loss
                        loss_distill = F.kl_div(student_output_soft, teacher_outputs, reduction='batchmean') * (temp * temp)
                        
                        # Also add standard cross-entropy loss with true labels
                        labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)
                        loss_ce = criterion(student_output, labels)
                        
                        # Combined loss
                        loss = 0.7 * loss_distill + 0.3 * loss_ce
                        loss.backward()
                        optimizer_student.step()
            
            if distill_epoch % 5 == 0:
                print(f"Distillation epoch {distill_epoch}, Loss: {loss.item()}")
        
        # Evaluate the distilled model
        print("Evaluating distilled model:")
        
        # Function to test with student model
        def test_student_model(model, device, test_graphs):
            model.eval()
            output = []
            idx = np.arange(len(test_graphs))
            for i in range(0, len(test_graphs), 1):
                sampled_idx = idx[i:i+1]
                if len(sampled_idx) == 0:
                    continue
                output.append(model([test_graphs[j] for j in sampled_idx]).detach())
            output = torch.cat(output, 0)
            pred = output.max(1, keepdim=True)[1]
            labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
            correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
            return correct / float(len(test_graphs))
        
        acc_test_clean_distill = test_student_model(student_model, device, test_graphs)
        print("Distilled model accuracy on clean test data (MA): %f" % acc_test_clean_distill)
        
        bkd_dr_ = [bkd_dr_test[idx] for idx in test_backdoor]
        acc_test_watermark_distill = test_student_model(student_model, device, bkd_dr_)
        print("Distilled model accuracy on watermarked data (WA): %f" % acc_test_watermark_distill)
    
    elif "finetuning" in attacks:
        # Fine-tuning attack
        print("Performing fine-tuning attack...")
        
        # Copy the trained model
        finetuned_model = copy.deepcopy(global_model)
        optimizer_finetune = optim.Adam(finetuned_model.parameters(), lr=args.lr * 0.1)  # Lower learning rate
        
        # Fine-tune on a subset of clean data
        clean_graphs = []
        for id in range(args.num_agents):
            if id >= args.num_corrupt:
                clean_graphs.extend(train_graphs[id])
        
        # Randomly select 20% of clean data for fine-tuning
        num_finetune = max(int(0.2 * len(clean_graphs)), 1)
        finetune_indices = np.random.choice(len(clean_graphs), num_finetune, replace=False)
        finetune_graphs = [clean_graphs[i] for i in finetune_indices]
        
        # Perform fine-tuning for fewer epochs
        for finetune_epoch in range(10):
            finetuned_model.train()
            for batch_idx in range(args.iters_per_epoch * 2):  # More iterations per epoch
                selected_idx = np.random.permutation(len(finetune_graphs))[:min(args.batch_size, len(finetune_graphs))]
                batch_graph = [finetune_graphs[idx] for idx in selected_idx]
                
                optimizer_finetune.zero_grad()
                output = finetuned_model(batch_graph)
                labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)
                loss = criterion(output, labels)
                loss.backward()
                optimizer_finetune.step()
            
            if finetune_epoch % 2 == 0:
                print(f"Fine-tuning epoch {finetune_epoch}, Loss: {loss.item()}")
        
        # Evaluate the fine-tuned model
        print("Evaluating fine-tuned model:")
        global_to_sub(finetuned_model, sub_model)
        
        acc_test_clean_finetune = test_ensemble(args, sub_model, device, test_graphs, tag2index)
        print("Fine-tuned model accuracy on clean test data (MA): %f" % acc_test_clean_finetune)
        
        bkd_dr_ = [bkd_dr_test[idx] for idx in test_backdoor]
        acc_test_watermark_finetune = test_ensemble(args, sub_model, device, bkd_dr_, tag2index)
        print("Fine-tuned model accuracy on watermarked data (WA): %f" % acc_test_watermark_finetune)
    
    # doesn't act as expected... MA stays always the same... As if the perturbation wouldn't affect the performance
    # ...
    elif "layerperturb" in attacks : 
        # TODO
        acc_test_clean_perturb = 0
        acc_test_watermark_perturb = 0
        pass
    
    elif args.attack == "none":
        # No attack, already evaluated in the previous section
        print("No attack performed, using previous evaluation results")
    
    else:
        print(f"Unknown attack type: {args.attack}")
    
    # Write attack results to file
    with open(args.filename + "_attack_" + args.attack + ".txt", 'w') as attack_file:
        attack_file.write("Attack Type: " + args.attack + "\n")
        if args.attack == "distillation":
            attack_file.write(f"Clean Accuracy (MA): {acc_test_clean_distill}\n")
            attack_file.write(f"Watermark Accuracy (WA): {acc_test_watermark_distill}\n")
        elif args.attack == "finetuning":
            attack_file.write(f"Clean Accuracy (MA): {acc_test_clean_finetune}\n")
            attack_file.write(f"Watermark Accuracy (WA): {acc_test_watermark_finetune}\n")
        elif args.attack == "layerperturb":
            attack_file.write(f"Clean Accuracy (MA): {acc_test_clean_perturb}\n")
            attack_file.write(f"Watermark Accuracy (WA): {acc_test_watermark_perturb}\n")
        elif args.attack == "none":
            attack_file.write(f"Clean Accuracy (MA): {acc_test_clean}\n")
            attack_file.write(f"Watermark Accuracy (WA): {test_watermark}\n")


if __name__ == '__main__':
    main()