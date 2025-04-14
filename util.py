import networkx as nx
import numpy as np
import random
import copy
import torch
from sklearn.model_selection import StratifiedKFold
from mask import gen_mask
from input import gen_input

import torch.nn as nn
criterion = nn.CrossEntropyLoss()

class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        # self.edge_mat = 0 # LOUIS: COMMENTED BUT USED BELLOW? IS THE PART WITH edge_mat BELLOW USEFUL? OR IS IT NEVER USED?

        self.max_neighbor = 0


def load_callgraph():
    import pickle

    dataset = ".//dataset/"
    list_of_malware_graph = pickle.load(open(dataset + 'malware_graphs.p', "rb"))
    list_of_goodware_graph = pickle.load(open(dataset + 'goodware_graphs.p', "rb"))

    print(list_of_malware_graph[0])
    print(list_of_malware_graph[1])

def load_data(dataset, degree_as_tag):
    '''
        dataset: name of dataset
        test_proportion: ratio of test train split
        seed: random seed for random splitting of dataset
    '''

    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}

    with open('./dataset/%s/%s.txt' % (dataset, dataset), 'r') as f:
        n_g = int(f.readline().strip())  
        for i in range(n_g):
            row = f.readline().strip().split() 
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped  
            g = nx.Graph()  
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):  
                g.add_node(j)
                row = f.readline().strip().split()  
                tmp = int(row[1]) + 2  
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if tmp > len(row):
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            assert len(g) == n
            g_list.append(S2VGraph(g, l, node_tags))

    # add labels and edge_mat
    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i] # LOUIS: WTF??
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values()) # LOUIS: NOT USED?
        edge_mat_temp = torch.zeros(len(g.g),len(g.g))
        for [x_i,y_i] in edges:
            edge_mat_temp[x_i,y_i] = 1
        # for x_i in range(len(g.g)):
        #     edge_mat_temp[x_i,x_i] = 1
        g.edge_mat = edge_mat_temp

    if degree_as_tag:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values())
            # using degree of nodes as tags

    # Extracting unique tag labels
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    max_degree=max(tagset)
    tag2index = {i: i for i in range(max_degree+1)}
    #tag2index = {tagset[i]: i for i in range(len(tagset))}

    for g in g_list:
        #g.node_features = torch.zeros(len(g.node_tags), 1)
        #g.node_features[range(len(g.node_tags)), [0]] = 1
        g.node_features = torch.zeros(len(g.node_tags), len(tag2index))
        g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1


    print('# classes: %d' % len(label_dict))
    print('# maximum node tag: %d' % len(tag2index))
    print("# data: %d" % len(g_list))

    return g_list, len(label_dict), tag2index

def separate_data(graph_list, seed, fold_idx):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
 
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed) 
    labels = [graph.label for graph in graph_list]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]

    return train_graph_list, test_graph_list, test_idx # LOUIS: test_idx IS NEVER USED

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
