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
from attack_distillation import run_distillation_attack
from attack_finetuning import run_finetuning_attack
from attack_layerperturb import run_layerperturb_attack
from main_malicious import run_malicious_training


import matplotlib
import matplotlib.pyplot as plt

criterion = nn.CrossEntropyLoss()

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
                        help='Type of attacks. Possible values are: {none, distillation, finetuning, layerperturb} where none conducts no attack.')
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
    parser.add_argument('--perturb_depth', type=int, default=1,
                    help='number of final layers to perturb per submodel')
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
    parser.add_argument('--malicious_frac', type=float, default=0.3,
                    help='Fraction of malicious clients that suppress watermark (0-1)')

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
    
    ma_list = []
    wa_list = []
    eval_epochs = []

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

                ma_list.append(acc_test_clean)
                wa_list.append(test_watermark)
                eval_epochs.append(epoch)
                f.flush()
            #scheduler.step()  

    f = open('./saved_model/' + str(args.graphtype) + '_' + str(args.dataset) + '_' + str(
            args.frac) + '_triggersize_' + str(args.triggersize), 'wb')

    pickle.dump(global_model, f)
    f.close()

    plt.figure(figsize=(10, 6))
    plt.plot(eval_epochs, ma_list, label='MA (Main Accuracy)', marker='o')
    plt.plot(eval_epochs, wa_list, label='WA (Watermarked Accuracy)', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy (MA & WA) over training')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('ma_wa_accuracy_plot.png')  # Save the plot
    plt.show()  # Display the plot


    #----------------- Evaluation under attack -----------------#
    
    print("Evaluating model under attacks...")
    
    # Create a copy of the global model for attack testing
    attack_model = copy.deepcopy(global_model)
    
    clean_model = Discriminatort(
        args, args.num_layers, args.num_mlp_layers, 
        train_graphs[0][0].node_features.shape[1],
        args.hidden_dim, num_classes,
        args.final_dropout, args.learn_eps,
        args.graph_pooling_type, args.neighbor_pooling_type,
        device
    ).to(device)
    
    # Test different types of attacks based on args.attack parameter
    
    if args.attack =="distillation":
        # acc_test_clean_distill, acc_test_watermark_distill = run_distillation_attack(
        #     args, attack_model, sub_model, train_graphs, test_graphs, test_backdoor, bkd_dr_test, num_classes, device
        # )
        acc_test_clean_distill, acc_test_watermark_distill = run_distillation_attack(
            args, attack_model, sub_model, train_graphs, test_graphs, test_backdoor, bkd_dr_test, num_classes, device, tag2index
        )
    
    elif args.attack=="finetuning":
        acc_test_clean_finetune, acc_test_watermark_finetune = run_finetuning_attack(
            args, attack_model, train_graphs, test_graphs, test_backdoor, bkd_dr_test, num_classes, sub_model, tag2index, device
        )
        
    elif args.attack == "layerperturb":
    # Définir les couches à perturber (par défaut les premières couches de prediction)
        layers_to_replace = [
            f"linears_prediction.{i}" for i in range(args.perturb_depth)
        ]

        # Exécuter l’attaque
        perturbed_model, acc_test_clean_lp, acc_test_watermark_lp = run_layerperturb_attack(
            model_w=attack_model,
            model_clean=clean_model,
            test_graphs=test_graphs,
            watermark_graphs=bkd_dr_test,
            device=device,
            layers_to_replace=layers_to_replace
        )
        
    elif args.attack == "falsification":
        acc_test_clean, acc_test_watermark = run_malicious_training(
        args, attack_model, sub_model,
        train_graphs, test_graphs, test_backdoor, bkd_dr_test, tag2index, device,
        optimizer_D, optimizer_G, generator,
        bkd_gids_train, Ainput_train, Xinput_train, nodenums_id, nodemax, num_classes
    )


    
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
        elif args.attack == "falsification":
            attack_file.write(f"Clean Accuracy (MA): {acc_test_clean:.6f}\n")
            attack_file.write(f"Watermark Accuracy (WA): {acc_test_watermark:.6f}\n")

        elif args.attack == "none":
            attack_file.write(f"Clean Accuracy (MA): {acc_test_clean}\n")
            attack_file.write(f"Watermark Accuracy (WA): {test_watermark}\n")


if __name__ == '__main__':
    main()
