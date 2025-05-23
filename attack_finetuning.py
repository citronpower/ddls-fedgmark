# import torch
# import torch.nn.functional as F
# import numpy as np
# import copy
# from torch import optim
# from util import global_to_sub, test_ensemble
# from check import check_model_difference, check_submodel_copy

# def run_finetuning_attack(args, global_model, train_graphs, test_graphs, test_backdoor, bkd_dr_test,
#                           num_classes, sub_model, tag2index, device):
#     print("Performing improved fine-tuning attack (targeting watermark)...")

#     finetuned_model = copy.deepcopy(global_model)
#     optimizer_finetune = optim.Adam(finetuned_model.parameters(), lr=args.lr * 0.1)

#     #Cleaned graphs
#     clean_graphs = []
#     for id in range(args.num_agents):
#         if id >= args.num_corrupt:
#             clean_graphs.extend(train_graphs[id])

#     #Inverted watermarked graphs 
#     trigger_graphs = [bkd_dr_test[idx] for idx in test_backdoor]
#     for g in trigger_graphs:
#         g.label = (g.label + 1) % num_classes  # inversion de la cible

#     # Combined training set: clean + triggers with wrong label
#     full_finetune_data = clean_graphs + trigger_graphs

#     for epoch in range(20):
#         finetuned_model.train()
#         for _ in range(args.iters_per_epoch * 2):
#             idx = np.random.permutation(len(full_finetune_data))[:args.batch_size]
#             batch_graph = [full_finetune_data[i] for i in idx]

#             optimizer_finetune.zero_grad()
#             output = finetuned_model(batch_graph)
#             labels = torch.LongTensor([g.label for g in batch_graph]).to(device)

#             loss = F.cross_entropy(output, labels)
#             loss.backward(retain_graph=True)

#             optimizer_finetune.step()

#         if epoch % 2 == 0:
#             print(f"Fine-tuning epoch {epoch}, Loss: {loss.item():.6f}")

#     print("Fine-tuning completed.")
#     check_model_difference(global_model, finetuned_model)
#     global_to_sub(finetuned_model, sub_model)
#     check_submodel_copy(finetuned_model, sub_model)

#     # evaluation
#     acc_test_clean = test_ensemble(args, sub_model, device, test_graphs, tag2index)
#     bkd_eval = [bkd_dr_test[idx] for idx in test_backdoor]
#     acc_test_watermark = test_ensemble(args, sub_model, device, bkd_eval, tag2index)

#     print(f"Fine-tuned model accuracy on clean test data (MA): {acc_test_clean:.6f}")
#     print(f"Fine-tuned model accuracy on watermarked data (WA): {acc_test_watermark:.6f}")

#     if args.attack != "none" and acc_test_watermark == 1.0:
#         print("[WARNING] Watermark Accuracy still at 1.0 — attack may have failed.")

#     return acc_test_clean, acc_test_watermark


import torch
import torch.nn.functional as F
import numpy as np
import copy
from torch import optim
from util import global_to_sub, test_ensemble, sub_to_global, average_weights
from graphcnnt1 import Discriminatort1

def run_finetuning_attack(args, global_model, train_graphs, test_graphs, test_backdoor, bkd_dr_test,
                          num_classes, sub_model, tag2index, device):
    print("Performing fine-tuning attack (targeting watermark removal)...")

    # Create a copy of the global model for fine-tuning
    finetuned_model = copy.deepcopy(global_model)
    
    # Create corresponding sub-models for fine-tuning (following original architecture)
    finetuned_sub_model = {}
    optimizer_sub = {}
    scheduler_sub = {}
    
    for i in range(args.T):
        finetuned_sub_model[i] = Discriminatort1(
            args, args.num_layers, args.num_mlp_layers, 
            train_graphs[0][0].node_features.shape[1],
            args.hidden_dim, num_classes, args.final_dropout, 
            args.learn_eps, args.graph_pooling_type,
            args.neighbor_pooling_type, device
        ).to(device)
        optimizer_sub[i] = optim.Adam(finetuned_sub_model[i].parameters(), lr=args.lr * 0.1)  # Lower learning rate
        scheduler_sub[i] = optim.lr_scheduler.StepLR(optimizer_sub[i], step_size=10, gamma=0.1)

    # Collect only clean training data (from non-corrupt clients)
    clean_graphs = []
    for id in range(args.num_agents):
        if id >= args.num_corrupt:  # Only use clean clients' data
            clean_graphs.extend(train_graphs[id])
    
    print(f"Using {len(clean_graphs)} clean graphs for fine-tuning")

    # Follow federated learning structure but with clean data only
    for epoch in range(30):  # More epochs for effective fine-tuning
        finetuned_weights = finetuned_model.state_dict()
        local_weights = []
        
        # Simulate multiple clients but all using clean data
        num_finetune_clients = min(args.num_agents - args.num_corrupt, 5)  # Limit to available clean clients
        
        for client_round in range(num_finetune_clients):
            # Load current model weights
            finetuned_model.load_state_dict(copy.deepcopy(finetuned_weights))
            global_to_sub(finetuned_model, finetuned_sub_model)
            
            # Train on clean data only
            for iter_idx in range(args.iters_per_epoch):
                # Sample batch from clean graphs
                if len(clean_graphs) >= args.batch_size:
                    selected_idx = np.random.choice(len(clean_graphs), args.batch_size, replace=False)
                else:
                    selected_idx = np.random.choice(len(clean_graphs), args.batch_size, replace=True)
                
                batch_graph = [clean_graphs[idx] for idx in selected_idx]
                
                # Train each sub-model
                for sub_idx in range(args.T):
                    optimizer_sub[sub_idx].zero_grad()
                    output = finetuned_sub_model[sub_idx](batch_graph, sub_idx)
                    labels = torch.LongTensor([g.label for g in batch_graph]).to(device)
                    
                    loss = F.cross_entropy(output, labels)
                    loss.backward()
                    optimizer_sub[sub_idx].step()
            
            # Update schedulers
            for sch_i in range(args.T):
                scheduler_sub[sch_i].step()
            
            # Aggregate sub-models back to global
            sub_to_global(finetuned_model, finetuned_sub_model)
            l_weights = finetuned_model.state_dict()
            local_weights.append(l_weights)
        
        # Average the local weights
        if local_weights:
            finetuned_weights = average_weights(local_weights)
            finetuned_model.load_state_dict(finetuned_weights)

        if epoch % 5 == 0:
            print(f"Fine-tuning epoch {epoch}, Loss: {loss.item():.6f}")

    print("Fine-tuning completed.")
    
    # Prepare final sub-models for evaluation
    global_to_sub(finetuned_model, finetuned_sub_model)

    # Evaluation using the same method as original training
    acc_test_clean = test_ensemble(args, finetuned_sub_model, device, test_graphs, tag2index)
    
    # Test on watermarked data
    bkd_eval = [bkd_dr_test[idx] for idx in test_backdoor]
    acc_test_watermark = test_ensemble(args, finetuned_sub_model, device, bkd_eval, tag2index)

    print(f"Fine-tuned model accuracy on clean test data (MA): {acc_test_clean:.6f}")
    print(f"Fine-tuned model accuracy on watermarked data (WA): {acc_test_watermark:.6f}")

    # Check if attack was successful (watermark should be removed/weakened)
    if acc_test_watermark < 0.5:
        print("[SUCCESS] Watermark appears to be successfully removed/weakened.")
    else:
        print("[PARTIAL SUCCESS] Watermark partially weakened but still present.")

    return acc_test_clean, acc_test_watermark