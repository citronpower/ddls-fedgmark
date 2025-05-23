# import torch
# import torch.nn.functional as F
# import numpy as np
# from torch import optim
# from graphcnnt import Discriminatort
# from graphcnnt1 import Discriminatort1
# from util import global_to_sub

     
# def run_distillation_attack(args, global_model, sub_model, train_graphs, test_graphs, test_backdoor, bkd_dr_test, num_classes, device):
#     print("Performing distillation attack...")

#     # Create a smaller student model
#     student_model = Discriminatort(
#         args,
#         max(2, args.num_layers - 2),
#         args.num_mlp_layers,
#         train_graphs[0][0].node_features.shape[1],
#         args.hidden_dim // 2,
#         num_classes,
#         args.final_dropout,
#         args.learn_eps,
#         args.graph_pooling_type,
#         args.neighbor_pooling_type,
#         device
#     ).to(device)

#     optimizer_student = optim.Adam(student_model.parameters(), lr=args.lr)
#     temp = 4.0  # distillation temperature

#     for distill_epoch in range(30):
#         global_to_sub(global_model, sub_model)
#         for id in range(args.num_agents):
#             # if id >= args.num_corrupt:
#             for _ in range(args.iters_per_epoch):
#                 selected_idx = np.random.permutation(len(train_graphs[id]))[:args.batch_size]
#                 batch_graph = [train_graphs[id][idx] for idx in selected_idx]

#                 with torch.no_grad():
#                     for i in range(len(sub_model)):
#                         out = sub_model[i](batch_graph, i)
#                         if i == 0:
#                             teacher_output = out
#                         else:
#                             teacher_output += out
#                     teacher_output = F.softmax(teacher_output / temp, dim=1)

#                 student_model.train()
#                 optimizer_student.zero_grad()
#                 student_output = student_model(batch_graph)
#                 student_output_soft = F.log_softmax(student_output / temp, dim=1)

#                 loss_distill = F.kl_div(student_output_soft, teacher_output, reduction='batchmean') * (temp * temp)
#                 labels = torch.LongTensor([g.label for g in batch_graph]).to(device)
#                 loss_ce = F.cross_entropy(student_output, labels)
#                 loss = 0.7 * loss_distill + 0.3 * loss_ce

#                 loss.backward()
#                 optimizer_student.step()

#         if distill_epoch % 5 == 0:
#             print(f"Distillation epoch {distill_epoch}, Loss: {loss.item()}")

#     print("Evaluating distilled model:")

#     def test_student_model(model, device, test_graphs):
#         model.eval()
#         output = []
#         for graph in test_graphs:
#             out = model([graph])
#             output.append(out.detach())
#         output = torch.cat(output, 0)
#         pred = output.max(1, keepdim=True)[1]
#         labels = torch.LongTensor([g.label for g in test_graphs]).to(device)
#         correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
#         return correct / float(len(test_graphs))

#     acc_test_clean = test_student_model(student_model, device, test_graphs)
#     acc_test_watermark = test_student_model(student_model, device, [bkd_dr_test[i] for i in test_backdoor])

#     print(f"Distilled model accuracy on clean test data (MA): {acc_test_clean:.6f}")
#     print(f"Distilled model accuracy on watermarked data (WA): {acc_test_watermark:.6f}")

#     return acc_test_clean, acc_test_watermark

import torch
import torch.nn.functional as F
import numpy as np
from torch import optim
from graphcnnt import Discriminatort
from graphcnnt1 import Discriminatort1
from util import global_to_sub, test_ensemble, sub_to_global, average_weights
import copy

     
def run_distillation_attack(args, global_model, sub_model, train_graphs, test_graphs, test_backdoor, bkd_dr_test, num_classes, device, tag2index):
    print("Performing distillation attack...")

    # Create a smaller student model - using the same architecture as the teacher for consistency
    student_model = Discriminatort(
        args,
        args.num_layers,  # Keep same layers to maintain compatibility
        args.num_mlp_layers,
        train_graphs[0][0].node_features.shape[1],
        args.hidden_dim // 2,  # Smaller hidden dimension for student
        num_classes,
        args.final_dropout,
        args.learn_eps,
        args.graph_pooling_type,
        args.neighbor_pooling_type,
        device
    ).to(device)

    # Create corresponding sub-models for the student (following the original architecture)
    student_sub_model = {}
    student_optimizer_sub = {}
    for i in range(args.T):
        student_sub_model[i] = Discriminatort1(
            args, args.num_layers, args.num_mlp_layers, 
            train_graphs[0][0].node_features.shape[1],
            args.hidden_dim // 2,  # Smaller for student
            num_classes, args.final_dropout, args.learn_eps, 
            args.graph_pooling_type, args.neighbor_pooling_type, device
        ).to(device)
        student_optimizer_sub[i] = optim.Adam(student_sub_model[i].parameters(), lr=args.lr)

    temp = 4.0  # distillation temperature

    # Follow the same federated learning structure as in main training
    for distill_epoch in range(50):  # Increased epochs for better distillation
        student_weights = student_model.state_dict()
        local_weights = []
        
        # Select random clients like in original training
        m = max(int(args.frac_epoch * args.num_agents), 1)
        idxs_users = np.random.choice(range(args.num_agents), m, replace=False)
        
        for id in idxs_users:
            # Load current student weights
            student_model.load_state_dict(copy.deepcopy(student_weights))
            global_to_sub(student_model, student_sub_model)
            
            # Only use clean data for distillation (non-watermarked clients)
            if id >= args.num_corrupt:
                # Get teacher predictions with ensemble
                global_to_sub(global_model, sub_model)
                
                for iter_idx in range(args.iters_per_epoch):
                    # Sample batch correctly
                    if len(train_graphs[id]) >= args.batch_size:
                        selected_idx = np.random.choice(len(train_graphs[id]), args.batch_size, replace=False)
                    else:
                        selected_idx = np.random.choice(len(train_graphs[id]), args.batch_size, replace=True)
                    
                    batch_graph = [train_graphs[id][idx] for idx in selected_idx]

                    # Get teacher ensemble predictions (soft targets)
                    with torch.no_grad():
                        teacher_outputs = []
                        for i in range(args.T):
                            out = sub_model[i](batch_graph, i)
                            teacher_outputs.append(out)
                        
                        # Average teacher outputs (ensemble)
                        teacher_output = torch.stack(teacher_outputs).mean(dim=0)
                        teacher_soft = F.softmax(teacher_output / temp, dim=1)

                    # Train student sub-models
                    for sub_idx in range(args.T):
                        student_optimizer_sub[sub_idx].zero_grad()
                        student_output = student_sub_model[sub_idx](batch_graph, sub_idx)
                        student_soft = F.log_softmax(student_output / temp, dim=1)

                        # Distillation loss
                        loss_distill = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (temp * temp)
                        
                        # Hard target loss
                        labels = torch.LongTensor([g.label for g in batch_graph]).to(device)
                        loss_ce = F.cross_entropy(student_output, labels)
                        
                        # Combined loss
                        loss = 0.7 * loss_distill + 0.3 * loss_ce
                        loss.backward()
                        student_optimizer_sub[sub_idx].step()

                # Aggregate student sub-models back to student global model
                sub_to_global(student_model, student_sub_model)
                l_weights = student_model.state_dict()
                local_weights.append(l_weights)
        
        # Update student global model with averaged weights
        if local_weights:
            student_weights = average_weights(local_weights)
            student_model.load_state_dict(student_weights)

        if distill_epoch % 10 == 0:
            print(f"Distillation epoch {distill_epoch}")

    print("Evaluating distilled model:")

    # Use the same test ensemble method as in original code
    global_to_sub(student_model, student_sub_model)
    
    acc_test_clean = test_ensemble(args, student_sub_model, device, test_graphs, tag2index)
    
    # Test on watermarked data
    bkd_dr_list = [bkd_dr_test[idx] for idx in test_backdoor]
    acc_test_watermark = test_ensemble(args, student_sub_model, device, bkd_dr_list, tag2index)

    print(f"Distilled model accuracy on clean test data (MA): {acc_test_clean:.6f}")
    print(f"Distilled model accuracy on watermarked data (WA): {acc_test_watermark:.6f}")

    return acc_test_clean, acc_test_watermark