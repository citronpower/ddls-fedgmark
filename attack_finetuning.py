import torch
import torch.nn.functional as F
import numpy as np
import copy
from torch import optim
from util import global_to_sub, test_ensemble
from check import check_model_difference, check_submodel_copy

def run_finetuning_attack(args, global_model, train_graphs, test_graphs, test_backdoor, bkd_dr_test,
                          num_classes, sub_model, tag2index, device):
    print("Performing improved fine-tuning attack (targeting watermark)...")

    finetuned_model = copy.deepcopy(global_model)
    optimizer_finetune = optim.Adam(finetuned_model.parameters(), lr=args.lr * 0.1)

    #Cleaned graphs
    clean_graphs = []
    for id in range(args.num_agents):
        if id >= args.num_corrupt:
            clean_graphs.extend(train_graphs[id])

    #Inverted watermarked graphs 
    trigger_graphs = [bkd_dr_test[idx] for idx in test_backdoor]
    for g in trigger_graphs:
        g.label = (g.label + 1) % num_classes  # inversion de la cible

    # Combined training set: clean + triggers with wrong label
    full_finetune_data = clean_graphs + trigger_graphs

    for epoch in range(20):
        finetuned_model.train()
        for _ in range(args.iters_per_epoch * 2):
            idx = np.random.permutation(len(full_finetune_data))[:args.batch_size]
            batch_graph = [full_finetune_data[i] for i in idx]

            optimizer_finetune.zero_grad()
            output = finetuned_model(batch_graph)
            labels = torch.LongTensor([g.label for g in batch_graph]).to(device)

            loss = F.cross_entropy(output, labels)
            loss.backward(retain_graph=True)

            optimizer_finetune.step()

        if epoch % 2 == 0:
            print(f"Fine-tuning epoch {epoch}, Loss: {loss.item():.6f}")

    print("Fine-tuning completed.")
    check_model_difference(global_model, finetuned_model)
    global_to_sub(finetuned_model, sub_model)
    check_submodel_copy(finetuned_model, sub_model)

    # evaluation
    acc_test_clean = test_ensemble(args, sub_model, device, test_graphs, tag2index)
    bkd_eval = [bkd_dr_test[idx] for idx in test_backdoor]
    acc_test_watermark = test_ensemble(args, sub_model, device, bkd_eval, tag2index)

    print(f"Fine-tuned model accuracy on clean test data (MA): {acc_test_clean:.6f}")
    print(f"Fine-tuned model accuracy on watermarked data (WA): {acc_test_watermark:.6f}")

    if args.attack != "none" and acc_test_watermark == 1.0:
        print("[WARNING] Watermark Accuracy still at 1.0 — attack may have failed.")

    return acc_test_clean, acc_test_watermark
