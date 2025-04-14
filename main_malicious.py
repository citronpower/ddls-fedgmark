import torch
import numpy as np
import random
import copy
from util import *
from attack_layerperturb import run_layerperturb_attack
from check import check_model_difference, check_submodel_copy

def run_malicious_training(
    args, global_model, sub_model,
    train_graphs, test_graphs, test_backdoor, bkd_dr_test, tag2index, device,
    optimizer_D, optimizer_G, generator,
    bkd_gids_train, Ainput_train, Xinput_train, nodenums_id, nodemax,
    num_classes
):
    print("\n===== Training with malicious clients (random watermarking / free-riders) =====")

    n_clients = args.num_agents
    n_malicious = int(n_clients * args.malicious_frac)
    malicious_clients = random.sample(range(n_clients), n_malicious)
    print(f"[INFO] Malicious clients: {malicious_clients}")

    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        local_weights = []
        idxs_users = np.random.choice(range(n_clients), n_clients, replace=False)

        for id in idxs_users:
            id = int(id)  # garantir un type de clé compatible
            global_to_sub(global_model, sub_model)

            if id in malicious_clients:
                # --- FREE-RIDER MALICIEUX ---
                # Génère un faux watermark aléatoire non coordonné
                graphs_copy = copy.deepcopy(train_graphs[id])

                available_ids = list(range(len(graphs_copy)))
                chosen_ids = np.random.choice(available_ids, size=min(2, len(available_ids)), replace=False)

                for idx in chosen_ids:
                    # Génère une fausse matrice d'adjacence (aléatoire mais symétrique)
                    rand_adj = np.random.rand(*graphs_copy[idx].edge_mat.shape)
                    rand_adj = ((rand_adj + rand_adj.T) / 2 > 0.8).astype(float)
                    graphs_copy[idx].edge_mat = torch.tensor(rand_adj)

                    # Nouvelles features aléatoires
                    graphs_copy[idx].node_features = torch.rand_like(graphs_copy[idx].node_features)

                    # Étiquette aléatoire (≠ target)
                    graphs_copy[idx].label = random.randint(0, num_classes - 1)

                loss = train_sub(args, global_model, sub_model, optimizer=[optimizer_D]*len(sub_model),
                                 device=device, train_graphs=graphs_copy, epoch=epoch, tag2index=tag2index)

            elif id < args.num_corrupt:
                # Clients qui injectent un vrai watermark
                for _ in range(args.n_train_D):
                    train_D_sub(args, global_model, sub_model, generator[id], optimizer=[optimizer_G[id]] * len(sub_model),
                                id=id, device=device, train_graphs=train_graphs[id], epoch=epoch, tag2index=tag2index,
                                bkd_gids_train=bkd_gids_train[id], Ainput_train=Ainput_train[id],
                                Xinput_train=Xinput_train[id], nodenums_id=nodenums_id[id], nodemax=nodemax)

                for _ in range(args.n_train_G):
                    train_G(args, global_model, sub_model, generator[id], optimizer_G[id], id, device,
                            train_graphs[id], epoch, tag2index, bkd_gids_train[id], Ainput_train[id],
                            Xinput_train[id], nodenums_id[id], nodemax)

            else:
                # Clients honnêtes (pas de watermark)
                loss = train_sub(args, global_model, sub_model, optimizer=[optimizer_D]*len(sub_model),
                                 device=device, train_graphs=train_graphs[id], epoch=epoch, tag2index=tag2index)

            sub_to_global(global_model, sub_model)
            local_weights.append(copy.deepcopy(global_model.state_dict()))

        # Moyenne pondérée des poids
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)
        global_to_sub(global_model, sub_model)

        # Évaluation
        acc_test_clean = test_ensemble(args, sub_model, device, test_graphs, tag2index)
        bkd_eval = [bkd_dr_test[idx] for idx in test_backdoor]
        acc_test_watermark = test_ensemble(args, sub_model, device, bkd_eval, tag2index)

        print(f"[EPOCH {epoch+1}] MA (clean): {acc_test_clean:.4f} | WA (watermark): {acc_test_watermark:.4f}")

        if acc_test_watermark < 1.0:
            print("[INFO] Watermark weakened by random triggers.")

    return acc_test_clean, acc_test_watermark
