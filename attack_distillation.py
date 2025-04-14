import torch
import torch.nn.functional as F
import numpy as np
from torch import optim
from graphcnnt import Discriminatort
from graphcnnt1 import Discriminatort1
from util import global_to_sub

     
def run_distillation_attack(args, global_model, sub_model, train_graphs, test_graphs, test_backdoor, bkd_dr_test, num_classes, device):
    print("Performing distillation attack...")

    # Create a smaller student model
    student_model = Discriminatort(
        args,
        max(2, args.num_layers - 2),
        args.num_mlp_layers,
        train_graphs[0][0].node_features.shape[1],
        args.hidden_dim // 2,
        num_classes,
        args.final_dropout,
        args.learn_eps,
        args.graph_pooling_type,
        args.neighbor_pooling_type,
        device
    ).to(device)

    optimizer_student = optim.Adam(student_model.parameters(), lr=args.lr)
    temp = 4.0  # distillation temperature

    for distill_epoch in range(30):
        global_to_sub(global_model, sub_model)
        for id in range(args.num_agents):
            if id >= args.num_corrupt:
                for _ in range(args.iters_per_epoch):
                    selected_idx = np.random.permutation(len(train_graphs[id]))[:args.batch_size]
                    batch_graph = [train_graphs[id][idx] for idx in selected_idx]

                    with torch.no_grad():
                        for i in range(len(sub_model)):
                            out = sub_model[i](batch_graph, i)
                            if i == 0:
                                teacher_output = out
                            else:
                                teacher_output += out
                        teacher_output = F.softmax(teacher_output / temp, dim=1)

                    student_model.train()
                    optimizer_student.zero_grad()
                    student_output = student_model(batch_graph)
                    student_output_soft = F.log_softmax(student_output / temp, dim=1)

                    loss_distill = F.kl_div(student_output_soft, teacher_output, reduction='batchmean') * (temp * temp)
                    labels = torch.LongTensor([g.label for g in batch_graph]).to(device)
                    loss_ce = F.cross_entropy(student_output, labels)
                    loss = 0.7 * loss_distill + 0.3 * loss_ce

                    loss.backward()
                    optimizer_student.step()

        if distill_epoch % 5 == 0:
            print(f"Distillation epoch {distill_epoch}, Loss: {loss.item()}")

    print("Evaluating distilled model:")

    def test_student_model(model, device, test_graphs):
        model.eval()
        output = []
        for graph in test_graphs:
            out = model([graph])
            output.append(out.detach())
        output = torch.cat(output, 0)
        pred = output.max(1, keepdim=True)[1]
        labels = torch.LongTensor([g.label for g in test_graphs]).to(device)
        correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
        return correct / float(len(test_graphs))

    acc_test_clean = test_student_model(student_model, device, test_graphs)
    acc_test_watermark = test_student_model(student_model, device, [bkd_dr_test[i] for i in test_backdoor])

    print(f"Distilled model accuracy on clean test data (MA): {acc_test_clean:.6f}")
    print(f"Distilled model accuracy on watermarked data (WA): {acc_test_watermark:.6f}")

    return acc_test_clean, acc_test_watermark
