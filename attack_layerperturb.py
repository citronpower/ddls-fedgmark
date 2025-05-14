import torch
import copy
from util import global_to_sub, test_ensemble
from check import check_model_difference, check_submodel_copy

def test_single_model(model, device, test_graphs):
    model.eval()
    pred = []
    labels = torch.LongTensor([g.label for g in test_graphs]).to(device)
    with torch.no_grad():
        for g in test_graphs:
            out = model([g])
            pred.append(out.argmax().item())
    pred = torch.tensor(pred).to(device)
    correct = pred.eq(labels).sum().item()
    return correct / len(test_graphs)

def apply_one_layer_perturb_guided(model, layer_name, trigger_graphs, target_labels, epsilon, n_steps=10, lr=1e-2, device='cpu', retain_graph=True):
    model = copy.deepcopy(model).to(device)
    model.train()

    for name, param in model.named_parameters():
        param.requires_grad = (layer_name in name)

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    labels = torch.LongTensor(target_labels).to(device)

    original_state = {name: param.detach().clone() for name, param in model.named_parameters() if layer_name in name}

    for step in range(n_steps):
        optimizer.zero_grad()
        outputs = model(trigger_graphs)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        (-loss).backward(retain_graph=retain_graph)
        print(f"[STEP {step + 1}] Trigger loss: {loss.item():.4f}")

        print("[DEBUG] Gradients sur les poids ciblés :")
        for name, param in model.named_parameters():
            if layer_name in name and param.grad is not None:
                print(f"{name}: grad norm = {param.grad.norm().item():.6f}")

        optimizer.step()

    with torch.no_grad():
        total_diff = 0.0
        for name, param in model.named_parameters():
            if name in original_state:
                delta = param.data - original_state[name]
                norm = torch.norm(delta).item()
                total_diff += norm
                if norm > epsilon:
                    param.data = original_state[name] + delta * (epsilon / norm)
                print(f"[PERTURBED] {name} (norm={norm:.4f})")
            else:
                print(f"[SKIP] {name}")

    print(f"[DEBUG] Perturbation L2 total: {total_diff:.4f}")
    return model

def run_layerperturb_attack(
    args,
    watermarked_model,
    clean_model,
    sub_model,
    test_graphs,
    test_backdoor,
    bkd_dr_test,
    tag2index,
    device,
    target_layers=["mlps.15"],
    epsilon=0.5,
    replace_with_clean=False,
    n_steps=10,
    lr=1e-2
):
    print(f"Performing Guided Perturb attack on layers {', '.join(target_layers)} with ε={epsilon}...")

    trigger_graphs = [bkd_dr_test[idx] for idx in test_backdoor]
    target_labels = [g.label for g in trigger_graphs]

    perturbed_model = copy.deepcopy(watermarked_model)

    for i, target_layer in enumerate(target_layers):
        retain = True if i < len(target_layers) - 1 else False
        perturbed_model = apply_one_layer_perturb_guided(
            model=perturbed_model,
            layer_name=target_layer,
            trigger_graphs=trigger_graphs,
            target_labels=target_labels,
            epsilon=epsilon,
            n_steps=n_steps,
            lr=lr,
            device=device,
            retain_graph=retain
        )

    if replace_with_clean:
        print("[INFO] Replacing some perturbed params with clean ones...")
        for target_layer in target_layers:
            for suffix in [".weight", ".bias"]:
                name = target_layer + suffix
                if name in perturbed_model.state_dict() and name in clean_model.state_dict():
                    perturbed_model.state_dict()[name].copy_(clean_model.state_dict()[name])
                    print(f"[INFO] Replacing: {name}")
                else:
                    print(f"[WARNING] {name} not found in state dict.")

    check_model_difference(watermarked_model, perturbed_model)
    acc_direct = test_single_model(perturbed_model, device, trigger_graphs)
    print(f"[DEBUG] WA direct from perturbed_model: {acc_direct:.4f}")

    global_to_sub(perturbed_model, sub_model)
    check_submodel_copy(perturbed_model, sub_model)

    acc_test_clean = test_ensemble(args, sub_model, device, test_graphs, tag2index)
    acc_test_watermark = test_ensemble(args, sub_model, device, trigger_graphs, tag2index)

    print(f"MA (clean test): {acc_test_clean:.6f}")
    print(f"WA (watermarked test): {acc_test_watermark:.6f}")

    return acc_test_clean, acc_test_watermark
