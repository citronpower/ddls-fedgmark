import torch
import copy

def test_single_model(model, device, test_graphs):
    model.eval()
    labels = torch.LongTensor([g.label for g in test_graphs]).to(device)
    pred = []
    with torch.no_grad():
        for g in test_graphs:
            out = model([g])
            pred.append(out.argmax().item())
    pred = torch.tensor(pred).to(device)
    correct = pred.eq(labels).sum().item()
    return correct / len(test_graphs)

def test_watermark_accuracy(model, device, watermark_graphs):
    model.eval()
    labels = torch.LongTensor([g.label for g in watermark_graphs]).to(device)
    pred = []
    with torch.no_grad():
        for g in watermark_graphs:
            out = model([g])
            pred.append(out.argmax().item())
    pred = torch.tensor(pred).to(device)
    correct = pred.eq(labels).sum().item()
    return correct / len(watermark_graphs)

def run_layerperturb_attack(model_w, model_clean, test_graphs, watermark_graphs, device, layers_to_replace):
    """
    Remplace toutes les couches listées dans layers_to_replace (via .startswith),
    puis teste le modèle sur les graphes propres et watermarkés.
    """
    perturbed_model = copy.deepcopy(model_w).to(device)
    clean_state = dict(model_clean.named_parameters())
    perturbed_state = dict(perturbed_model.named_parameters())

    for name in clean_state:
        if any(name.startswith(layer) for layer in layers_to_replace):
            if name in perturbed_state:
                with torch.no_grad():
                    perturbed_state[name].copy_(clean_state[name])
                    print(f"[LayerPerturb] Replaced layer: {name}")

    ma = test_single_model(perturbed_model, device, test_graphs)
    wa = test_watermark_accuracy(perturbed_model, device, watermark_graphs)
    return perturbed_model, ma, wa
