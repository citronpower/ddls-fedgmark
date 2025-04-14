import torch

def check_model_difference(model_a, model_b):
    total_diff = 0.0
    for (name_a, param_a), (name_b, param_b) in zip(model_a.named_parameters(), model_b.named_parameters()):
        diff = torch.norm(param_a - param_b).item()
        total_diff += diff
        if diff == 0:
            print(f"[WARNING] No change in param: {name_a}")
    print(f"[DEBUG] Total L2 difference between models: {total_diff:.6f}")


def check_submodel_copy(perturbed_model, sub_model):
    for name_p, param_p in perturbed_model.named_parameters():
        if "linears_prediction.0.weight" in name_p:
            ref = param_p.mean().item()
            break

    models = []

    if isinstance(sub_model, (list, tuple)):
        models = sub_model
    elif isinstance(sub_model, dict):
        models = list(sub_model.values())
    else:
        models = [sub_model]

    for m in models:
        if hasattr(m, "named_parameters"):
            for name_s, param_s in m.named_parameters():
                if "linears_prediction.0.weight" in name_s:
                    print(f"[DEBUG] Copy check — sub_model param mean: {param_s.mean().item():.6f} vs perturbed: {ref:.6f}")
                    return


