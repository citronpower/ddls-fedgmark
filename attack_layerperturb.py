import torch
import copy
from util import global_to_sub, test_ensemble
from check import check_model_difference, check_submodel_copy

def run_layerperturb_attack(args, watermarked_model, clean_model, sub_model, test_graphs, test_backdoor, bkd_dr_test, tag2index, device):
    print("Performing LayerPerturb attack (targeted replacement)...")

    perturbed_model = copy.deepcopy(watermarked_model)
    replaced = 0

    for name in perturbed_model.state_dict():
        if (
            name.startswith("mlps.") or
            name.startswith("linears_prediction.")
        ):
            try:
                perturbed_model.state_dict()[name].copy_(clean_model.state_dict()[name])
                print(f"[INFO] Replacing: {name}")
                replaced += 1
            except Exception as e:
                print(f"[ERROR] Failed to replace {name}: {e}")

    if replaced == 0:
        print("[WARNING] No parameters were replaced. Check naming or model compatibility.")

    # Difference analysis
    check_model_difference(watermarked_model, perturbed_model)

    # Propagation to sub-models
    global_to_sub(perturbed_model, sub_model)
    check_submodel_copy(perturbed_model, sub_model)

    # evaluation
    acc_test_clean = test_ensemble(args, sub_model, device, test_graphs, tag2index)
    bkd_eval = [bkd_dr_test[idx] for idx in test_backdoor]
    acc_test_watermark = test_ensemble(args, sub_model, device, bkd_eval, tag2index)

    print(f"MA (clean test): {acc_test_clean:.6f}")
    print(f"WA (watermarked): {acc_test_watermark:.6f}")
    
    return acc_test_clean, acc_test_watermark
