# FedGMark: Certifiably Robust Watermarking for Federated Graph Learning

This is **NOT** the official implementation of "FedGMark: Certifiably Robust Watermarking for Federated Graph Learning"

This repository is a fork of the official FedGMark used in the context of the Distributed Deep Learning System course of the University of Neuchâtel, Switzerland.
In this work, we want to reproduce the experiments provided by the authors of the original paper and study the case where some clients are malicious.

## Introduction

Federated graph learning (FedGL) is an emerging learning paradigm to collaboratively train graph data from various clients. However, during the development and deployment of FedGL models, they are susceptible to illegal copying and model theft. Backdoor-based watermarking is a well-known method for mitigating these attacks, as it offers ownership verification to the model owner. We take the first step to protect the ownership of FedGL models via backdoor-based watermarking. Existing techniques have challenges in achieving the goal: 1) they either cannot be directly applied or yield unsatisfactory performance; 2) they are vulnerable to watermark removal attacks; and 3) they lack of formal guarantees. To address all the challenges, we propose FedGMark, the first certified robust backdoor-based watermarking for FedGL. FedGMark leverages the unique graph structure and client information in FedGL to learn customized and diverse watermarks. It also designs a novel GL architecture that facilitates defending against both the empirical and theoretically worst-case watermark removal attacks. Extensive experiments validate the promising empirical and provable watermarking performance of FedGMark.

## How to use

# Datasets

Only the MUTAG dataset is provided through the repository because of the size of the others. However, you can use the "data_preprocessing.py" file to download and reformat the datasets to use them for FedGMark. You just have to launch the script as follows:

``` python data_processing.py ```


# Running the pipeline

You should be able to run every experiments using the "main.py" files and the corresponding arguments. 
The most important arguments to reproduce the experiments are:
1. --dataset [MUTAG|DD|PROTEINS|COLLAB]
2. --attack [none|distillation|finetuning|onelayer]

For example,

`` python main.py --dataset PROTEINS --attack finetuning ``

By default, the MUTAG dataset is used if the argument is left empty. Moreover, if you want to run the experiment on one DD or COLLAB, you should ensure you have enough ressources (and time...) available.
By default, no attack is used (i.e. --attack none).

The authors of FedGMark provided many other arguments that you might want to tune according to your goal. In our case, we leave them with the default values. See the main.py file for more information about the other arguments.

# Interprete the results

While the script is running (main.py), you can observe the main accuracy (MA) as well as the watermark accuracy (WA) at each epoc as well as the final accuracies.
If you applied an attack, the main accuracy and the watermark accuracy of the model under attack will also be printed once the model has been trained.

## Cite
```python
@misc{yang2024fedgmarkcertifiablyrobustwatermarking,
      title={FedGMark: Certifiably Robust Watermarking for Federated Graph Learning}, 
      author={Yuxin Yang and Qiang Li and Yuan Hong and Binghui Wang},
      year={2024},
      eprint={2410.17533},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2410.17533}, 
}
```
