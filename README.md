# FedGMark
# FedGMark: Certifiably Robust Watermarking for Federated Graph Learning

This is **NOT** the official implementation of "FedGMark: Certifiably Robust Watermarking for Federated Graph Learning"

This repository is a fork of the official FedGMark used in the context of the Distributed Deep Learning System course of the University of Neuchâtel, Switzerland.
In this work, we want to reproduce the experiments provided by the authors of the original paper and study the case where some clients are malicious.

## Introduction

Federated graph learning (FedGL) is an emerging learning paradigm to collaboratively train graph data from various clients. However, during the development and deployment of FedGL models, they are susceptible to illegal copying and model theft. Backdoor-based watermarking is a well-known method for mitigating these attacks, as it offers ownership verification to the model owner. We take the first step to protect the ownership of FedGL models via backdoor-based watermarking. Existing techniques have challenges in achieving the goal: 1) they either cannot be directly applied or yield unsatisfactory performance; 2) they are vulnerable to watermark removal attacks; and 3) they lack of formal guarantees. To address all the challenges, we propose FedGMark, the first certified robust backdoor-based watermarking for FedGL. FedGMark leverages the unique graph structure and client information in FedGL to learn customized and diverse watermarks. It also designs a novel GL architecture that facilitates defending against both the empirical and theoretically worst-case watermark removal attacks. Extensive experiments validate the promising empirical and provable watermarking performance of FedGMark.

## How to use

# Datasets

Only the MUTAG dataset is provided through the repository because of the size of the others. However, you can use the "data_preprocessing.py" files to download and reformat the datasets to use them for FedGMark.

# Running the pipeline

You should be able to run every experiments using the "main.py" files and the corresponding arguments.
(I will complete this text later...)

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
