# [REVa (Robustness Enhancement via Validation)](https://arxiv.org/abs/2509.19197)

![Journal Framework](https://github.com/arnuhu08/REVa/blob/main/REVaFramework.png)

## Introduction

This repository is the implementtion of REVa, [A Validation Strategy for Deep Learning Models: Evaluating and Enhancing Robustness](https://doi.ieeecomputersociety.org/10.1109/OJCS.2025.3650722). REVa is a two-part enhancement procedure for deep learning classifiers. First, it evaluates models on adversarial and corruption datasets generated from weak robust samples. Then, it uses the evaluation outcomes to guide targeted improvements in model robustness. To generate the weak robust samples, a per-input resilient analyzer is proposed. It reorders a given dataset from most weak robust to strong robust. These weak robust samples are then employed to create adversarial and common corruption datasets using Torchattacks modules and the corruption types defined in [here](https://github.com/hendrycks/robustness).

## Per-Input Resilient Analyzer

## Setup

### Requirements

- numpy >= 1.15.0  
- Pillow >= 6.1.0  
- torch == 2.1.0  
- torchvision >= 0.16.0  
- tensorflow >= 2.0

1. Clone this repository and install PyTorch and the required Python libraries:  

    ```bash
    conda env create -f REVaEnv.yml
    ```

2. Download CIFAR-10-C, CIFAR-100-C, and ImageNet-C datasets from [here](https://github.com/hendrycks/robustness).

    ```bash
    cd Per-input-resilient-analyzer
    ```

- For CIFAR dataset:  
  1. Train the preferred model architecture using `train.py`.  
     Example (AllConvNet):  

     ```bash
     python train.py -m allconv
     ```

  2. Assign a misclassification score to each data instance for the chosen model using `localRobustnessAnalysis.py`.  
     Example (AllConvNet):  

     ```bash
     python localRobustnessAnalysis.py -r <path/to/model/checkpoint>
     ```

- For IN100 dataset:  
  1. Get the required pretrained models from PyTorch (e.g., ResNet-18 or Swin_V2_B).  
  2. Run stability analysis:  

     ```bash
     python IN100Stability.py
     ```

## CIFAR Folder

```bash
  cd CIFAR  
```

1. Contains scripts for reproducing results of REVa-enhanced models on CIFAR datasets.  
2. To generate adversarial datasets for training REVa-enhanced models, run the notebook `AdversarialDatasetGeneration.ipynb` in the `adversarial-attacks-pytorch` folder.  

## IN100 Folder

```bash
  cd IN100
```

1. Contains scripts for reproducing results of REVa, AugMix, and the standard method on ImageNet100.  
2. To generate adversarial datasets for training REVa-enhanced models, run the notebook `AdversarialDatasetGeneration.ipynb` in the `adversarial-attacks-pytorch` folder.  

## Usage

### Training Configurations

- **CIFAR Datasets**  
  Use the same training recipes as [AugMix](https://arxiv.org/abs/1912.02781) for fair comparison.

- **ImageNet100**  
  - **ResNet-18**  

    ```bash
    python IN100.py -m resnet18 <path/to/imagenet100> <path/to/imagenet100-c> --pretrained
    ```  

  - **Swin_V2_B**  

    ```bash
    python IN100.py -m swin_v2_b <path/to/imagenet100> <path/to/imagenet100-c> --pretrained
    ```

### Misc

- **Generating ImageNet-100 and ImageNet-100-C**

  - Repository: [ImageNet-100-Pytorch](https://github.com/danielchyeh/ImageNet-100-Pytorch)  

  - Steps:
    1. Download the **ImageNet-1K** and **ImageNet-C** datasets from their official sources.  
    2. Follow the procedure in the linked repository to generate **ImageNet-100** and **ImageNet-100-C**.  

## Reference

If you find this framework useful, please cite our work:

```bibtex
@article{nuhu2026validation,
      title={A Validation Strategy for Deep Learning Models: Evaluating and Enhancing Robustness}, 
      author={Abdul-Rauf Nuhu and Parham Kebria and Vahid Hemmati and Benjamin Lartey and
      Mahmoud Nabil Mahmoud and Abdollah Homaifar and Edward Tunstel},
      journal={IEEE Open Journal of Computer Society},
      publisher={IEEE Computer Society},
      year={2026},
      volume={7},
      pages={276-289},
      url={https://doi.ieeecomputersociety.org/10.1109/OJCS.2025.3650722}, 
}

@misc{nuhu2025validation,
      title={A Validation Strategy for Deep Learning Models: Evaluating and Enhancing Robustness}, 
      author={Abdul-Rauf Nuhu and Parham Kebria and Vahid Hemmati and Benjamin Lartey and
      Mahmoud Nabil Mahmoud and Abdollah Homaifar and Edward Tunstel},
      year={2025},
      eprint={2509.19197},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2509.19197}, 
}
```
