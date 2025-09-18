# REVa (Robustness Enhancement via Validation)
![Journal Framework](https://github.com/arnuhu08/REVa/blob/main/pictures%20/JournalFramework.png)
# Introduction
We introduce REVa, a two-part enhancement procedure for deep learning classifiers. First, it evaluates models on adversarial and corruption datasets generated from weak robust samples. Then, it uses the evaluation outcomes to guide targeted improvements in model robustness. To generate the weak robust samples, a per-input resilient analyzer is proposed. It reorders a given dataset from most weak robust to strong robust. These weak robust samples are then employed to create adversarial and common corruption datasets using Torchattacks modules and the corruption types defined in https://github.com/hendrycks/robustness.


# Per-Input Resilient Analyzer

cd Per-input-resilient-analyzer

- For CIFAR dataset:  
  1. Train the preferred model architecture using `train.py`.  
     Example (AllConvNet):  
     ```
     python train.py -m allconv
     ```
  2. Assign a misclassification score to each data instance for the chosen model using `localRobustnessAnalysis.py`.  
     Example (AllConvNet):  
     ```
     python localRobustnessAnalysis.py -r <path/to/model/checkpoint>
     ```

- For IN100 dataset:  
  1. Evaluate the required pretrained models from PyTorch (e.g., ResNet-18 or Swin_V2_B).  
  2. Run stability analysis:  
     ```
     python IN100Stability.py
     ```

# CIFAR Folder
cd CIFAR  
1. Contains scripts for reproducing results of REVa-enhanced models on CIFAR datasets.  
2. To generate adversarial datasets for training REVa-enhanced models, run the notebook `AdversarialDatasetGeneration.ipynb` in the `adversarial-attacks-pytorch` folder.  

# IN100 Folder
cd IN100  
1. Contains scripts for reproducing results of REVa, AugMix, and the standard method on ImageNet100.  
2. To generate adversarial datasets for training REVa-enhanced models, run the notebook `AdversarialDatasetGeneration.ipynb` in the `adversarial-attacks-pytorch` folder.  

# Requirements
- numpy >= 1.15.0  
- Pillow >= 6.1.0  
- torch == 2.1.0  
- torchvision >= 0.16.0  

# Setup
1. Install PyTorch and the required Python libraries:  
   ```bash
   conda env create -f REVaEnv.yml
2. Down CIFAR-10-C, CIFAR-100-C and ImageNet-C datasets from the following link:
   https://github.com/hendrycks/robustness

# Usage

## Training Configurations

- **CIFAR Datasets**  
  Use the same training recipes as the [AugMix paper](https://arxiv.org/abs/1912.02781) for fair comparison.

- **ImageNet100**  
  - **ResNet-18**  
    ```bash
    python IN100.py -m resnet18 <path/to/imagenet100> <path/to/imagenet100-c> --pretrained
    ```  

  - **Swin_V2_B**  
    ```bash
    python IN100.py -m swin_v2_b <path/to/imagenet100> <path/to/imagenet100-c> --pretrained
    ```  


