# REVa (Robustness Enhancement via Validation)
REVa is a two-part enhancement procedure for deep learning classifiers. First, it evaluates models on adversarial and corruption datasets generated from weak robust samples. Then, it uses the evaluation outcomes to guide targeted improvements in model robustness. To generate the weak robust samples, a per-input resilient analyzer is proposed. It reorders a given dataset from most weak robust to strong robust. These weak robust samples are then employed to create adversarial and common corruption datasets using Torchattacks modules and the corruption types defined in https://github.com/hendrycks/robustness.
https://github.com/arnuhu08/REVa/blob/main/pictures%20/JournalFramework.png
![Journal Framework](https://github.com/arnuhu08/REVa/blob/main/pictures%20/JournalFramework.png)

# Per-input resilient Analyzer
cd Per-input resilient analyzer
- For CIFAR dataset:
1. first train the preferred model architecture using train.py. E.g. AllConvNet: python train.py -m allconv
2. assign misclassification score to each data instance for a given model architecture by running localRobustnessAnalysis.py. E.g. AllConvNet: python localRobustnessAnalysis.py -r <path/to/modelcheckpoint>
- For IN100 dataset:
# CIFARFolder
cd CIFAR 
1. This folder contains the scripts for reproducing the results for REVa enhanced models on CIFAR datasets.
2. To generate the adversarial dataset for training REVa enhanced models run the notebook AdversarialDatasetGeneration.ipynb in the adversarial-attacks-pytorch folder
# INfolder
cd IN100 
1. This folder contains the scripts for reproducing the results for REVa, Augmix and the standard method.
2. To generate the adversarial dataset for training REVa enhanced models run the notebook AdversarialDatasetGeneration.ipynb in the adversarial-attacks-pytorch folder 
# Requirements
- numpy>=1.15.0
- Pillow>=6.1.0
- torch==2.1.0
- torchvision>=0.16.0
# Setup
1. Install PyTorch and other required python libraries with:
   ```markdown
   Run `conda env create -f REVaEnv.yml`
2. Down CIFAR-10-C, CIFAR-100-C and ImageNet-C datasets from the following link:
   https://github.com/hendrycks/robustness

# Usage
Training Configurations used in our paper:
- For the CIFAR DATASETs same use training recipes used in the default augmix paper for fair comparisons.
- For ImageNet100
- ResNet-18: python IN100.py -m resnet18 <path/to/imagenet100> <path/to/imagenet100-c> --pretrained
- Swin_V2_B: python IN100.py -m swin_v2_b <path/to/imagenet100> <path/to/imagenet100-c> --pretrained
# Citations
@inproceedings{hendrycksbenchmarking,
  title={Benchmarking Neural Network Robustness to Common Corruptions and Perturbations},
  author={Hendrycks, Dan and Dietterich, Thomas},
  booktitle={International Conference on Learning Representations}
}
