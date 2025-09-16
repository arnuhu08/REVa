# REVa (Robustness Enhancement via Validation)
REVa is a two-part enhancement procedure for deep learning classifiers. First, it evaluates models on adversarial and corruption datasets generated from weak robust samples. Then, it uses the evaluation outcomes to guide targeted improvements in model robustness. To generate the weak robust samples, a per-input resilient analyzer is proposed. It reorders a given dataset from most weak robust to strong robust. These weak robust samples are then employed to create adversarial and common corruption datasets using Torchattacks modules and the corruption types defined in https://github.com/hendrycks/robustness.
# Per-input resilient Analyzer
cd Per-input resilient analyzer
1. first train the preferred model architecture using train.py. E.g. AllConvNet: python train.py -m allconv
2. assign misclassification score to each data instance for a given model architecture by running localRobustnessAnalysis.py. E.g. AllConvNet: python localRobustnessAnalysis.py -r <path/to/modelcheckpoint>

# INfolder
This folder contains the scripts for reproducing the results for REVa, Augmix and the standard method.
# Citations
@inproceedings{hendrycksbenchmarking,
  title={Benchmarking Neural Network Robustness to Common Corruptions and Perturbations},
  author={Hendrycks, Dan and Dietterich, Thomas},
  booktitle={International Conference on Learning Representations}
}
