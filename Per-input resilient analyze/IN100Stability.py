import os
import csv
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import time

# ---------- Configuration ----------
imagenet_path = "<path to the IN100 training dataset>"
dataset_split = "train"  # Change to "val" for validation set
run_num = 1
batch_size = 32 # 64
num_workers = min(64, os.cpu_count())
epsilon = 0.1  # Perturbation magnitude
samples_per_input = 20
unstable_threshold = 0.3

mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

# Use all available GPUs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()

# ---------- Transforms ----------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])
def create_model(model_name, num_classes):
    """Creates a torchvision model (CNN or Transformer) and modifies the classification head."""

    model = models.__dict__[model_name]()

    # Try to identify and replace the classification head
    if hasattr(model, 'fc') and isinstance(model.fc, torch.nn.Linear):
        # For models like ResNet
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

    elif hasattr(model, 'head') and isinstance(model.head, torch.nn.Linear):
        # For transformer models like swin_v2_b, vit_b_16, etc.
        model.head = torch.nn.Linear(model.head.in_features, num_classes)

        for param in model.parameters():
            param.requires_grad = False
        for param in model.head.parameters():
            param.requires_grad = True

    else:
        raise ValueError(f"Model '{'swin_v2_b'}' does not have a recognizable classification head ('.fc' or '.head').")

    return model

model_names = ["swin_v2_b"]

# ---------- Model ----------
for model_name in model_names:
    print(f"Loading pretrained {model_name} on multiple GPUs...")
    if model_name == "resnet101_V1":
        model = create_model(model_name, 100)
    elif model_name == "resnet101":
        model = create_model(model_name, 100)
    elif model_name == "resnet50_V1":
        model = create_model(model_name, 100)
    elif model_name == "resnet50":
        model = create_model(model_name, 100)
    elif model_name == "resnet34":
        model = create_model(model_name, 100)
    elif model_name == "resnet18":
        model = create_model(model_name, 100)
    elif model_name == "resnet152_V1":
        model = create_model(model_name, 100)
    elif model_name == "resnet152":
        model = create_model(model_name, 100)
    elif model_name == 'vgg13':
        model = create_model(model_name, 100)
    elif model_name == 'vgg13_bn':
        model = create_model(model_name, 100)
    elif model_name == 'vgg19':
        model = create_model(model_name, 100)
    elif model_name == 'vgg19_bn':
        model = create_model(model_name, 100)
    elif model_name == 'densenet121':
        model = create_model(model_name, 100)
    elif model_name == 'densenet161':
        model = create_model(model_name, 100)
    elif model_name == 'densenet169':
        model = create_model(model_name, 100)
    elif model_name == 'densenet201':
        model = create_model(model_name, 100)
    elif model_name == 'swin_t':
        model = create_model(model_name, 100)
    elif model_name == 'swin_v2_t':
        model = create_model(model_name, 100)
    elif model_name == 'swin_b':
        model = create_model(model_name, 100)
    elif model_name == 'swin_v2_b':
        model = create_model(model_name, 100)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # ---------- Summary/Checkpoint Files ----------
    if dataset_split == "train":
        csv_file = f"{model_name}_robustness_summary_{dataset_split}.csv"
        # Only write header if file does not exist or is empty
        if not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0:
            with open(csv_file, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Step", "Stable", "avg_stable_loss", "Unstable", "avg_unstable_loss", "empirical_risk", "Unstable %"])

        checkpoint_file = f"{model_name}_robustness_checkpoint_{dataset_split}.pkl"
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, "rb") as f:
                checkpoint = pickle.load(f)
                start_idx = checkpoint["start_idx"]
                stable_indices = checkpoint["stable_indices"]
                unstable_indices = checkpoint["unstable_indices"]
                # epoch_risk  = checkpoint["empirical_risk"]
                misclassification_counts = checkpoint["misclassification_counts"]
        else:
            start_idx = 0
            stable_indices, unstable_indices = [], []
            misclassification_counts = []
    else:
        start_idx = 0
        stable_indices, unstable_indices = [], []
        misclassification_counts = []
    
    # ---------- Model Setup ----------
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
        
    model = model.to(device)  # Important: move after DataParallel wrapping
    model.eval()
    cudnn.benchmark = True
    checkpoint =  torch.load('<path to the saved model checkpoints>')
    model.load_state_dict(checkpoint['state_dict'])

    # ---------- Dataset ----------
    print("Loading ImageNet dataset...")
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    imagenet_data = datasets.ImageFolder(imagenet_path, transform=transform)
    imagenet_loader = DataLoader(imagenet_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # ---------- Neighbor Generation ----------
    def generate_neighbors_batch(images, sample_size, epsilon):
        n = images.size(0)
        images = images.unsqueeze(1).expand(n, sample_size, 3, 224, 224)
        images = images.reshape(n * sample_size, 3, 224, 224).to(device)
        noise = (torch.rand_like(images) - 0.5) * 2 * epsilon
        perturbed = (images + noise).clamp(0, 1)
        normalized = (perturbed - mean) / std
        return normalized

    # ---------- Evaluation ----------
    print("Starting robustness evaluation...")
    start_time = time.time()

    use_clean_data_only = False  # Set to True to test only clean images
    epoch_risk = 0.0
    with torch.no_grad():
        for idx, (imgs, labels) in enumerate(imagenet_loader):
            if idx < start_idx:
                continue

            imgs, labels = imgs.to(device), labels.to(device)
            # Step 1: Predict on clean inputs
            normalized_imgs = (imgs - mean) / std
            outputs = model(normalized_imgs)
            preds = outputs.argmax(dim=1)
            correct_mask = preds.eq(labels)  # True for correct predictions
            misclassified_mask = ~correct_mask

            # Compute cross-entropy loss for all samples in the batch
            batch_losses = F.cross_entropy(outputs, labels, reduction='none').detach().cpu().numpy()

            # Step 2: Process misclassified samples — directly label as unstable
            misclassified_indices = torch.arange(len(imgs), device=device)[misclassified_mask].tolist()
            misclassified_losses = batch_losses[misclassified_mask.cpu().numpy()]
            for i in misclassified_indices:
                sample_idx = idx * batch_size + i
                unstable_indices.append(sample_idx)
                misclassification_counts.append(samples_per_input)  # max instability

            # Step 3: Process correctly classified samples for stability
            if correct_mask.sum().item() == 0:
                avg_correct_loss = 0.0
                avg_misclassified_loss = np.mean(misclassified_losses) if len(misclassified_losses) > 0 else 0.0
                # Print or log here if needed
                continue  # No correct predictions to process

            correct_imgs = imgs[correct_mask]
            correct_labels = labels[correct_mask]
            correct_indices = torch.arange(len(imgs), device=device)[correct_mask].tolist()
            correct_losses = batch_losses[correct_mask.cpu().numpy()]

            # Generate neighbors and evaluate
            neighbors = generate_neighbors_batch(correct_imgs, samples_per_input, epsilon)
            neighbor_outputs = model(neighbors)
            neighbor_preds = neighbor_outputs.argmax(dim=1)

            labels_expanded = correct_labels.view(-1, 1).repeat(1, samples_per_input).view(-1)
            neighbor_correct = neighbor_preds.eq(labels_expanded)
            neighbor_correct = neighbor_correct.view(correct_imgs.size(0), samples_per_input)

            misclassified_counts_batch = (neighbor_correct == 0).sum(dim=1).tolist()
            misclassification_counts.extend(misclassified_counts_batch)

            for i in range(correct_imgs.size(0)):
                sample_idx = idx * batch_size + correct_indices[i]
                if misclassified_counts_batch[i] / samples_per_input > unstable_threshold:
                    unstable_indices.append(sample_idx)
                else:
                    stable_indices.append(sample_idx)

            # Compute average loss for correctly predicted and mispredicted samples in this batch
            avg_correct_loss = np.mean(correct_losses) if len(correct_losses) > 0 else 0.0
            avg_misclassified_loss = np.mean(misclassified_losses) if len(misclassified_losses) > 0 else 0.0
            epoch_risk += misclassified_mask.sum().item() / len(misclassified_mask)
            empirical_risk = epoch_risk/ len(imagenet_data)
            # ---------- Save Progress ----------
            total_samples = len(stable_indices) + len(unstable_indices)
            unstable_pct = len(unstable_indices) / total_samples if total_samples > 0 else 0.0
            if idx % 10 == 0:
                print(
                    f"[{idx * batch_size}/{len(imagenet_data)}] "
                    f"Stable: {len(stable_indices)} | "
                    f"Avg Loss: {avg_correct_loss:.3f} | --- | "
                    f"Unstable: {len(unstable_indices)} | "
                    f"Avg Loss: {avg_misclassified_loss:.3f} | "
                    f"empirical_risk: {empirical_risk:.6f} | "
                    f"% : {unstable_pct:.2%}"                
                )

            # Append results to CSV
            if dataset_split == "train":
                with open(csv_file, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        idx * batch_size,
                        len(stable_indices),
                        f"{avg_correct_loss:.3f}",
                        len(unstable_indices),
                        f"{avg_misclassified_loss:.3f}",
                        f"{empirical_risk:.6f}",
                        f"{100*unstable_pct:.2f}"
                    ])

                    # Save checkpoint for recovery
                with open(checkpoint_file, "wb") as f:
                    pickle.dump({
                        "start_idx": idx + 1,
                        "stable_indices": stable_indices,
                        "unstable_indices": unstable_indices,
                        # "empirical_risk": empirical_risk,
                        "misclassification_counts": misclassification_counts
                    }, f)
    print(f"time take for the per-input resilient analysis on {model_name}  is {time.time()-start_time}")
    # ---------- Save Final Results ----------
    if dataset_split == "train":
        print(f"Saving stable and unstable index files for the '{dataset_split}' dataset and model '{model_name}'...")

        with open(f"imagenet_{dataset_split}_{model_name}_stable_indices.pkl", "wb") as f:
            pickle.dump(stable_indices, f)

        with open(f"imagenet_{dataset_split}_{model_name}_unstable_indices.pkl", "wb") as f:
            pickle.dump(unstable_indices, f)

        print("Done.")
    else:
        print("Validation complete. No files saved.")








