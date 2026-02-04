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
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------- Configuration ----------
imagenet_path = os.getenv('IMAGENET_PATH', './data/imagenet')  # Use environment variable or default
checkpoint_path = os.getenv('CHECKPOINT_PATH', './checkpoints/model.pth')  # Use environment variable or default
dataset_split = "train"  # Change to "val" for validation set
run_num = 1
batch_size = 32 # 64
num_workers = min(64, os.cpu_count())
epsilon = 0.1  # Perturbation magnitude
samples_per_input = 20
unstable_threshold = 0.3

# Validate configuration
if epsilon <= 0 or epsilon > 1:
    raise ValueError(f"Epsilon must be between 0 and 1, got {epsilon}")
if batch_size <= 0:
    raise ValueError(f"Batch size must be positive, got {batch_size}")
if samples_per_input <= 0:
    raise ValueError(f"Samples per input must be positive, got {samples_per_input}")
if not (0 <= unstable_threshold <= 1):
    raise ValueError(f"Unstable threshold must be between 0 and 1, got {unstable_threshold}")

# Check CUDA availability and set device
if torch.cuda.is_available():
    device = torch.device("cuda")
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
    logger.info(f"CUDA available. Using {torch.cuda.device_count()} GPU(s)")
else:
    device = torch.device("cpu")
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    logger.warning("CUDA not available. Using CPU - this will be significantly slower")

# Validate dataset path
if not os.path.exists(imagenet_path):
    logger.error(f"ImageNet dataset path does not exist: {imagenet_path}")
    logger.info("Please set the IMAGENET_PATH environment variable to the correct path")
    sys.exit(1)

# Validate checkpoint path
if not os.path.exists(checkpoint_path):
    logger.error(f"Checkpoint file does not exist: {checkpoint_path}")
    logger.info("Please set the CHECKPOINT_PATH environment variable to the correct path")
    sys.exit(1)

num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

# ---------- Transforms ----------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])
def create_model(model_name, num_classes):
    """Creates a torchvision model (CNN or Transformer) and modifies the classification head."""
    try:
        if not hasattr(models, model_name):
            raise ValueError(f"Model '{model_name}' is not available in torchvision.models")
        
        model = models.__dict__[model_name]()
        logger.info(f"Successfully created {model_name} model")

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
            raise ValueError(f"Model '{model_name}' does not have a recognizable classification head ('.fc' or '.head').")

        return model
    except Exception as e:
        logger.error(f"Failed to create model '{model_name}': {str(e)}")
        raise

# Supported models - simplified configuration
supported_models = {
    "resnet101_V1", "resnet101", "resnet50_V1", "resnet50", "resnet34", "resnet18",
    "resnet152_V1", "resnet152", "vgg13", "vgg13_bn", "vgg19", "vgg19_bn",
    "densenet121", "densenet161", "densenet169", "densenet201",
    "swin_t", "swin_v2_t", "swin_b", "swin_v2_b"
}

model_names = ["swin_v2_b"]

# ---------- Model ----------
for model_name in model_names:
    logger.info(f"Processing model: {model_name}")
    
    if model_name not in supported_models:
        logger.error(f"Unsupported model: {model_name}")
        logger.info(f"Supported models: {sorted(supported_models)}")
        continue
    
    try:
        logger.info(f"Loading pretrained {model_name}...")
        model = create_model(model_name, 100)
    except Exception as e:
        logger.error(f"Failed to create model {model_name}: {str(e)}")
        continue

    # ---------- Summary/Checkpoint Files ----------
    if dataset_split == "train":
        csv_file = f"{model_name}_robustness_summary_{dataset_split}.csv"
        checkpoint_file = f"{model_name}_robustness_checkpoint_{dataset_split}.pkl"
        
        # Initialize CSV file with error handling
        try:
            if not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0:
                with open(csv_file, mode="w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Step", "Stable", "avg_stable_loss", "Unstable", "avg_unstable_loss", "empirical_risk", "Unstable %"])
                logger.info(f"Created CSV file: {csv_file}")
        except (OSError, IOError) as e:
            logger.error(f"Failed to create/write CSV file {csv_file}: {str(e)}")
            raise

        # Load checkpoint with error handling
        try:
            if os.path.exists(checkpoint_file):
                with open(checkpoint_file, "rb") as f:
                    checkpoint_data = pickle.load(f)
                    
                # Validate checkpoint data
                required_keys = ["start_idx", "stable_indices", "unstable_indices", "misclassification_counts"]
                if all(key in checkpoint_data for key in required_keys):
                    start_idx = checkpoint_data["start_idx"]
                    stable_indices = checkpoint_data["stable_indices"]
                    unstable_indices = checkpoint_data["unstable_indices"]
                    misclassification_counts = checkpoint_data["misclassification_counts"]
                    logger.info(f"Loaded checkpoint from {checkpoint_file}, starting from index {start_idx}")
                else:
                    logger.warning(f"Invalid checkpoint file {checkpoint_file}, starting fresh")
                    start_idx = 0
                    stable_indices, unstable_indices = [], []
                    misclassification_counts = []
            else:
                start_idx = 0
                stable_indices, unstable_indices = [], []
                misclassification_counts = []
                logger.info("No checkpoint found, starting fresh")
        except (pickle.PickleError, OSError, IOError) as e:
            logger.error(f"Failed to load checkpoint {checkpoint_file}: {str(e)}")
            logger.info("Starting fresh")
            start_idx = 0
            stable_indices, unstable_indices = [], []
            misclassification_counts = []
    else:
        start_idx = 0
        stable_indices, unstable_indices = [], []
        misclassification_counts = []
    
    # ---------- Model Setup ----------
    try:
        if num_gpus > 1:
            logger.info(f"Using {num_gpus} GPUs with DataParallel")
            model = nn.DataParallel(model)
        elif num_gpus == 1:
            logger.info("Using single GPU")
        else:
            logger.info("Using CPU")
            
        model = model.to(device)  # Important: move after DataParallel wrapping
        model.eval()
        
        if torch.cuda.is_available():
            cudnn.benchmark = True
        
        # Load checkpoint with proper error handling
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        try:
            checkpoint_data = torch.load(checkpoint_path, map_location=device)
            if 'state_dict' in checkpoint_data:
                model.load_state_dict(checkpoint_data['state_dict'])
                logger.info("Successfully loaded model weights")
            else:
                # Try loading directly if no 'state_dict' key
                model.load_state_dict(checkpoint_data)
                logger.info("Successfully loaded model weights (direct format)")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {str(e)}")
            raise
            
    except Exception as e:
        logger.error(f"Failed to setup model: {str(e)}")
        raise

    # ---------- Dataset ----------
    try:
        logger.info(f"Loading ImageNet dataset from {imagenet_path}...")
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        imagenet_data = datasets.ImageFolder(imagenet_path, transform=transform)
        
        if len(imagenet_data) == 0:
            raise ValueError(f"Dataset is empty: {imagenet_path}")
            
        logger.info(f"Dataset loaded successfully. Total samples: {len(imagenet_data)}")
        imagenet_loader = DataLoader(imagenet_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        raise

    # ---------- Neighbor Generation ----------
    def generate_neighbors_batch(images, sample_size, epsilon):
        try:
            n = images.size(0)
            if n == 0:
                raise ValueError("Empty batch provided to generate_neighbors_batch")
                
            images = images.unsqueeze(1).expand(n, sample_size, 3, 224, 224)
            images = images.reshape(n * sample_size, 3, 224, 224).to(device)
            noise = (torch.rand_like(images) - 0.5) * 2 * epsilon
            perturbed = (images + noise).clamp(0, 1)
            normalized = (perturbed - mean.to(device)) / std.to(device)
            return normalized
        except Exception as e:
            logger.error(f"Error in generate_neighbors_batch: {str(e)}")
            raise

    # ---------- Evaluation ----------
    logger.info("Starting robustness evaluation...")
    start_time = time.time()

    use_clean_data_only = False  # Set to True to test only clean images
    epoch_risk = 0.0
    
    try:
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
                epoch_risk += misclassified_mask.sum().item() / max(len(misclassified_mask), 1)  # Prevent division by zero
                empirical_risk = epoch_risk / max(len(imagenet_data), 1)  # Prevent division by zero
                # ---------- Save Progress ----------
                total_samples = len(stable_indices) + len(unstable_indices)
                unstable_pct = len(unstable_indices) / max(total_samples, 1)  # Prevent division by zero
                if idx % 10 == 0:
                    logger.info(
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
                    try:
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
                    except (OSError, IOError) as e:
                        logger.error(f"Failed to write to CSV file: {str(e)}")

                    # Save checkpoint for recovery
                    try:
                        with open(checkpoint_file, "wb") as f:
                            pickle.dump({
                                "start_idx": idx + 1,
                                "stable_indices": stable_indices,
                                "unstable_indices": unstable_indices,
                                "misclassification_counts": misclassification_counts
                            }, f)
                    except (pickle.PickleError, OSError, IOError) as e:
                        logger.error(f"Failed to save checkpoint: {str(e)}")
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise
    
    elapsed_time = time.time() - start_time
    logger.info(f"Time taken for per-input resilient analysis on {model_name}: {elapsed_time:.2f} seconds")
    
    # ---------- Save Final Results ----------
    if dataset_split == "train":
        logger.info(f"Saving stable and unstable index files for the '{dataset_split}' dataset and model '{model_name}'...")

        try:
            stable_file = f"imagenet_{dataset_split}_{model_name}_stable_indices.pkl"
            with open(stable_file, "wb") as f:
                pickle.dump(stable_indices, f)
            logger.info(f"Saved stable indices to {stable_file}")

            unstable_file = f"imagenet_{dataset_split}_{model_name}_unstable_indices.pkl"
            with open(unstable_file, "wb") as f:
                pickle.dump(unstable_indices, f)
            logger.info(f"Saved unstable indices to {unstable_file}")
            
            # Clean up checkpoint file after successful completion
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
                logger.info(f"Removed checkpoint file: {checkpoint_file}")

        except (pickle.PickleError, OSError, IOError) as e:
            logger.error(f"Failed to save final results: {str(e)}")
            raise
    else:
        logger.info("Validation complete. No files saved.")
        
    # Memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    logger.info(f"Model {model_name} processing completed successfully")
