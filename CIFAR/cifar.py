#==============================================================================
"""Main script to REVa training on CIFAR-10/100.

Supports WideResNet, AllConv, DenseNet models on CIFAR-10 and CIFAR-100 as well
as evaluation on CIFAR-10-C and CIFAR-100-C.

Example usage:
  `python cifar.py`
"""
from __future__ import print_function

import argparse
import os
import shutil
import time
import sys
import logging
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("Error: PIL (Pillow) is required. Install with: pip install Pillow")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Warning: matplotlib not found. Some visualization features may not work.")
    plt = None

try:
    import augmentations
except ImportError:
    print("Error: augmentations module not found. Check file path.")
    sys.exit(1)

try:
    from models.cifar.allconv import AllConvNet
except ImportError:
    print("Error: AllConvNet model not found. Check models directory.")
    AllConvNet = None

try:
    import numpy as np
except ImportError:
    print("Error: numpy is required. Install with: pip install numpy")
    sys.exit(1)

try:
    from third_party.ResNeXt_DenseNet.models.densenet import densenet
except ImportError:
    print("Warning: DenseNet model not found. Check third_party directory.")
    densenet = None

try:
    from third_party.ResNeXt_DenseNet.models.resnext import resnext29
except ImportError:
    print("Warning: ResNeXt model not found. Check third_party directory.")
    resnext29 = None

try:
    from third_party.WideResNet_pytorch.wideresnet import WideResNet
except ImportError:
    print("Warning: WideResNet model not found. Check third_party directory.")
    WideResNet = None

try:
    import torch
    import torch.backends.cudnn as cudnn
    import torch.nn.functional as F
    from torchvision import datasets
    from torchvision import transforms
except ImportError:
    print("Error: PyTorch is required. Install with: pip install torch torchvision")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

parser = argparse.ArgumentParser(
    description='Trains a CIFAR Classifier',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--dataset',
    type=str,
    default='cifar10',
    choices=['cifar10', 'cifar100'],
    help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument(
    '--model',
    '-m',
    type=str,
    default='wrn',
    choices=['wrn', 'allconv', 'densenet'],
    help='Choose architecture.')
# Optimization options
parser.add_argument(
    '--epochs', '-e', type=int, default=100, help='Number of epochs to train.')
parser.add_argument(
    '--learning-rate',
    '-lr',
    type=float,
    default=0.1,
    help='Initial learning rate.')
parser.add_argument(
    '--batch-size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--eval-batch-size', type=int, default=1000)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument(
    '--decay',
    '-wd',
    type=float,
    default=0.0005,
    help='Weight decay (L2 penalty).')
# WRN Architecture options
parser.add_argument(
    '--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='Widen factor')
parser.add_argument(
    '--droprate', default=0.0, type=float, help='Dropout probability')
# AugMix options
parser.add_argument(
    '--mixture-width',
    default=3,
    type=int,
    help='Number of augmentation chains to mix per augmented example')
parser.add_argument(
    '--mixture-depth',
    default=-1,
    type=int,
    help='Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]')
parser.add_argument(
    '--aug-severity',
    default=3,
    type=int,
    help='Severity of base augmentation operators')
parser.add_argument(
    '--no-jsd',
    '-nj',
    action='store_true',
    help='Turn off JSD consistency loss.')
parser.add_argument(
    '--all-ops',
    '-all',
    action='store_true',
    help='Turn on all operations (+brightness,contrast,color,sharpness).')
# Checkpointing options
parser.add_argument(
    '--save',
    '-s',
    type=str,
    default='./snapshots',
    help='Folder to save checkpoints.')
parser.add_argument(
    '--resume',
    '-r',
    type=str,
    default='',
    help='Checkpoint path for resume / test.')
parser.add_argument('--evaluate', action='store_true', help='Eval only.')
parser.add_argument(
    '--print-freq',
    type=int,
    default=50,
    help='Training loss print frequency (batches).')
# Acceleration
parser.add_argument(
    '--num-workers',
    type=int,
    default=4,
    help='Number of pre-fetching threads.')

args = parser.parse_args()

# Validate num_workers based on system capabilities
if args.num_workers < 0:
    args.num_workers = 0
    logging.warning("num_workers set to 0 (negative value not allowed)")
elif args.num_workers > os.cpu_count():
    args.num_workers = min(args.num_workers, os.cpu_count())
    logging.warning(f"num_workers reduced to {args.num_workers} (max CPU count)")

CORRUPTIONS = [
    'brightness','frost','fog','snow', 'contrast',
    'elastic_transform','pixelate','jpeg_compression','zoom_blur','motion_blur', 
    'glass_blur', 'defocus_blur','impulse_noise','gaussian_noise', 'shot_noise'  
]

Adversarial = ['BIM', 'FGSM', 'PGD', 'RFGSM', 'UAP', 'UMIFGSM']


def get_lr(step, total_steps, lr_max, lr_min):
  """Compute learning rate according to cosine annealing schedule."""
  return lr_min + (lr_max - lr_min) * 0.5 * (1 +
                                             np.cos(step / total_steps * np.pi))


def aug(image, preprocess):
  """Perform AugMix augmentations and compute mixture.

  Args:
    image: PIL.Image input image
    preprocess: Preprocessing function which should return a torch tensor.

  Returns:
    mixed: Augmented and mixed image.
  """
  try:
    if not hasattr(augmentations, 'augmentations'):
      raise AttributeError("augmentations module missing 'augmentations' attribute")
    
    aug_list = augmentations.augmentations
    if args.all_ops and hasattr(augmentations, 'augmentations_all'):
      aug_list = augmentations.augmentations_all
    elif args.all_ops:
      logging.warning("all_ops flag set but augmentations_all not found, using default")

    if not aug_list:
      raise ValueError("Empty augmentation list")

    ws = np.float32(np.random.dirichlet([1] * args.mixture_width))
    m = np.float32(np.random.beta(1, 1))

    # Handle potential preprocessing errors
    try:
      base_tensor = preprocess(image)
    except Exception as e:
      logging.error(f"Preprocessing failed for base image: {e}")
      raise

    mix = torch.zeros_like(base_tensor)
    for i in range(args.mixture_width):
      try:
        image_aug = image.copy()
        depth = args.mixture_depth if args.mixture_depth > 0 else np.random.randint(1, 4)
        
        for _ in range(depth):
          op = np.random.choice(aug_list)
          try:
            image_aug = op(image_aug, args.aug_severity)
          except Exception as e:
            logging.warning(f"Augmentation operation failed: {e}, using original image")
            break
        
        # Preprocessing commutes since all coefficients are convex
        try:
          aug_tensor = preprocess(image_aug)
          mix += ws[i] * aug_tensor
        except Exception as e:
          logging.warning(f"Preprocessing failed for augmented image {i}: {e}")
          mix += ws[i] * base_tensor
          
      except Exception as e:
        logging.warning(f"Augmentation chain {i} failed: {e}, using base image")
        mix += ws[i] * base_tensor

    mixed = (1 - m) * base_tensor + m * mix
    return mixed
    
  except Exception as e:
    logging.error(f"Critical error in aug function: {e}")
    # Return original preprocessed image as fallback
    return preprocess(image)


class AugMixDataset(torch.utils.data.Dataset):
  """Dataset wrapper to perform AugMix augmentation."""

  def __init__(self, dataset, adv_dataset, preprocess, no_jsd=False):
    self.dataset = dataset
    self.adv_dataset = adv_dataset
    self.preprocess = preprocess
    self.no_jsd = no_jsd
    
    # Validate datasets
    if self.adv_dataset is not None and len(dataset) != len(adv_dataset):
      raise ValueError(f"Dataset length mismatch: clean={len(dataset)}, adv={len(adv_dataset)}")
    
    logging.info(f"AugMixDataset initialized with {len(dataset)} samples, JSD={not no_jsd}")

  def __getitem__(self, i):
    try:
      if i >= len(self.dataset):
        raise IndexError(f"Index {i} out of range for dataset of size {len(self.dataset)}")
      
      x, y = self.dataset[i]
      
      # Handle adversarial data
      if self.adv_dataset is not None:
        try:
          if isinstance(self.adv_dataset, np.ndarray):
            # Handle case where adv_dataset is a numpy array
            if i >= len(self.adv_dataset):
              logging.warning(f"Adversarial index {i} out of range, using clean image")
              x_adv = x
            else:
              x_adv_data = self.adv_dataset[i]
              if isinstance(x_adv_data, np.ndarray):
                # Convert numpy array to PIL Image
                if x_adv_data.dtype != np.uint8:
                  x_adv_data = (x_adv_data * 255).astype(np.uint8)
                if len(x_adv_data.shape) == 3 and x_adv_data.shape[0] == 3:
                  # Handle CHW format
                  x_adv_data = np.transpose(x_adv_data, (1, 2, 0))
                x_adv = Image.fromarray(x_adv_data)
              else:
                x_adv = x_adv_data
          else:
            # Handle case where adv_dataset is a regular dataset
            x_adv, _ = self.adv_dataset[i]
        except Exception as e:
          logging.warning(f"Failed to load adversarial sample {i}: {e}, using clean image")
          x_adv = x
      else:
        x_adv = x
      
      if self.no_jsd:
        return aug(x, self.preprocess), y
      else:
        try:
          clean_tensor = self.preprocess(x)
          aug1_tensor = aug(x, self.preprocess)
          aug2_tensor = aug(x, self.preprocess)
          adv_tensor = self.preprocess(x_adv)
          
          im_tuple = (clean_tensor, aug1_tensor, aug2_tensor, adv_tensor)
          return im_tuple, y
        except Exception as e:
          logging.error(f"Failed to create augmented tuple for sample {i}: {e}")
          # Fallback to clean image only
          clean_tensor = self.preprocess(x)
          return (clean_tensor, clean_tensor, clean_tensor, clean_tensor), y
    
    except Exception as e:
      logging.error(f"Critical error loading sample {i}: {e}")
      # Return a dummy sample to avoid crashing
      dummy_tensor = torch.zeros(3, 32, 32)  # CIFAR image size
      if self.no_jsd:
        return dummy_tensor, 0
      else:
        return (dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor), 0

  def __len__(self):
    return len(self.dataset)


def train(net, train_loader, optimizer, scheduler):
    """Train the network for one epoch."""
    net.train()
    loss_ema = 0.
    device = next(net.parameters()).device
    
    for i, (images, targets) in enumerate(train_loader):
        try:
            optimizer.zero_grad()

            # Handle both JSD and no-JSD cases
            if isinstance(images, tuple):
                if len(images) != 4:
                    raise ValueError(f"Expected 4 image tensors, got {len(images)}")
                clean_images, aug1_images, aug2_images, adversarial_images = images
            else:
                # No JSD case - only one augmented image
                clean_images = images
                aug1_images = images
                aug2_images = images
                adversarial_images = images

            # Move to device with error handling
            try:
                clean_images = clean_images.to(device)
                aug1_images = aug1_images.to(device)
                aug2_images = aug2_images.to(device)
                adversarial_images = adversarial_images.to(device)
                targets = targets.to(device)
            except Exception as e:
                logging.error(f"Failed to move tensors to device {device}: {e}")
                continue

            # Forward pass and loss calculation for clean images
            try:
                logits_clean = net(clean_images)
                loss_clean = F.cross_entropy(logits_clean, targets)
            except Exception as e:
                logging.error(f"Forward pass failed for clean images: {e}")
                continue

            # Forward pass and loss calculation for augmented images (JSD loss)
            loss_jsd = torch.tensor(0.0).to(device)
            if not args.no_jsd and isinstance(images, tuple):
                try:
                    logits_aug1 = net(aug1_images)
                    logits_aug2 = net(aug2_images)
                    
                    p_clean = F.softmax(logits_clean, dim=1)
                    p_aug1 = F.softmax(logits_aug1, dim=1)
                    p_aug2 = F.softmax(logits_aug2, dim=1)
                    
                    # Add small epsilon to avoid log(0)
                    eps = 1e-8
                    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., eps, 1 - eps).log()
                    
                    loss_jsd = 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                                    F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                                    F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.
                except Exception as e:
                    logging.warning(f"JSD loss calculation failed: {e}, using zero JSD loss")
                    loss_jsd = torch.tensor(0.0).to(device)

            # Forward pass and loss calculation for adversarial images
            loss_adv = torch.tensor(0.0).to(device)
            try:
                logits_adv = net(adversarial_images)
                loss_adv = F.cross_entropy(logits_adv, targets)
            except Exception as e:
                logging.warning(f"Adversarial loss calculation failed: {e}, using zero adv loss")
                loss_adv = torch.tensor(0.0).to(device)

            # Combined loss with weights
            jsd_weight = 0.0 if args.no_jsd else 0.5
            adv_weight = 0.5 if adversarial_images is not None else 0.0
            
            loss = loss_clean + jsd_weight * loss_jsd + adv_weight * loss_adv

            # Check for NaN or infinite loss
            if not torch.isfinite(loss):
                logging.warning(f"Non-finite loss detected: {loss}, skipping batch")
                continue

            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Update EMA with error handling
            loss_value = float(loss.item())
            if np.isfinite(loss_value):
                loss_ema = loss_ema * 0.9 + loss_value * 0.1

            if i % args.print_freq == 0:
                logging.info(f'Batch {i}: Loss {loss_ema:.3f} (Clean: {loss_clean:.3f}, '
                           f'JSD: {loss_jsd:.3f}, Adv: {loss_adv:.3f})')

        except Exception as e:
            logging.error(f"Training batch {i} failed: {e}")
            # Continue with next batch instead of crashing
            continue

    return loss_ema


def test(net, test_loader):
  """Evaluate network on given dataset."""
  net.eval()
  total_loss = 0.
  total_correct = 0
  total_samples = 0
  device = next(net.parameters()).device
  
  with torch.no_grad():
    for batch_idx, (images, targets) in enumerate(test_loader):
      try:
        # Handle case where images might be a tuple (from AugMix dataset)
        if isinstance(images, tuple):
          images = images[0]  # Use clean images for testing
        
        images = images.to(device)
        targets = targets.to(device)
        
        logits = net(images)
        loss = F.cross_entropy(logits, targets, reduction='sum')
        
        pred = logits.argmax(dim=1)
        
        total_loss += float(loss.item())
        total_correct += pred.eq(targets).sum().item()
        total_samples += targets.size(0)
        
      except Exception as e:
        logging.error(f"Test batch {batch_idx} failed: {e}")
        continue

  if total_samples == 0:
    logging.error("No valid test samples processed")
    return float('inf'), 0.0
    
  avg_loss = total_loss / total_samples
  accuracy = total_correct / total_samples
  
  return avg_loss, accuracy

def test_c(net, test_data, base_path):
  """Evaluate network on given corrupted dataset."""
  corruption_accs = []
  base_path = Path(base_path)
  
  if not base_path.exists():
    logging.error(f"Corruption dataset path does not exist: {base_path}")
    return 0.0
  
  # Store original data to restore later
  original_data = test_data.data.copy() if hasattr(test_data.data, 'copy') else None
  original_targets = test_data.targets
  
  try:
    for corruption in CORRUPTIONS:
      try:
        corruption_file = base_path / f'{corruption}.npy'
        labels_file = base_path / 'labels.npy'
        
        if not corruption_file.exists():
          logging.warning(f"Corruption file not found: {corruption_file}")
          continue
          
        if not labels_file.exists():
          logging.warning(f"Labels file not found: {labels_file}")
          continue
        
        # Load corrupted data
        test_data.data = np.load(corruption_file)
        test_data.targets = torch.LongTensor(np.load(labels_file))
        
        # Validate data shape
        if len(test_data.data) != len(test_data.targets):
          logging.error(f"Data/target length mismatch for {corruption}")
          continue
        
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available())

        test_loss, test_acc = test(net, test_loader)
        corruption_accs.append(test_acc)
        print(f'{corruption}\n\tTest Loss {test_loss:.3f} | Test Error {100 - 100. * test_acc:.3f}%')
        
      except Exception as e:
        logging.error(f"Failed to test corruption {corruption}: {e}")
        continue
        
  finally:
    # Restore original data
    if original_data is not None:
      test_data.data = original_data
    test_data.targets = original_targets

  if not corruption_accs:
    logging.error("No corruption tests completed successfully")
    return 0.0
    
  return np.mean(corruption_accs)

def test_a(net, test_data, base_path):
  """Evaluate network on given adversarial dataset."""
  adversarial_accs = []
  base_path = Path(base_path)
  
  if not base_path.exists():
    logging.error(f"Adversarial dataset path does not exist: {base_path}")
    return 0.0
  
  # Store original data to restore later
  original_data = test_data.data.copy() if hasattr(test_data.data, 'copy') else None
  original_targets = test_data.targets
  
  try:
    for attack in Adversarial:
      try:
        attack_file = base_path / f'{attack}.npy'
        labels_file = base_path / 'labels.npy'
        
        if not attack_file.exists():
          logging.warning(f"Adversarial file not found: {attack_file}")
          continue
          
        if not labels_file.exists():
          logging.warning(f"Labels file not found: {labels_file}")
          continue
        
        # Load adversarial data
        test_data.data = np.load(attack_file)
        test_data.targets = torch.LongTensor(np.load(labels_file))
        
        # Validate data shape
        if len(test_data.data) != len(test_data.targets):
          logging.error(f"Data/target length mismatch for {attack}")
          continue
        
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available())

        test_loss, test_acc = test(net, test_loader)
        adversarial_accs.append(test_acc)
        print(f'{attack}\n\tTest Loss {test_loss:.3f} | Test Error {100 - 100. * test_acc:.3f}%')
        
      except Exception as e:
        logging.error(f"Failed to test attack {attack}: {e}")
        continue
        
  finally:
    # Restore original data
    if original_data is not None:
      test_data.data = original_data
    test_data.targets = original_targets

  if not adversarial_accs:
    logging.error("No adversarial tests completed successfully")
    return 0.0
    
  return np.mean(adversarial_accs)


def validate_args(args):
  """Validate command line arguments."""
  if args.epochs <= 0:
    raise ValueError("epochs must be positive")
  if args.learning_rate <= 0:
    raise ValueError("learning_rate must be positive")
  if args.batch_size <= 0:
    raise ValueError("batch_size must be positive")
  if args.mixture_width <= 0:
    raise ValueError("mixture_width must be positive")
  if args.aug_severity < 1 or args.aug_severity > 10:
    raise ValueError("aug_severity must be between 1 and 10")
  
  # Check model availability
  if args.model == 'densenet' and densenet is None:
    raise ValueError("DenseNet model not available")
  if args.model == 'wrn' and WideResNet is None:
    raise ValueError("WideResNet model not available")
  if args.model == 'allconv' and AllConvNet is None:
    raise ValueError("AllConvNet model not available")
  if args.model == 'resnext' and resnext29 is None:
    raise ValueError("ResNeXt model not available")

def main():
  try:
    # Validate arguments
    validate_args(args)
    
    # Set seeds for reproducibility
    torch.manual_seed(1)
    np.random.seed(1)
    if torch.cuda.is_available():
      torch.cuda.manual_seed(1)
      torch.cuda.manual_seed_all(1)
    
    # Check CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    if not torch.cuda.is_available():
      logging.warning("CUDA not available, training on CPU will be very slow")

    # Load datasets with error handling
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4)
    ])
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])
    test_transform = preprocess

    # Create data directory if it doesn't exist
    data_dir = Path('./data/cifar')
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == 'cifar10':
      try:
        train_data = datasets.CIFAR10(
            str(data_dir), train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR10(
            str(data_dir), train=False, transform=test_transform, download=True)
      except Exception as e:
        logging.error(f"Failed to load CIFAR-10: {e}")
        sys.exit(1)
        
      adv_file = 'cifar10_adversarial_dataset.npy'
      base_c_path = './data/cifar/CIFAR-10-C/'
      base_a_path = './data/cifar/CIFAR-10-Adv/'
      num_classes = 10
    else:
      try:
        train_data = datasets.CIFAR100(
            str(data_dir), train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR100(
            str(data_dir), train=False, transform=test_transform, download=True)
      except Exception as e:
        logging.error(f"Failed to load CIFAR-100: {e}")
        sys.exit(1)
        
      adv_file = 'cifar100_adversarial_dataset.npy'
      base_c_path = './data/cifar/CIFAR-100-C/'
      base_a_path = './data/cifar/CIFAR-100-Adv/'
      num_classes = 100
    
    # Load adversarial data with error handling
    adv_data = None
    if Path(adv_file).exists():
      try:
        adv_data = np.load(adv_file)
        logging.info(f"Loaded adversarial dataset: {adv_data.shape}")
      except Exception as e:
        logging.warning(f"Failed to load adversarial dataset {adv_file}: {e}")
        adv_data = None
    else:
      logging.warning(f"Adversarial dataset {adv_file} not found, training without adversarial samples")

    train_data = AugMixDataset(train_data, adv_data, preprocess, args.no_jsd)

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available())

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available())

    # Create model with error handling
    try:
      if args.model == 'densenet':
        if densenet is None:
          raise ValueError("DenseNet model not available")
        net = densenet(num_classes=num_classes)
      elif args.model == 'wrn':
        if WideResNet is None:
          raise ValueError("WideResNet model not available")
        net = WideResNet(args.layers, num_classes, args.widen_factor, args.droprate)
      elif args.model == 'allconv':
        if AllConvNet is None:
          raise ValueError("AllConvNet model not available")
        net = AllConvNet(num_classes)
      elif args.model == 'resnext':
        if resnext29 is None:
          raise ValueError("ResNeXt model not available")
        net = resnext29(num_classes=num_classes)
      else:
        raise ValueError(f"Unknown model: {args.model}")
        
      logging.info(f"Created {args.model} model with {num_classes} classes")
    except Exception as e:
      logging.error(f"Failed to create model: {e}")
      sys.exit(1)

    optimizer = torch.optim.SGD(
        net.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.decay,
        nesterov=True)

    # Move model to appropriate device
    if torch.cuda.is_available():
      net = torch.nn.DataParallel(net).cuda()
      cudnn.benchmark = True
    else:
      net = net.to(device)
    
    logging.info(f"Model moved to device: {device}")
    start_epoch = 0
    best_acc = 0

    # Resume from checkpoint if specified
    if args.resume:
      if os.path.isfile(args.resume):
        try:
          logging.info(f"Loading checkpoint: {args.resume}")
          checkpoint = torch.load(args.resume, map_location=device)
          start_epoch = checkpoint['epoch'] + 1
          best_acc = checkpoint.get('best_acc', 0)
          net.load_state_dict(checkpoint['state_dict'])
          optimizer.load_state_dict(checkpoint['optimizer'])
          logging.info(f'Model restored from epoch: {start_epoch}')
        except Exception as e:
          logging.error(f"Failed to load checkpoint: {e}")
          logging.info("Starting training from scratch")
          start_epoch = 0
          best_acc = 0
      else:
        logging.warning(f"Checkpoint file not found: {args.resume}")
        logging.info("Starting training from scratch")

    # Evaluation mode
    if args.evaluate:
      try:
        # Evaluate clean accuracy first because test_c mutates underlying data
        test_loss, test_acc = test(net, test_loader)
        print(f'Clean\n\tTest Loss {test_loss:.3f} | Test Error {100 - 100. * test_acc:.2f}%')

        test_c_acc = test_c(net, test_data, base_c_path)
        print(f'Mean Corruption Error: {100 - 100. * test_c_acc:.3f}%')
        
        test_a_acc = test_a(net, test_data, base_a_path)
        print(f'Mean Adversarial Error: {100 - 100. * test_a_acc:.3f}%')
      except Exception as e:
        logging.error(f"Evaluation failed: {e}")
      return

    # Training mode
    try:
      scheduler = torch.optim.lr_scheduler.LambdaLR(
          optimizer,
          lr_lambda=lambda step: get_lr(
              step,
              args.epochs * len(train_loader),
              1,  # lr_lambda computes multiplicative factor
              1e-6 / args.learning_rate))

      # Create save directory
      save_dir = Path(args.save)
      save_dir.mkdir(parents=True, exist_ok=True)
      
      if not save_dir.is_dir():
        raise ValueError(f'{args.save} is not a directory')

      log_path = save_dir / f'{args.dataset}_{args.model}_training_log.csv'
      
      # Initialize log file
      try:
        with open(log_path, 'w') as f:
          f.write('epoch,time(s),train_loss,test_loss,test_error(%)\n')
      except Exception as e:
        logging.error(f"Failed to create log file: {e}")
        log_path = None

      logging.info(f'Beginning training from epoch: {start_epoch + 1}')
      
      for epoch in range(start_epoch, args.epochs):
        try:
          begin_time = time.time()

          train_loss_ema = train(net, train_loader, optimizer, scheduler)
          test_loss, test_acc = test(net, test_loader)

          is_best = test_acc > best_acc
          best_acc = max(test_acc, best_acc)
          
          # Save checkpoint
          checkpoint = {
              'epoch': epoch,
              'dataset': args.dataset,
              'model': args.model,
              'state_dict': net.state_dict(),
              'best_acc': best_acc,
              'optimizer': optimizer.state_dict(),
          }

          try:
            save_path = save_dir / 'checkpoint.pth.tar'
            torch.save(checkpoint, save_path)
            if is_best:
              best_path = save_dir / 'model_best.pth.tar'
              shutil.copyfile(save_path, best_path)
          except Exception as e:
            logging.error(f"Failed to save checkpoint: {e}")

          # Log training progress
          epoch_time = time.time() - begin_time
          if log_path:
            try:
              with open(log_path, 'a') as f:
                f.write(f'{epoch + 1:03d},{epoch_time:05.0f},{train_loss_ema:.6f},'
                       f'{test_loss:.5f},{100 - 100. * test_acc:.2f}\n')
            except Exception as e:
              logging.error(f"Failed to write to log file: {e}")

          print(
              f'Epoch {epoch + 1:3d} | Time {int(epoch_time):5d}s | Train Loss {train_loss_ema:.4f} | '
              f'Test Loss {test_loss:.3f} | Test Error {100 - 100. * test_acc:.2f}%')
          
        except Exception as e:
          logging.error(f"Training epoch {epoch + 1} failed: {e}")
          continue

      # Final evaluation
      try:
        test_c_acc = test_c(net, test_data, base_c_path)
        print(f'Mean Corruption Error: {100 - 100. * test_c_acc:.3f}%')
        
        test_a_acc = test_a(net, test_data, base_a_path)
        print(f'Mean Adversarial Error: {100 - 100. * test_a_acc:.3f}%')

        # Log final results
        if log_path:
          try:
            with open(log_path, 'a') as f:
              f.write(f'{args.epochs + 1:03d},0,0,0,{100 - 100 * test_c_acc:.2f}\n')
          except Exception as e:
            logging.error(f"Failed to write final log entry: {e}")
            
      except Exception as e:
        logging.error(f"Final evaluation failed: {e}")
        
    except Exception as e:
      logging.error(f"Training failed: {e}")
      sys.exit(1)
      
  except Exception as e:
    logging.error(f"Main function failed: {e}")
    sys.exit(1)


if __name__ == '__main__':
  main()
