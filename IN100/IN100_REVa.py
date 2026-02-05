"""Main script to launch REVa training on ImageNet-100.

Currently only supports both CNNs and Swin Transformers training.

Example usage:
  `python IN100_REVa.py <path/to/ImageNet> <path/to/ImageNet-C>`
"""
from __future__ import print_function

import argparse
import os
import shutil
import time

import augmentations

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
torch.cuda.empty_cache()
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import models
from torchvision import transforms
from torch.utils.data import Subset
from multiprocessing import Pool
# from third_party.WideResNet_pytorch.wideresnet import WideResNet

def process_image(img):
    return Image.fromarray(img)

def check_nan_inf(tensor, name):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"NaN or Inf detected in {name}!")

augmentations.IMAGE_SIZE = 224
num_classes = 100

model_names = sorted(name for name in models.__dict__
                    if name.islower() and not name.startswith('__') and
                    callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Trains an ImageNet Classifier')
parser.add_argument(
    'clean_data', metavar='DIR', help='path to clean ImageNet dataset')
parser.add_argument(
    'corrupted_data', metavar='DIR_C', help='path to ImageNet-C dataset')
parser.add_argument(
    '--model',
    '-m',
    default='resnet50',
    choices=model_names,
    help='model architecture: ' + ' | '.join(model_names) +
    ' (default: resnet50)')
# Optimization options
parser.add_argument(
    '--epochs', '-e', type=int, default=90, help='Number of epochs to train.')
parser.add_argument(
    '--learning-rate',
    '-lr',
    type=float,
    default=0.1,
    help='Initial learning rate.')
parser.add_argument(
    '--batch-size', '-b', type=int, default=256, help='Batch size.')
parser.add_argument('--eval-batch-size', type=int, default=1000)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument(
    '--decay',
    '-wd',
    type=float,
    default=0.0001,
    help='Weight decay (L2 penalty).')
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
    default=1,
    type=int,
    help='Severity of base augmentation operators')
parser.add_argument(
    '--aug-prob-coeff',
    default=1.,
    type=float,
    help='Probability distribution coefficients')
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
    default=10,
    help='Training loss print frequency (batches).')
parser.add_argument(
    '--pretrained',
    dest='pretrained',
    action='store_true',
    help='use pre-trained model')
# Acceleration
parser.add_argument(
    '--num-workers',
    type=int,
    default=32,
    help='Number of pre-fetching threads.')
#adversarial examples parameters settings:
parser.add_argument(
    '--epsilon',
    type=float,
    default=4/255,
    help='The magnitude of perturbation.')

parser.add_argument(
    '--alpha',
    type=float,
    default=1/255,
    help='The magnitude of step size.')

parser.add_argument(
    '--iteration',
    type=int,
    default=10,
    help='The magnitude of steps.')

args = parser.parse_args()

CORRUPTIONS = [
    'brightness', 'frost', 'fog', 'snow', 'contrast', 
    'elastic_transform', 'pixelate', 'jpeg_compression', 
    'zoom_blur', 'motion_blur', 'glass_blur', 'defocus_blur', 
    'impulse_noise', 'gaussian_noise', 'shot_noise']

# Raw AlexNet errors taken from https://github.com/hendrycks/robustness
ALEXNET_ERR = [
    0.886428, 0.894468, 0.922640, 0.819880, 0.826268, 0.785948, 0.798360,
    0.866816, 0.826572, 0.819324, 0.564592, 0.853204, 0.646056, 0.717840,
    0.606500
]


def adjust_learning_rate(optimizer, epoch):
  """Sets the learning rate to the initial LR (linearly scaled to batch size) decayed by 10 every n / 3 epochs."""
  b = args.batch_size / 256.
  k = args.epochs // 3
  if epoch < k:
    m = 1
  elif epoch < 2 * k:
    m = 0.1
  else:
    m = 0.01
  lr = args.learning_rate * m * b
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
  """Computes the accuracy over the k top predictions for the specified values of k."""
  with torch.no_grad():
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
      correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True) 
      # correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
      res.append(correct_k.mul_(100.0 / batch_size))
    return res


def compute_mce(corruption_accs):
  """Compute mCE (mean Corruption Error) normalized by AlexNet performance."""
  mce = 0.
  uce = 0.
  for i in range(len(CORRUPTIONS)):
    avg_err = 1 - np.mean(corruption_accs[CORRUPTIONS[i]])
    uce+=avg_err*100
    ce = 100 * avg_err / ALEXNET_ERR[i]
    mce += ce / 15
  return mce, uce/15


def aug(image, preprocess):
  """Perform AugMix augmentations and compute mixture.

  Args:
    image: PIL.Image input image
    preprocess: Preprocessing function which should return a torch tensor.

  Returns:
    mixed: Augmented and mixed image.
  """
  aug_list = augmentations.augmentations
  if args.all_ops:
    aug_list = augmentations.augmentations_all

  ws = np.float32(
      np.random.dirichlet([args.aug_prob_coeff] * args.mixture_width))
  m = np.float32(np.random.beta(args.aug_prob_coeff, args.aug_prob_coeff))

  mix = torch.zeros_like(preprocess(image))
  for i in range(args.mixture_width):
    image_aug = image.copy()
    depth = args.mixture_depth if args.mixture_depth > 0 else np.random.randint(
        1, 4)
    for _ in range(depth):
      op = np.random.choice(aug_list)
      image_aug = op(image_aug, args.aug_severity)
    # Preprocessing commutes since all coefficients are convex
    mix += ws[i] * preprocess(image_aug)

  mixed = (1 - m) * preprocess(image) + m * mix
  return mixed

class adversarial_generator(torch.utils.data.Dataset):
    def __init__(self, dataset, label):
        self.dataset = dataset
        self.label = label

    def __getitem__(self, i):
        x = self.dataset[i]
        y = self.label[i]
        x = Image.fromarray(x)
        return x, y
    
    def __len__(self):
        return len(self.dataset)

class AugMixDataset(torch.utils.data.Dataset):
  """Dataset wrapper to perform REVa augmentation."""

  def __init__(self, dataset, adv_dataset, preprocess,preprocess1, no_jsd=False):
    self.dataset = dataset
    self.adv_dataset = adv_dataset
    self.preprocess = preprocess
    self.preprocess1 = preprocess1
    self.no_jsd = no_jsd

  def __getitem__(self, i):
    x, y = self.dataset[i]
    x_adv, y_adv = self.adv_dataset[i]  # Get both x and y from adversarial dataset

    if self.no_jsd:
      return aug(x, self.preprocess), y
    else:
      im_tuple = (self.preprocess(x), aug(x, self.preprocess),
                  aug(x, self.preprocess), self.preprocess1(x_adv))
      return im_tuple, y

  def __len__(self):
    return len(self.dataset)


def train(net, train_loader, optimizer):
  """Train for one epoch."""
  net.train()
  data_ema = 0.
  batch_ema = 0.
  loss_ema = 0.
  acc1_ema = 0.
  acc5_ema = 0.

  end = time.time()
  for i, (images, targets) in enumerate(train_loader):
    # Compute data loading time
    data_time = time.time() - end
    optimizer.zero_grad()
    if not args.no_jsd:
        assert all(torch.is_tensor(element) for element in images), "All elements in the batch should be PyTorch tensors."
    else:
        assert torch.is_tensor(images), "Input should be a PyTorch tensor."
    # print(targets)
    if args.no_jsd:
      images = images.cuda(non_blocking=True)
      targets = targets.cuda(non_blocking=True)
      logits = net(images)
      loss = F.cross_entropy(logits, targets)
      acc1, acc5 = accuracy(logits, targets, topk=(1, 5))  # pylint: disable=unbalanced-tuple-unpacking
    else:
      images_all = torch.cat(images, 0).cuda()
      targets = targets.cuda()
      logits_all = net(images_all)
      logits_clean, logits_aug1, logits_aug2, logits_adv = torch.split(
          logits_all, images[0].size(0))
      
      check_nan_inf(images_all, "images_all")
      check_nan_inf(targets, "targets")


      # Cross-entropy is only computed on clean images
      loss_clean = F.cross_entropy(logits_clean, targets)

      p_clean, p_aug1, p_aug2, p_adv = F.softmax(
          logits_clean, dim=1), F.softmax(
              logits_aug1, dim=1), F.softmax(
                  logits_aug2, dim=1), F.softmax(
                  logits_adv, dim=1)
      
     
      p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2 + p_adv) / 4., 1e-7, 1).log()

      loss_jsd = 16 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                    F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                    F.kl_div(p_mixture, p_aug2, reduction='batchmean') +
                    F.kl_div(p_mixture, p_adv, reduction='batchmean')) / 4.
      
      loss_adv =  F.cross_entropy(logits_adv, targets)

      # Combined loss
      loss = loss_clean + 0.5*loss_jsd + 0.5*loss_adv

      acc1, acc5 = accuracy(logits_clean, targets, topk=(1, 5))  # pylint: disable=unbalanced-tuple-unpacking

    loss.backward()
    optimizer.step()
    
    # Clear gradients and free up memory
    if i % 50 == 0:  # Every 50 batches
        torch.cuda.empty_cache()

    # Compute batch computation time and update moving averages.
    batch_time = time.time() - end
    end = time.time()

    data_ema = data_ema * 0.1 + float(data_time) * 0.9
    batch_ema = batch_ema * 0.1 + float(batch_time) * 0.9
    loss_ema = loss_ema * 0.1 + float(loss) * 0.9
    acc1_ema = acc1_ema * 0.1 + float(acc1) * 0.9
    acc5_ema = acc5_ema * 0.1 + float(acc5) * 0.9

    if i % args.print_freq == 0:
      print(
          'Batch {}/{}: Data Time {:.3f} | Batch Time {:.3f} | Train Loss {:.3f} | Train Acc1 '
          '{:.3f} | Train Acc5 {:.3f}'.format(i, len(train_loader), data_ema,
                                              batch_ema, loss_ema, acc1_ema,
                                              acc5_ema))

  return loss_ema, acc1_ema, batch_ema


def test(net, test_loader):
  """Evaluate network on given dataset."""
  net.eval()
  total_loss = 0.
  total_correct = 0
  with torch.no_grad():
    for images, targets in test_loader:
      images, targets = images.cuda(), targets.cuda()
      logits = net(images)
      loss = F.cross_entropy(logits, targets)
      pred = logits.data.max(1)[1]
      total_loss += float(loss.data)
      total_correct += pred.eq(targets.data).sum().item()

  return total_loss / len(test_loader.dataset), total_correct / len(
      test_loader.dataset)


def test_c(net, test_transform):
  """Evaluate network on given corrupted dataset."""
  corruption_accs = {}
  for c in CORRUPTIONS:
    print(c)
    for s in range(1, 6):
      valdir = os.path.join(args.corrupted_data, c, str(s))
      val_loader = torch.utils.data.DataLoader(
          datasets.ImageFolder(valdir, test_transform),
          batch_size=args.eval_batch_size,
          shuffle=False,
          num_workers=args.num_workers,
          pin_memory=True)

      loss, acc1 = test(net, val_loader)
      if c in corruption_accs:
        corruption_accs[c].append(acc1)
      else:
        corruption_accs[c] = [acc1]

      print('\ts={}: Test Loss {:.3f} | Test Acc1 {:.3f}'.format(
          s, loss, 100. * acc1))

  return corruption_accs

#     return model
def create_model(args, num_classes):
    """Creates a torchvision model (CNN or Transformer) and modifies the classification head."""

    if args.model not in models.__dict__:
        raise ValueError(f"Model '{args.model}' not recognized in torchvision.models")

    if args.pretrained:
        print(f"=> using pre-trained torchvision model '{args.model}'")
        model = models.__dict__[args.model](weights='DEFAULT')  # Works for both CNN and ViT/Swin
    else:
        print(f"=> creating torchvision model '{args.model}' from scratch")
        model = models.__dict__[args.model]()

    # Try to identify and replace the classification head
    if hasattr(model, 'fc') and isinstance(model.fc, torch.nn.Linear):
        
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
        raise ValueError(f"Model '{args.model}' does not have a recognizable classification head ('.fc' or '.head').")

    return model

def unfreeze_layers(net, epoch, freeze_after_epoch=5):
    """Unfreeze all layers of the model after a certain epoch."""
    if epoch == freeze_after_epoch:
        model = net.module if isinstance(net, torch.nn.DataParallel) else net
        for param in model.parameters():
            param.requires_grad = True
        print(f"=> All layers unfrozen at epoch {epoch}")


def main():
  torch.manual_seed(1)
  np.random.seed(1)

  # Load datasets
  mean = [0.485, 0.456, 0.406]
  std = [0.229, 0.224, 0.225]
  
  train_transform = transforms.Compose([
      transforms.RandomResizedCrop(224, antialias=True),
      transforms.RandomHorizontalFlip()
      ])

  preprocess = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean, std)
      ])

  preprocess1 = transforms.Compose([
      train_transform,
      transforms.ToTensor(),
      transforms.Normalize(mean, std)
      ])
  
  test_transform = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      preprocess,
      ])
  
  
  traindir = os.path.join(args.clean_data, 'train')
  valdir = os.path.join(args.clean_data, 'val')
  train_dataset = datasets.ImageFolder(traindir, train_transform)

  # Load adversarial dataset with error handling
  adv_dataset_path = 'adversarial_dataset_swin.npy'
  adv_labels_path = 'adversarial_label.npy'
  
  if not os.path.exists(adv_dataset_path) or not os.path.exists(adv_labels_path):
      raise FileNotFoundError(f"Adversarial dataset files not found: {adv_dataset_path}, {adv_labels_path}")
      
  try:
      loaded_dataset = np.load(adv_dataset_path, mmap_mode='r')
      adversarial_labels = np.load(adv_labels_path, mmap_mode='r')
  except Exception as e:
      raise RuntimeError(f"Error loading adversarial dataset: {e}")

  # Validate adversarial dataset dimensions
  if len(loaded_dataset) != len(adversarial_labels):
      raise ValueError(f"Adversarial dataset size mismatch: {len(loaded_dataset)} vs {len(adversarial_labels)}")
      
  adv_dataset = adversarial_generator(loaded_dataset, adversarial_labels)
  train_dataset = AugMixDataset(train_dataset, adv_dataset, preprocess, preprocess1, args.no_jsd)
  train_loader = torch.utils.data.DataLoader(
      train_dataset,
      batch_size=args.batch_size,
      shuffle=True,
      num_workers=args.num_workers,
      pin_memory=True)
  
  val_loader = torch.utils.data.DataLoader(
      datasets.ImageFolder(valdir, test_transform),
      batch_size=args.batch_size,
      shuffle=False,
      num_workers=args.num_workers,
      pin_memory=True)

  # Check CUDA availability
  if not torch.cuda.is_available():
      raise RuntimeError("CUDA is not available. This script requires GPU support.")
      
  net = create_model(args, num_classes)
  
  optimizer = torch.optim.SGD(
      net.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.decay)

  # Distribute model across all visible GPUs
  net = torch.nn.DataParallel(net).cuda()
  cudnn.benchmark = True

  start_epoch = 0
  best_acc1 = 0

  if args.resume:
    if os.path.isfile(args.resume):
      try:
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch'] + 1
        best_acc1 = checkpoint['best_acc1']
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('Model restored from epoch:', start_epoch)
      except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Starting training from scratch...")
        start_epoch = 0
        best_acc1 = 0
    else:
      print(f"Checkpoint file {args.resume} not found. Starting from scratch...")
      start_epoch = 0
      best_acc1 = 0

  if args.evaluate:
    test_loss, test_acc1 = test(net, val_loader)
    print('Clean\n\tTest Loss {:.3f} | Test Acc1 {:.3f}'.format(
        test_loss, 100 * test_acc1))

    corruption_accs = test_c(net, test_transform)
    for c in CORRUPTIONS:
      print('\t'.join(map(str, [c] + corruption_accs[c])))

    print('mCE (normalized by AlexNet): ', compute_mce(corruption_accs)[0])
    print('uCE (average of unnormalized CE): ', compute_mce(corruption_accs)[1])
    return

  if not os.path.exists(args.save):
    os.makedirs(args.save)
  if not os.path.isdir(args.save):
    raise Exception('%s is not a dir' % args.save)

  log_path = os.path.join(args.save,
                          'imagenetREVa1_{}_training_log.csv'.format(args.model))
  with open(log_path, 'w') as f:
    f.write(
        'epoch,batch_time,train_loss,train_acc1(%),test_loss,test_acc1(%)\n')

  best_acc1 = 0
  print('Beginning training from epoch:', start_epoch + 1)
  for epoch in range(start_epoch, args.epochs):
    adjust_learning_rate(optimizer, epoch)

    # Unfreeze layers after certain epochs
    unfreeze_layers(net, epoch)

    train_loss_ema, train_acc1_ema, batch_ema = train(net, train_loader,
                                                      optimizer)
    test_loss, test_acc1 = test(net, val_loader)

    is_best = test_acc1 > best_acc1
    best_acc1 = max(test_acc1, best_acc1)
    checkpoint = {
        'epoch': epoch,
        'model': args.model,
        'state_dict': net.state_dict(),
        'best_acc1': best_acc1,
        'optimizer': optimizer.state_dict(),
    }

    save_path = os.path.join(args.save, f'REVacheckpoint_{args.model}.pth.tar')
    torch.save(checkpoint, save_path)
    if is_best:
      shutil.copyfile(save_path, os.path.join(args.save, f'REVamodel_{args.model}_best.pth.tar'))

    with open(log_path, 'a') as f:
      f.write('%03d,%0.3f,%0.6f,%0.2f,%0.5f,%0.2f\n' % (
          (epoch + 1),
          batch_ema,
          train_loss_ema,
          100. * train_acc1_ema,
          test_loss,
          100. * test_acc1,
      ))

    print(
        'Epoch {:3d} | Train Loss {:.4f} | Test Loss {:.3f} | Test Acc1 '
        '{:.2f}'
        .format((epoch + 1), train_loss_ema, test_loss, 100. * test_acc1))

  corruption_accs = test_c(net, test_transform)
  for c in CORRUPTIONS:
    print('\t'.join(map(str, [c] + corruption_accs[c])))

  print('mCE (normalized by AlexNet): ', compute_mce(corruption_accs)[0])
  print('uCE (average of unnormalized CE): ', compute_mce(corruption_accs)[1])

if __name__ == '__main__':
  main()
