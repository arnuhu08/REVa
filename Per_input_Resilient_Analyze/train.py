from __future__ import print_function

import argparse
import os
import random
import shutil
import time 
import warnings
import logging
from models import *
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

parser = argparse.ArgumentParser(description='PyTorch Classifiers Training')

parser.add_argument(
    '--dataset',
    type=str,
    default='cifar10',
    choices=['cifar10', 'cifar100'],
    help='Choose between CIFAR-10, CIFAR-100.')


parser.add_argument('-m', '--model', metavar='ARCH', default='vgg16', 
                    choices = [ 'vgg16', 'resnet18','resnet34', 'densenet_cifar' ],
                    help='choose architecture.')

parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--num-classes', default=10,
                    type=int,
                    help= 'number of classes')

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

parser.add_argument(
    '--num-workers',
    type=int,
    default=4,
    help='Number of pre-fetching threads.')

parser.add_argument(
    '--print-freq',
    type=int,
    default=50,
    help='Training loss print frequency (batches).')

parser.add_argument(
    '--val-freq',
    type=int,
    default=1,
    help='Validation frequency (epochs).')

parser.add_argument(
    '--seed',
    type=int,
    default=1,
    help='Random seed for reproducibility.')

args = parser.parse_args()

'''
  from models import VGG
  from torchvision.models import vgg16


  from torchvision.models import resnet18, resnet34
  from models.resnet import ResNet18, ResNet34


  from models.densenet import densenet_cifar
  from models import densenet_cifar


  from torchvision import models
'''

def get_model(model_name, num_classes=10):
    """Get model instance based on name."""
    models = {
        "vgg16": lambda: VGG("VGG16", num_classes=num_classes),
        "resnet18": lambda: ResNet18(num_classes=num_classes),
        "resnet34": lambda: ResNet34(num_classes=num_classes),
        "densenet_cifar": lambda: densenet_cifar(num_classes=num_classes)
    }
    
    if model_name not in models:
        raise ValueError(f"Model {model_name} not supported. Available models: {list(models.keys())}")
    
    return models[model_name]()


def get_lr(step, total_steps, lr_max, lr_min):
  """Compute learning rate according to cosine annealing schedule."""
  return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


def train(net, train_loader, optimizer, scheduler):
    """Train for one epoch."""
    net.train()
    loss_ema = 0.
    device = next(net.parameters()).device
    
    for i, (images, targets) in enumerate(train_loader):
        optimizer.zero_grad()

        images = images.to(device)
        targets = targets.to(device)
        logits = net(images)
        loss = F.cross_entropy(logits, targets)

        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_ema = loss_ema * 0.9 + float(loss) * 0.1
        
        if i % args.print_freq == 0:
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(f'Batch {i:4d} | Train Loss {loss_ema:.3f} | LR {current_lr:.6f}')

    return loss_ema

def test(net, test_loader):
    """Evaluate network on given dataset."""
    net.eval()
    total_loss = 0.
    total_correct = 0
    device = next(net.parameters()).device
    
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            logits = net(images)
            loss = F.cross_entropy(logits, targets)
            pred = logits.max(1)[1]
            total_loss += float(loss)
            total_correct += pred.eq(targets).sum().item()

    avg_loss = total_loss / len(test_loader)
    accuracy = total_correct / len(test_loader.dataset)
    
    return avg_loss, accuracy

def get_dataset(dataset_name, data_dir='./data'):
    """Get dataset loaders based on dataset name."""
    # Normalization values for CIFAR datasets
    if dataset_name == 'cifar10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                       std=[0.2023, 0.1994, 0.2010])
        num_classes = 10
        dataset_class = datasets.CIFAR10
    elif dataset_name == 'cifar100':
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                       std=[0.2675, 0.2565, 0.2761])
        num_classes = 100
        dataset_class = datasets.CIFAR100
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        normalize
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    
    train_data = dataset_class(
        os.path.join(data_dir, dataset_name),
        train=True,
        transform=train_transform,
        download=True
    )
    
    test_data = dataset_class(
        os.path.join(data_dir, dataset_name),
        train=False,
        transform=test_transform,
        download=True
    )
    
    return train_data, test_data, num_classes

def main():
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # Check device availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        logging.warning("CUDA is not available. Training will be slower on CPU.")
    
    # Load datasets
    train_data, test_data, dataset_num_classes = get_dataset(args.dataset)
    
    # Update num_classes based on dataset if not explicitly set
    if args.num_classes != dataset_num_classes:
        logging.info(f"Updating num_classes from {args.num_classes} to {dataset_num_classes} based on dataset")
        args.num_classes = dataset_num_classes

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    # Create model
    logging.info(f"Creating {args.model} model with {args.num_classes} classes")
    net = get_model(args.model, args.num_classes)

    # Setup optimizer
    optimizer = torch.optim.SGD(
        net.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.decay,
        nesterov=True
    )

    # Move model to device
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs")
        net = torch.nn.DataParallel(net)
    
    net = net.to(device)
    
    if torch.cuda.is_available():
        cudnn.benchmark = True

    start_epoch = 0
    best_acc = 0.0

    # Resume from checkpoint if specified
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint['best_acc']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logging.info(f'Model restored from epoch: {start_epoch}, best_acc: {best_acc:.4f}')
        else:
            logging.error(f"No checkpoint found at {args.resume}")
            return

    # Setup learning rate scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr(
            step,
            args.epochs * len(train_loader),
            1,  # lr_lambda computes multiplicative factor
            1e-6 / args.learning_rate
        )
    )

    # Create save directory
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    if not os.path.isdir(args.save):
        raise Exception(f'{args.save} is not a directory')

    # Setup logging
    log_path = os.path.join(
        args.save,
        f'{args.dataset}_{args.model}_training_log.csv'
    )

    # Write CSV header with correct format
    with open(log_path, 'w') as f:
        f.write('epoch,time_sec,train_loss,test_loss,test_error\n')

    logging.info(f'Beginning training from epoch: {start_epoch + 1}')
    logging.info(f'Total epochs: {args.epochs}')
    logging.info(f'Best accuracy so far: {best_acc:.4f}')

    try:
        for epoch in range(start_epoch, args.epochs):
            begin_time = time.time()
            
            # Training
            train_loss_ema = train(net, train_loader, optimizer, scheduler)
            
            # Validation (with configurable frequency)
            if (epoch + 1) % args.val_freq == 0 or epoch == args.epochs - 1:
                test_loss, test_acc = test(net, test_loader)
            else:
                test_loss, test_acc = None, None
            
            epoch_time = time.time() - begin_time
            
            # Save checkpoint
            is_best = test_acc is not None and test_acc > best_acc
            if test_acc is not None:
                best_acc = max(test_acc, best_acc)
            
            checkpoint = {
                'epoch': epoch,
                'dataset': args.dataset,
                'model': args.model,
                'state_dict': net.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'args': vars(args)
            }

            save_path = os.path.join(args.save, 'checkpoint.pth.tar')
            torch.save(checkpoint, save_path)
            
            if is_best:
                shutil.copyfile(save_path, os.path.join(args.save, 'model_best.pth.tar'))
                logging.info(f'New best accuracy: {best_acc:.4f}')

            # Log results
            if test_acc is not None:
                test_error = 100 - 100.0 * test_acc
                with open(log_path, 'a') as f:
                    f.write(f'{epoch + 1},{epoch_time:.0f},{train_loss_ema:.6f},{test_loss:.5f},{test_error:.2f}\n')

                logging.info(
                    f'Epoch {epoch + 1:3d} | Time {epoch_time:5.0f}s | '
                    f'Train Loss {train_loss_ema:.4f} | Test Loss {test_loss:.3f} | '
                    f'Test Error {test_error:.2f}% | Best Acc {best_acc:.4f}'
                )
            else:
                with open(log_path, 'a') as f:
                    f.write(f'{epoch + 1},{epoch_time:.0f},{train_loss_ema:.6f},,\n')
                
                logging.info(
                    f'Epoch {epoch + 1:3d} | Time {epoch_time:5.0f}s | '
                    f'Train Loss {train_loss_ema:.4f} | Best Acc {best_acc:.4f}'
                )
    
    except KeyboardInterrupt:
        logging.info('Training interrupted by user')
    except Exception as e:
        logging.error(f'Training failed with error: {str(e)}')
        raise
    
    logging.info('Training completed!')
    logging.info(f'Best accuracy achieved: {best_acc:.4f}')

if __name__ == '__main__':
  main()
