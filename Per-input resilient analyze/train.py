from __future__ import print_function

import argparse
import os
import random
import shutil
import time 
import warnings
from models import*
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms

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
		     type = int,
		     help = 'number of classes')

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

args = parser.parse_args()

all_classifiers  = {
	"vgg16" :  VGG("VGG16"),
	"resnet18": ResNet18(),
	"resnet34": ResNet34(),
  "densenet_cifar": densenet_cifar()
}


def get_lr(step, total_steps, lr_max, lr_min):
  """Compute learning rate according to cosine annealing schedule."""
  return lr_min + (lr_max - lr_min) * 0.5 * (1 +
                                             np.cos(step / total_steps * np.pi))


def train(net, train_loader, optimizer, scheduler):
  """Train for one epoch."""
  net.train()
  loss_ema = 0.
  for i, (images, targets) in enumerate(train_loader):
    optimizer.zero_grad()

    images = images.cuda()
    targets = targets.cuda()
    logits = net(images)
    loss = F.cross_entropy(logits, targets)

    loss.backward()
    optimizer.step()
    scheduler.step()
    loss_ema = loss_ema * 0.9 + float(loss) * 0.1
    if i % args.print_freq == 0:
      print('Train Loss {:.3f}'.format(loss_ema))

  return loss_ema
 
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

def main():
  torch.manual_seed(1)
  np.random.seed(1)

  # Load datasets
  preprocess = transforms.Compose(
      [transforms.ToTensor(),
       transforms.Normalize([0.5] * 3, [0.5] * 3)])

  train_transform = transforms.Compose(
      [transforms.RandomHorizontalFlip(),
       transforms.RandomCrop(32, padding=4),
       preprocess
       ])

  test_transform = preprocess

  if args.dataset == 'cifar10':
    train_data = datasets.CIFAR10(
        './data/cifar', train=True, transform=train_transform, download=True)
    test_data = datasets.CIFAR10(
        './data/cifar', train=False, transform=test_transform, download=True)

#train_data loading 
  train_loader = torch.utils.data.DataLoader(
      train_data,
      batch_size=args.batch_size,
      shuffle=True,
      num_workers=args.num_workers,
      pin_memory=True)

  test_loader = torch.utils.data.DataLoader(
      test_data,
      batch_size=args.eval_batch_size,
      shuffle=False,
      num_workers=args.num_workers,
      pin_memory=True)

  if args.model == 'vgg16':
    net = all_classifiers[args.model]
  elif args.model == 'resnet18':
    net = all_classifiers[args.model]
  elif args.model == 'resnet34':
    net = all_classifiers[args.model]
  elif args.model == 'densenet_cifar':
    net = all_classifiers[args.model]

  optimizer = torch.optim.SGD(
     net.parameters(),
     args.learning_rate,
     momentum =  args.momentum,
     weight_decay = args.decay,
     nesterov=True)

# Distribute model across all visible GPUs
  net = torch.nn.DataParallel(net).cuda()
  cudnn.benchmark = True

  start_epoch = 0

  if args.resume:
    if os.path.isfile(args.resume):
      checkpoint = torch.load(args.resume)
      start_epoch = checkpoint['epoch'] + 1
      best_acc = checkpoint['best_acc']
      net.load_state_dict(checkpoint['state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer'])
      print('Model restored from epoch:', start_epoch)
  

  scheduler = torch.optim.lr_scheduler.LambdaLR(
      optimizer,
      lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
          step,
          args.epochs * len(train_loader),
          1,  # lr_lambda computes multiplicative factor
          1e-6 / args.learning_rate))

  if not os.path.exists(args.save):
    os.makedirs(args.save)
  if not os.path.isdir(args.save):
    raise Exception('%s is not a dir' % args.save)

  
  log_path = os.path.join(args.save,
                          args.dataset + '_' + args.model + '_training_log.csv')

  with open(log_path, 'w') as f:
    f.write('epoch, time(s), train_loss\n') 
  

  best_acc = 0

  print('Beginning training from epoch:', start_epoch + 1)
  
  for epoch in range(start_epoch, args.epochs):
    begin_time = time.time()
    train_loss_ema = train(net, train_loader, optimizer, scheduler)
    test_loss, test_acc = test(net, test_loader)

    is_best = test_acc > best_acc
    best_acc = max(test_acc, best_acc)
    checkpoint = {
         'epoch': epoch,
         'dataset': args.dataset,
         'model': args.model, 
         'state_dict': net.state_dict(),
         'best_acc':best_acc,
         'optimizer': optimizer.state_dict(),} 

    save_path = os.path.join(args.save, 'checkpoint.pth.tar')
    torch.save(checkpoint, save_path)
    if is_best:
       shutil.copyfile(save_path, os.path.join(args.save, 'model_best.pth.tar'))

    with open(log_path, 'a') as f:
      f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' % (
          (epoch + 1),
          time.time() - begin_time,
          train_loss_ema,
          test_loss,
          100 - 100. * test_acc,))
     
    print(
        'Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} |'
        ' Test Error {4:.2f}'
        .format((epoch + 1), int(time.time() - begin_time), train_loss_ema,
                test_loss, 100 - 100. * test_acc))
     
if __name__ == '__main__':
  main()
   
   
  


