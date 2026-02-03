'''
    Per-input resilient analyzer implementation.
    Generates perturbed images and analyzes model robustness on local neighborhoods.
'''

import argparse
import os
import pickle
import time
# 3rd party imports
from models import *
import numpy as np
# torch imports 
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms

parser = argparse.ArgumentParser(description='Per-input resilient analyzer implementation')

parser.add_argument(
    '--dataset',
    type=str,
    default='cifar10',
    choices=['cifar10', 'cifar100'],
    help='Choose between CIFAR-10, CIFAR-100.')

parser.add_argument('-m', '--model', metavar='ARCH', default='vgg16', 
                    choices = [ 'vgg16', 'resnet18','resnet34', 'densenet_cifar' ],
                    help='choose architecture.')

parser.add_argument('--num-classes', default=10,
                    type = int,
                    help = 'number of classes')

parser.add_argument(
    '--batch-size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--eval-batch-size', type=int, default=64)

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

args = parser.parse_args()


def neighbors_generation(image, preprocess, samples_size, epsilon):
    """Generate neighbors by adding uniform random noise to the image.
    
    Args:
        image: Original image tensor
        preprocess: Preprocessing transform to apply
        samples_size: Number of perturbed samples to generate
        epsilon: Maximum perturbation magnitude (noise in [-epsilon, epsilon])
    
    Returns:
        Tensor of perturbed images with shape (samples_size, 3, 32, 32)
    """
    new_images = torch.zeros((samples_size, 3, 32, 32))
    for perturbation_image in range(samples_size):
        # Generate uniform noise in [-epsilon, epsilon]
        noise = (torch.rand_like(image) * 2 - 1) * epsilon
        new_image = image + noise
        new_image = new_image.clamp(0, 1)
        new_image = (new_image.permute(1, 2, 0).numpy() * 255).astype('uint8')
        new_images[perturbation_image] = preprocess(new_image)
    return new_images


class NeighborsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, preprocess, sample_size, epsilon):
        self.dataset = dataset
        self.preprocess = preprocess
        self.epsilon = epsilon
        self.sample_size = sample_size

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        neighbors = neighbors_generation(image, self.preprocess, self.sample_size, self.epsilon)

        return (image, neighbors), label

    def __len__(self):
        return len(self.dataset)


def test(net, data):
    """
    Evaluate network robustness on local neighborhoods.
    
    For each sample, generates neighbors and computes the average confidence
    score on mispredictions across all neighbors.
    
    Args:
        net: Neural network model
        data: NeighborsDataset containing original images and their neighbors
    
    Returns:
        Tuple of (images, labels, confidence_scores) where confidence_scores
        represent average misprediction confidence across neighbors
    """
    net.eval()
    confidence_score = []
    new_data, labels = [], []

    test_loader = torch.utils.data.DataLoader(
        data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    total_start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            batch_start_time = time.time()
            
            images, targets = batch
            
            for instance_indx in range(len(targets)):
                confi_score = []
                images_c = images[1][instance_indx]
                images_c = images_c.float()
                target = torch.LongTensor([targets[instance_indx].item()] * len(images_c))
                images_c = images_c.cuda()
                target = target.cuda()
                
                # Get predictions for all neighbors at once
                logits = net(images_c)
                probabilities = F.softmax(logits, dim=1)
                predicted_classes = logits.argmax(dim=1)
                correct_predictions = predicted_classes.eq(target)

                # Compute confidence scores for mispredictions
                for i, correct in enumerate(correct_predictions):
                    confi_score.append(
                        float(probabilities[i, predicted_classes[i]]) if not correct else 0
                    )
                
                # Store average misprediction confidence
                confidence_score.append(sum(confi_score) / len(images_c))
                new_data.append(images[0][instance_indx].cpu())
                labels.append(targets[instance_indx].cpu().item())
                
            batch_elapsed = time.time() - batch_start_time
            print(f'Batch {batch_idx + 1}/{len(test_loader)}: {batch_elapsed:.3f} seconds')

    # Convert to numpy arrays
    new_data_cpu = [tensor.cpu().numpy() for tensor in new_data]
    total_elapsed = time.time() - total_start_time
    print(f'Total execution time: {total_elapsed:.2f} seconds')
    
    return (new_data_cpu, labels, confidence_score)

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(1)
    np.random.seed(1)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires GPU support.")

    # Define transforms
    train_transform = transforms.ToTensor()
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    if args.dataset == 'cifar10':
        train_data = datasets.CIFAR10(
            './data/cifar', train=True, transform=train_transform, download=True
        )
        test_data = datasets.CIFAR10(
            './data/cifar', train=False, transform=preprocess, download=True
        )
    elif args.dataset == 'cifar100':
        train_data = datasets.CIFAR100(
            './data/cifar', train=True, transform=train_transform, download=True
        )
        test_data = datasets.CIFAR100(
            './data/cifar', train=False, transform=preprocess, download=True
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    print(f"Dataset loaded: {len(train_data)} training samples")
    
    # Model initialization
    all_classifiers = {
        "vgg16": VGG("VGG16"),
        "resnet18": ResNet18(),
        "resnet34": ResNet34(),
        "densenet_cifar": densenet_cifar()
    }
    
    if args.model not in all_classifiers:
        raise ValueError(f"Unsupported model: {args.model}")
    
    print(f"Initializing model: {args.model}")
    net = all_classifiers[args.model]
    net = torch.nn.DataParallel(net).cuda()
    cudnn.benchmark = True

    # Load checkpoint if provided
    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"Checkpoint not found: {args.resume}")
        
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume)
        net.load_state_dict(checkpoint['state_dict'])
        print("Checkpoint loaded successfully")
    else:
        print("Warning: No checkpoint provided. Using randomly initialized model.")
    
    # Robustness analysis parameters
    epsilon = 0.22
    sample_size = 50
    
    print(f"\nStarting robustness analysis:")
    print(f"  Epsilon: {epsilon}")
    print(f"  Neighbors per sample: {sample_size}")
    print(f"  Total samples to process: {len(train_data)}\n")
    
    # Create neighbors dataset and analyze
    neighborsData = NeighborsDataset(train_data, preprocess, sample_size, epsilon)
    weakSample_tuples = test(net, neighborsData)

    # Save results
    output_file = 'Scores_tuple.pkl'
    print(f"\nSaving results to {output_file}")
    with open(output_file, 'wb') as f:
        pickle.dump(weakSample_tuples, f)
    
    print(f"Analysis complete! Results saved.")
    print(f"  Total samples analyzed: {len(weakSample_tuples[0])}")
    print(f"  Average confidence score: {np.mean(weakSample_tuples[2]):.4f}")

if __name__ == '__main__':
    main()
    
