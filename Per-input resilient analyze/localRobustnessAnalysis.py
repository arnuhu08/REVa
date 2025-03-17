#Roadmap: implement a function for generating the perturbed images
#the per-input resilient analyzer function
import argparse
import os
import pickle
import random
import shutil
import time 
import warnings
from models import*
import numpy as np
import transformation
from PIL import Image
from itertools import product
from collections import defaultdict

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
    """neighbors generation function
    it generate n_k samples for each data instance"""
    new_images = torch.zeros((samples_size, 3,32,32))
    for perturbation_image in range(samples_size):
        noise = torch.rand_like(image) * epsilon - 0.5 * epsilon
        new_image = image + noise
        new_image = new_image.clamp(0, 1)
        new_image =  (new_image.permute(1,2,0).numpy()*255).astype('uint8')
        new_images[perturbation_image] = preprocess(new_image)
        # new_images[perturbation_image] = new_images[perturbation_image].clamp(0, 1)  # Torch equivalent of `np.clip`
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
    modified  test function to accomodate multiple neighbors for each data instance
    Evaluate the network and compute the fraction of total samples 
    that were mispredicted, weighted by their confidence scores.
    """
    net.eval()
    confidence_score = []
    new_data, labels = [],[]
    # misprediction_count = 0.0

    test_loader = torch.utils.data.DataLoader(
             data,
             batch_size=args.eval_batch_size,
             shuffle=False,
             num_workers=args.num_workers,
           pin_memory=False)

    with torch.no_grad():
        for batch in test_loader:

            start_time = time.time()

            images, targets = batch
            # print(f"the actual classes: {targets}")
            for instance_indx in range(0,len(targets)):
                confi_score = []
                images_c = images[1][instance_indx]
                # print(f"the length of the corrupted dataset is: {len(images_c)}")
                images_c = images_c.float()  # Convert the list of tensors into a single tensor
                target = torch.LongTensor([targets[instance_indx].item()]*len(images_c))
                images_c = images_c.cuda()
                target = target.cuda()
                # print(f"the shape of the neighbors is: {images_c.shape}")
                logits = net(images_c)  # Raw predictions (logits)

                # Apply softmax to get probabilities
                probabilities = F.softmax(logits, dim=1)

                # Predicted class indices
                predicted_classes = logits.argmax(dim=1)

                # print(f"predicted classes are {predicted_classes}")
                # Compare predictions with ground truth
                correct_predictions = predicted_classes.eq(target)

                # For mispredicted samples, accumulate the confidence score
                for i, correct in enumerate(correct_predictions):
                    # Append confidence score for mispredictions; otherwise, append 0
                    confi_score.append(
                        float(probabilities[i, predicted_classes[i]]) if not correct else 0
                    )
                # Append only the original image and target
                confidence_score.append(sum(confi_score)/len(images_c))
                new_data.append(images[0][instance_indx].cpu())  # Original (non-perturbed) image
                labels.append(targets[instance_indx].cpu().item())  # Target class
                # if not correct:  # Mispredicted sample
                #     confidence_score.append(float(probabilities[i, predicted_classes[i]]))
                #     new_data.append(images[0][i].cpu())
                #     labels.append(targets[i].cpu().item())
                # confidence_score.append(0)
                # new_data.append(images[0][i].cpu())
                # labels.append(targets[i].cpu().item())
                
            elapsed_time = time.time() - start_time
            print(f'Execution time for a batch: {elapsed_time:.3f} seconds')
   
    # Move new_data to CPU and convert to list of numpy arrays
    new_data_cpu = [tensor.cpu().numpy() for tensor in new_data]
    final_time = time.time()-start_time
    # Return fraction of total samples weighted by misprediction confidence
    print(f'Execution time for the entire dataset: {final_time:.3f} seconds')
    return (new_data_cpu, labels, confidence_score)

def main():
    torch.manual_seed(1)
    np.random.seed(1)

    train_transform = transforms.ToTensor()
    preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])

    if args.dataset == 'cifar10':
        train_data = datasets.CIFAR10('./data/cifar', train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR10('./data/cifar', train=False, transform=preprocess, download=True)
    
    
    # Model initialization
    all_classifiers  = {
	"vgg16" :  VGG("VGG16"),
	"resnet18": ResNet18(),
	"resnet34": ResNet34(),
    "densenet_cifar": densenet_cifar()
    }
    
    net = all_classifiers[args.model]
    net = torch.nn.DataParallel(net).cuda()
    cudnn.benchmark = True

    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            net.load_state_dict(checkpoint['state_dict'])
    
    epsilon = 0.22
    sample_size = 50
    neighborsData = NeighborsDataset(train_data, preprocess, sample_size, epsilon)
    weakSample_tuples = test(net, neighborsData)

    # weakRobust_samples_by_class = store_data_by_class(weakSample_tuples)
    # strongRobust_samples_by_class = store_data_by_class(strongSample_tuples)

    with open('Scores_tuple', 'wb') as f:
        pickle.dump(weakSample_tuples, f)

    # with open('strongRobust_samples', 'wb') as f:
    #     pickle.dump(strongRobust_samples_by_class, f)

if __name__ == '__main__':
    main()
    
