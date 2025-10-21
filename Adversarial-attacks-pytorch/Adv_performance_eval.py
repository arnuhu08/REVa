import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Subset, DataLoader, TensorDataset
from torchvision import models, datasets, transforms
from utils import get_accuracy
import torchattacks
from typing import Optional, Callable, Tuple

parent_directory = '.' # Adjust this path as needed
sys.path.append(parent_directory)

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_model(num_classes):
    """Creates a torchvision model (CNN or Transformer) and modifies the classification head."""

    model = models.__dict__['swin_v2_b']()

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

net_base = create_model(100)
net_base = torch.nn.DataParallel(net_base).cuda()
cudnn.benchmark = True
checkpoint =  torch.load('/PATH/TO/augmix/snapshots/IN100checkpoint.pth.tar')
net_base.load_state_dict(checkpoint['state_dict'])

net_AX = create_model(100)
net_AX = torch.nn.DataParallel(net_AX).cuda()
cudnn.benchmark = True
checkpoint =  torch.load('/PATH/TO/augmix/snapshots/AXcheckpoint.pth.tar')
net_AX.load_state_dict(checkpoint['state_dict'])

net_REVa = create_model(100)
net_REVa = torch.nn.DataParallel(net_REVa).cuda()
cudnn.benchmark = True
checkpoint =  torch.load('/PATH/TO/augmix/snapshots/REVacheckpoint_{args.model}.pth.tar')
net_REVa.load_state_dict(checkpoint['state_dict'])

# Assuming PREPROCESSINGS['Res256Crop224'] is a predefined transformation
# If not, you can define it manually like below:
PREPROCESSINGS = {
    'Res256Crop224': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
}

def load_imagenet(
    n_examples: Optional[int] = 500,
    data_dir: str = './data',
    transforms_test: Callable = PREPROCESSINGS['Res256Crop224']
) -> Tuple[torch.Tensor, torch.Tensor]:
    if n_examples > 5000:
        raise ValueError(
            'The evaluation is currently possible on at most 5000 points.')

    # Use PyTorch's built-in ImageFolder
    imagenet_dataset = datasets.ImageFolder(root=data_dir + '/val', transform=transforms_test)

    # Create DataLoader for the test set
    test_loader = DataLoader(imagenet_dataset, batch_size=n_examples, shuffle=False, num_workers=4)

    # Get a batch of data
    x_test, y_test = next(iter(test_loader))

    return x_test, y_test

def train_data_loader(
    n_examples: Optional[int] = 5000,
    data_dir: str = './data',
    transforms_test: Callable = PREPROCESSINGS['Res256Crop224']
) -> Tuple[torch.Tensor, torch.Tensor]:
    if n_examples > 5000:
        raise ValueError(
            'The evaluation is currently possible on at most 5000 points.')

    # Use PyTorch's built-in ImageFolder
    imagenet_dataset = datasets.ImageFolder(root=data_dir + '/train', transform=transforms_test)

    # Create DataLoader for the test set
    test_loader = DataLoader(imagenet_dataset, batch_size=n_examples, shuffle=True, num_workers=4)


    return test_loader\
    

def test_data_loader(
    n_examples: Optional[int] = 5000,
    data_dir: str = './data',
    transforms_test: Callable = PREPROCESSINGS['Res256Crop224']
) -> Tuple[torch.Tensor, torch.Tensor]:
    if n_examples > 5000:
        raise ValueError(
            'The evaluation is currently possible on at most 5000 points.')

    # Use PyTorch's built-in ImageFolder
    imagenet_dataset = datasets.ImageFolder(root=data_dir + '/val', transform=transforms_test)

    # Create DataLoader for the test set
    test_loader = DataLoader(imagenet_dataset, batch_size=n_examples, shuffle=False, num_workers=4)


    return test_loader

x_test, y_test = load_imagenet(n_examples=5000, data_dir= 'PATH/TO/IN100')

# Assessing the models performances on adversarial generation types
atk = torchattacks.PGD(net_base, eps=4/255, alpha=1/225, steps=40, random_start=True)
atk.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
print(f"Assessing the models performances on adversarial generation types:\n{atk}")

def generate_adversarial_dataset_in_batches(x_test, y_test, atk, batch_size=50):
    # Initialize an empty list to store adversarial images
    adv_images_list = []

    # Get the number of samples in the test dataset
    num_samples = x_test.size(0)
    
    # Process the dataset in batches
    for i in range(0, num_samples, batch_size):
        # Select the current batch of images and labels
        x_batch = x_test[i:i + batch_size]
        y_batch = y_test[i:i + batch_size]
        
        # Generate adversarial examples for the current batch
        adv_batch = atk(x_batch, y_batch)
        
        # Append the generated adversarial examples to the list
        adv_images_list.append(adv_batch)
    
    # Concatenate the list of adversarial examples into a single tensor
    adv_images = torch.cat(adv_images_list, dim=0)

    return adv_images

adv_images = generate_adversarial_dataset_in_batches(x_test, y_test, atk, batch_size=32)
val_loader = test_data_loader(n_examples=100, data_dir= '/PATH/TO/IN100')
# val_loader.data =  adv_images
# val_loader.targets = y_test

acc1 = get_accuracy(net_base, [(adv_images.to(device), y_test.to(device))])
acc2 = get_accuracy(net_AX, [(adv_images.to(device), y_test.to(device))])
acc3 = get_accuracy(net_REVa, [(adv_images.to(device), y_test.to(device))])
print('Base model Acc: %2.2f %%'%(acc1))
print('AX model Acc: %2.2f %%'%(acc2))
print('REVa model Acc: %2.2f %%'%(acc3))

# FGSM adversarial performance evaluation
atk1 = torchattacks.FGSM(net_base, eps=4/255)
atk1.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
print(f"FGSM adversarial performance evaluation:\n{atk1}")

adv_images_FGSM = generate_adversarial_dataset_in_batches(x_test, y_test, atk1, batch_size=32)
acc1 = get_accuracy(net_base, [(adv_images_FGSM.to(device), y_test.to(device))])
acc2 = get_accuracy(net_AX, [(adv_images_FGSM.to(device), y_test.to(device))])
acc3 = get_accuracy(net_REVa, [(adv_images_FGSM.to(device), y_test.to(device))])
print('Base model Acc: %2.2f %%'%(acc1))
print('AX model Acc: %2.2f %%'%(acc2))
print('REVa model Acc: %2.2f %%'%(acc3))

# BIM adversarial performance evaluation
atk2 = torchattacks.BIM(net_base, eps=4/255, alpha=1/255, steps=10)
atk2.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
print(f"BIM adversarial performance evaluation:\n{atk2}")

adv_images_BIM = generate_adversarial_dataset_in_batches(x_test, y_test, atk2, batch_size=16)

# Create dataset and dataloader
torch.cuda.empty_cache()
adv_dataset = TensorDataset(adv_images_BIM, y_test)
adv_loader = DataLoader(adv_dataset, batch_size=128, shuffle=False)

# Get accuracy
acc1 = get_accuracy(net_base, adv_loader, device=device)
acc2 = get_accuracy(net_AX, adv_loader, device=device)
acc3 = get_accuracy(net_REVa, adv_loader, device=device)

print(f'Base model Acc: {acc1:.2f} %')
print(f'AX model Acc: {acc2:.2f} %')
print(f'REVa model Acc: {acc3:.2f} %')

# RFGSM adversarial performance evaluation
atk3 = torchattacks.RFGSM(net_base, eps=4/255, alpha=1/255, steps=10)
atk3.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
print(f"RFGSM adversarial performance evaluation:\n{atk3}")

adv_images_RFGSM = generate_adversarial_dataset_in_batches(x_test, y_test, atk3, batch_size=32)

# Create dataset and dataloader
adv_dataset = TensorDataset(adv_images_RFGSM, y_test)
adv_loader = DataLoader(adv_dataset, batch_size=128, shuffle=False)

# Get accuracy
acc1 = get_accuracy(net_base, adv_loader, device=device)
acc2 = get_accuracy(net_AX, adv_loader, device=device)
acc3 = get_accuracy(net_REVa, adv_loader, device=device)

print(f'Base model Acc: {acc1:.2f} %')
print(f'AX model Acc: {acc2:.2f} %')
print(f'REVa model Acc: {acc3:.2f} %')

# MIFGSM adversarial performance evaluation
atk4 = torchattacks.MIFGSM(net_base, eps=4/255, alpha=1/255, steps=10)
atk4.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
print(f"MIFGSM adversarial performance evaluation:\n{atk4}")

adv_images_MIFGSM = generate_adversarial_dataset_in_batches(x_test, y_test, atk4, batch_size=32)

# Create dataset and dataloader
adv_dataset = TensorDataset(adv_images_MIFGSM, y_test)
adv_loader = DataLoader(adv_dataset, batch_size=128, shuffle=False)

# Get accuracy
acc1 = get_accuracy(net_base, adv_loader, device=device)
acc2 = get_accuracy(net_AX, adv_loader, device=device)
acc3 = get_accuracy(net_REVa, adv_loader, device=device)

print(f'Base model Acc: {acc1:.2f} %')
print(f'AX model Acc: {acc2:.2f} %')
print(f'REVa model Acc: {acc3:.2f} %')

class UAPAttack:
    def __init__(self, model, data_loader, epsilon=0.1, alpha=0.01, max_iter=100, device='cuda'):
        """
        Parameters:
        - model: The target model.
        - data_loader: The dataset (DataLoader) for which we will generate the UAP.
        - epsilon: Perturbation step size.
        - alpha: Learning rate for the perturbation update.
        - max_iter: Maximum iterations for optimization.
        - device: Device to run the computation ('cpu' or 'cuda').
        """
        self.model = model
        self.data_loader = data_loader
        self.epsilon = epsilon
        self.alpha = alpha
        self.max_iter = max_iter
        self.device = device

        # Initialize perturbation (delta) with small random values instead of zeros
        # This helps avoid starting with a black background (zeros).
        self.delta = torch.randn_like(next(iter(self.data_loader))[0][0]).to(self.device) * 0.001
        self.delta.requires_grad = True  # Allow gradients to be computed for delta

    def generate_uap(self):
        # Optimizer for perturbation
        optimizer = optim.Adam([self.delta], lr=self.alpha)

        for iteration in range(self.max_iter):
            total_loss = 0
            correct = 0
            total = 0
            for inputs, targets in self.data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Apply the perturbation to the inputs
                perturbed_inputs = inputs + self.delta
                perturbed_inputs = torch.clamp(perturbed_inputs, 0, 1)  # Clamp the values to be in [0, 1]

                # Forward pass
                outputs = self.model(perturbed_inputs)
                loss = torch.nn.CrossEntropyLoss()(outputs, targets)  # Standard cross-entropy loss

                # Backward pass to compute gradients of delta
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Evaluate the effectiveness of the attack (whether the model misclassifies)
                _, predicted = outputs.max(1)
                correct += (predicted != targets).sum().item()
                total += targets.size(0)

            # Print the current status of the attack
            print(f"Iteration {iteration+1}/{self.max_iter}, Loss: {loss.item()}, Misclassification rate: {100 * correct / total}%")

            # Check if perturbation exceeds epsilon (clipping)
            self.delta.data = torch.clamp(self.delta.data, -self.epsilon, self.epsilon)

        return self.delta
    
# Instantiate UAP attack
uap_attack = UAPAttack(net_base, val_loader, epsilon=0.1, alpha=0.01, max_iter=10, device='cuda')

# Generate the UAP perturbation
uap = uap_attack.generate_uap()

# Assuming UAPAttack class is already defined

def generate_adversarial_dataset_in_batches_uap(x_test, y_test, uap, batch_size=50):
    """
    Generate an adversarial dataset using a universal adversarial perturbation (UAP).

    Parameters:
    - x_test: Input test data (images).
    - y_test: Corresponding labels for the test data.
    - atk: The UAPAttack object used for generating the universal perturbation.
    - batch_size: The size of the batch to process at a time.

    Returns:
    - adv_images: A tensor containing the adversarial images generated by applying the UAP.
    """
    # Initialize an empty list to store adversarial images
    adv_images_list = []

    # Get the number of samples in the test dataset
    num_samples = x_test.size(0)
    
    # Generate the UAP perturbation (only need to generate once)
    
    # Process the dataset in batches
    for i in range(0, num_samples, batch_size):
        # Select the current batch of images and labels
        x_batch = x_test[i:i + batch_size]
        y_batch = y_test[i:i + batch_size]
        
        # Apply the universal perturbation (UAP) to the batch
        adv_batch = x_batch + uap.cpu().detach()
        adv_batch = torch.clamp(adv_batch, 0, 1)  # Ensure the pixel values remain in the valid range [0, 1]
        
        # Append the generated adversarial examples to the list
        adv_images_list.append(adv_batch)
    
    # Concatenate the list of adversarial examples into a single tensor
    adv_images = torch.cat(adv_images_list, dim=0)

    return adv_images

# Example usage:
# Assuming you have a pretrained model and a data loader `x_test` and `y_test` as your test dataset

# Generate adversarial dataset using the UAP
adv_images_uap = generate_adversarial_dataset_in_batches_uap(x_test, y_test, uap, batch_size=32)
# Create dataset and dataloader
adv_dataset = TensorDataset(adv_images_uap, y_test)
adv_loader = DataLoader(adv_dataset, batch_size=128, shuffle=False)

# Get accuracy
acc1 = get_accuracy(net_base, adv_loader, device=device)
acc2 = get_accuracy(net_AX, adv_loader, device=device)
acc3 = get_accuracy(net_REVa, adv_loader, device=device)

print(f'Base model Acc: {acc1:.2f} %')
print(f'AX model Acc: {acc2:.2f} %')
print(f'REVa model Acc: {acc3:.2f} %')

# Given data
corruption_types = [
    [0.00, 22.54, 0.02, 0.04, 0.16, 78.16],   
    [48.82, 50.86, 37.72, 39.56, 25.84, 63.58], 
    [75.30, 66.96, 72.50, 72.5, 65.30,49.60 ]
] 

# test results
corruption_array = np.array(corruption_types)
print(f"these are the corruption error {100-corruption_array}")
averages = corruption_array.mean(axis=1)
result = np.round(100 - averages, 7)

print(result)
