import sys
import os

parent_directory = '.'  # Replace with the actual path to the parent directory
sys.path.append(parent_directory)

import os
import torchattacks
import numpy as np
from PIL import Image
import torch
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import models
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision.datasets as dsets
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_names = sorted(name for name in models.__dict__
                    if name.islower() and not name.startswith('__') and
                    callable(models.__dict__[name]))

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

net_base = create_model(100)
net_base = torch.nn.DataParallel(net_base).cuda()
cudnn.benchmark = True
checkpoint =  torch.load('/home/abdulrauf/Desktop/augmix/snapshots/IN100checkpoint.pth.tar')
net_base.load_state_dict(checkpoint['state_dict'])


# model = models.__dict__['resnet50'](pretrained=True)
# net = ResNet_Model('resnet18', 100)
# net = torch.nn.DataParallel(net).cuda()
# cudnn.benchmark = True
# checkpoint =  torch.load('/home/abdulrauf/Desktop/augmix/snapshots/IN100checkpoint.pth.tar')
# net.load_state_dict(checkpoint['state_dict'])

def image_folder_custom_label(root, transform, idx2label) :
    # custom_label
    # type : List
    # index -> label
    # ex) ['tench', 'goldfish', 'great_white_shark', 'tiger_shark']
    
    old_data = dsets.ImageFolder(root=root, transform=transform)
    old_classes = old_data.classes
    
    label2idx = {}
    
    for i, item in enumerate(idx2label) :
        label2idx[item] = i
    
    new_data = dsets.ImageFolder(root=root, transform=transform, 
                                target_transform=lambda x : idx2label.index(old_classes[x]))
    new_data.classes = idx2label
    new_data.class_to_idx = label2idx

    return new_data

def load_labels_from_txt(file_path):
    """Loads class labels from a .txt file, where each line contains a label."""
    with open(file_path, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

def get_imagenet_data():
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # Load ImageNet class labels from a .txt file instead of JSON
    idx2label = load_labels_from_txt("/home/abdulrauf/Desktop/ImageNet-100-Pytorch/IN100.txt")

    # Define transformation pipeline
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
        # transforms.Normalize(mean=MEAN, std=STD)
    ])

    # Load dataset with custom labels
    imagnet_data = image_folder_custom_label(root=parent_directory + '/train', 
                                            transform=transform,
                                            idx2label=idx2label)

    # Create DataLoader
    data_loader = torch.utils.data.DataLoader(imagnet_data, batch_size=256, shuffle=False)

    print("Used normalization: mean=", MEAN, "std=", STD)
    
    # Return the first batch
    # return next(iter(data_loader))
    return data_loader

atk = torchattacks.PGD(net_base.to(device), eps=4/255, alpha=1/255, steps=10, random_start=True)
atk.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
print(atk)

def adv_generator(train_loader):
    adv_batches = []
    label_batches = []
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        adv_batch = atk(images, labels) 
        adv_batches.append(adv_batch.cpu().detach())
        label_batches.append(labels.cpu().detach())
    # Concatenate images and labels
    adv_images = torch.cat(adv_batches, dim=0)
    adv_labels = torch.cat(label_batches, dim=0)
    
    # Convert adversarial images to numpy, clip values to [0, 1], and save in uint8
    adv_images = adv_images.permute(0, 2, 3, 1).numpy()  # Convert to HWC format
    adv_images = np.clip(adv_images, 0, 1)  # Clip to valid range
    adv_images = (adv_images * 255).astype(np.uint8)  # Scale to [0, 255] and convert to uint8
    return adv_images, adv_labels

train_loader = get_imagenet_data()
adversarial_dataset = adv_generator(train_loader)
adv_imgs = adversarial_dataset[0]
adv_labels = adversarial_dataset[1]
adv_labels[120000]

plt.figure(figsize=(2,2))
plt.imshow(adv_imgs[20])

np.save(parent_directory + '/adversarial_dataset_swin.npy', adversarial_dataset[0])
np.save(parent_directory + '/adversarial_label_swin.npy', adversarial_dataset[1])

path = parent_directory + '/IN100'
train_transform = transforms.Compose(
                                    [transforms.ToTensor(),
                                    transforms.RandomResizedCrop(224, antialias=True),
                                    transforms.RandomHorizontalFlip()
                                    ])
traindir =  os.path.join(path, 'train')
dataset = datasets.ImageFolder(traindir, train_transform)

img, label = dataset[2]
plt.imshow(img.permute(1,2,0))

train_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
first_batch =  next(iter(train_loader))
first_batch[0][0]
converted_img = ((first_batch[0][1]).permute(1,2,0).numpy()*255).astype(np.uint8)

plt.figure(figsize=(2,2))
plt.imshow(converted_img)

def adv_generator(train_loader):
    adv_batches = []
    label_batches = []
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        adv_batch = atk(images, labels) 
        adv_batches.append(adv_batch.cpu().detach())
        label_batches.append(labels.cpu().detach())
    # Concatenate images and labels
    adv_images = torch.cat(adv_batches, dim=0)
    adv_images = (adv_images.permute(0,2,3,1).numpy()*255).astype(np.uint8)
    adv_labels = torch.cat(label_batches, dim=0)
    return adv_images, adv_labels

adversarial_dataset = adv_generator(train_loader)
x = adversarial_dataset[0]

class adversarial_generator(torch.utils.data.Dataset):
    def __init__(self, dataset, label):
        self.dataset = dataset
        self.label = label

    def __getitem__(self, i):
        x = self.dataset[i]
        y = self.label[i]
        x = Image.fromarray(x)
        return x,y
    
    def __len__(self):
        return len(self.dataset)

adv_dataset = adversarial_generator(adversarial_dataset)
img,y =  adv_dataset[4]

np.save(parent_directory + '/adversarial_dataset.npy', adversarial_dataset[0])
np.save(parent_directory + '/adversarial_label.npy', adversarial_dataset[1])

loaded_dataset = np.load('adversarial_dataset.npy')
labels = np.load('adversarial_label.npy')

adv_dataset = adversarial_generator(loaded_dataset, labels)
img, label = adv_dataset[0]

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load the dataset and apply transformations
dataset = datasets.ImageFolder(traindir, test_transform)
dataset.data = loaded_dataset
dataset.targets = labels
val_loader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=False, num_workers=4)

model=model.to(device)

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

loss, acc1 =  test(model, val_loader)
# 100-100*acc1

plt.figure(figsize=(2,2))
plt.imshow(loaded_dataset[3])
loaded_dataset[3]
# print(len(labels))

# adversarial_dataset = torch.cat(adversarial_dataset,dim=0)
# adversarial_dataset = (adversarial_dataset.permute(0,2,3,1).numpy()*255).astype(np.uint8)
imgs,labels = adversarial_dataset
# print(len(imgs))

plt.figure(figsize=(2,2))
plt.imshow(imgs[0])
images, labels = first_batch

plt.figure(figsize=(2,2))
plt.imshow(images[0].permute(1,2,0))
plt.show()

adv_images = atk(images, labels)
adv_img =  adv_images[0]
plt.figure(figsize=(2,2))
plt.imshow(adv_img.permute(1,2,0))
plt.show()

class AugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix augmentation."""

    def __init__(self, dataset, preprocess):
        self.dataset = dataset
        self.preprocess = preprocess
        

    def __getitem__(self, i):
        x, y = self.dataset[i]
        x = self.preprocess(x)
        adv_data = adv_batch = atk(x, y)
        y = torch.tensor(y)
        return adv_data

    def __len__(self):
        return len(self.dataset)

adv_dataset = AugMixDataset(dataset, transforms.ToTensor())
img, label = dataset[0]

def image_folder_custom_label(root, transform, idx2label) :
    # custom_label
    # type : List
    # index -> label
    # ex) ['tench', 'goldfish', 'great_white_shark', 'tiger_shark']
    
    old_data = dsets.ImageFolder(root=root, transform=transform)
    old_classes = old_data.classes
    
    label2idx = {}
    
    for i, item in enumerate(idx2label) :
        label2idx[item] = i
    
    new_data = dsets.ImageFolder(root=root, transform=transform, 
                                target_transform=lambda x : idx2label.index(old_classes[x]))
    new_data.classes = idx2label
    new_data.class_to_idx = label2idx

    return new_data

def load_labels_from_txt(file_path):
    """Loads class labels from a .txt file, where each line contains a label."""
    with open(file_path, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

def get_imagenet_data():
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # Load ImageNet class labels from a .txt file instead of JSON
    idx2label = load_labels_from_txt(parent_directory + "/ImageNet-100-Pytorch/IN100.txt")

    # Define transformation pipeline
    transform = transforms.Compose([
        transforms.Resize((224, 244)),
        transforms.ToTensor()
        # transforms.Normalize(mean=MEAN, std=STD)
    ])

    # Load dataset with custom labels
    imagnet_data = image_folder_custom_label(root= parent_directory + '/train', 
                                            transform=transform,
                                            idx2label=idx2label)

    # Create DataLoader
    data_loader = torch.utils.data.DataLoader(imagnet_data, batch_size=256, shuffle=False)

    print("Used normalization: mean=", MEAN, "std=", STD)
    
    # Return the first batch
    return next(iter(data_loader))
    # return data_loader

images, labels = get_imagenet_data()
# images, labels = next(iter(train_loader))
adv_images =  atk(images, labels)
img = adv_images[0]
img = img.cpu().detach()
img = img.numpy()
# img =  np.clip(img.numpy(),0,1)

plt.figure(figsize=(2,2))
plt.imshow(img.transpose(1,2,0))

def imshow(img, title):
    img = torchvision.utils.make_grid(img.cpu().data, normalize=True)
    npimg = img.numpy()
    fig = plt.figure(figsize = (2, 2))
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.title(title)
    plt.show()

imshow(adv_images[0], title='vis of image')

