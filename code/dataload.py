import numpy as np # random seed

# Preprocessing
import torch
import torchvision.datasets as dset  
import torchvision.transforms as transforms 
from torch.utils.data import DataLoader, sampler, random_split, Dataset
from PIL import Image


class NewDataset(Dataset):
    def __init__(self, data, targets, name, transforms=None):
        self.data = np.array(data)
        self.targets = torch.LongTensor(targets)
        self.transform = transforms
        if name == 'cifar10':
            self.class_to_idx = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
        elif name == 'mnist':
            self.class_to_idx = {'0 - zero': 0, '1 - one': 1, '2 - two': 2, '3 - three': 3, '4 - four': 4, '5 - five': 5, '6 - six': 6, '7 - seven': 7, '8 - eight': 8, '9 - nine': 9}
        
    def __len__(self):
        return len(self.data)
            
    def __getitem__(self, idx):
        
        image = Image.fromarray(self.data[idx])

        if self.transform:
            image = self.transform(image)

        label = self.targets[idx]

        return image, label


def mnist_load(batch_size=128, validation_rate=0.2, shuffle=True, random_seed=223):
    # Transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))
    ])

    # Data Load
    mnist_data = dset.MNIST(root='../dataset/MNIST',
                            train=True,
                            transform=transform,
                            download=True)
    mnist_test = dset.MNIST(root='../dataset/MNIST',
                            train=False,
                            transform=transform,
                            download=True)

    x_data = mnist_data.data
    y_data = np.array(mnist_data.targets)

    # Split train to train and validation
    total_size = x_data.shape[0]
    indices = np.arange(total_size)
    split = int(validation_rate * total_size)

    np.random.seed(random_seed)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    x_train = x_data[train_idx]
    y_train = y_data[train_idx]
    x_valid = x_data[valid_idx]
    y_valid = y_data[valid_idx]

    # Make Dataset
    train_dataset = NewDataset(x_train, y_train, 'mnist', transform)
    valid_dataset = NewDataset(x_valid, y_valid, 'mnist', transform)
    test_dataset = NewDataset(mnist_test.data, mnist_test.targets, 'mnist', transform)
    
    # Data loader
    trainloader = DataLoader(dataset=train_dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    
    validloader = DataLoader(dataset=valid_dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)

    testloader = DataLoader(dataset=test_dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=0)
    print('Data Complete')

    return trainloader, validloader, testloader


def cifar10_load(batch_size=128, validation_rate=0.2, shuffle=True, random_seed=223, augmentation=True):
    # Transforms
    if augmentation:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Data Load
    cifar10_data = dset.CIFAR10(root='../dataset/CIFAR10',
                                train=True,
                                download=True)
    cifar10_test = dset.CIFAR10(root='../dataset/CIFAR10',
                                train=False,
                                download=True)

    x_data = cifar10_data.data
    y_data = np.array(cifar10_data.targets)

    # Split train to train and validation
    total_size = x_data.shape[0]
    indices = np.arange(total_size)
    split = int(validation_rate * total_size)

    np.random.seed(random_seed)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    x_train = x_data[train_idx]
    y_train = y_data[train_idx]
    x_valid = x_data[valid_idx]
    y_valid = y_data[valid_idx]

    # Make Dataset
    train_dataset = NewDataset(x_train, y_train, 'cifar10', transform_train)
    valid_dataset = NewDataset(x_valid, y_valid, 'cifar10', transform_test)
    test_dataset = NewDataset(cifar10_test.data, cifar10_test.targets, 'cifar10', transform_test)

    # Data loader
    trainloader = DataLoader(dataset=train_dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    
    validloader = DataLoader(dataset=valid_dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)

    testloader = DataLoader(dataset=test_dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=0)
    print('Data Complete')                            

    return trainloader, validloader, testloader