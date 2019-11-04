import numpy as np # random seed

# Preprocessing
from torch.utils.data import DataLoader, sampler, random_split
import torchvision.datasets as dset  
import torchvision.transforms as transforms 


def mnist_load(batch_size, validation_rate, shuffle=True, random_seed=223):
    # Transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))
    ])

    # Data Load
    mnist_train = dset.MNIST(root='../dataset/MNIST',
                            train=True,
                            transform=transform,
                            download=True)
    mnist_test = dset.MNIST(root='../dataset/MNIST',
                            train=False,
                            transform=transform,
                            download=True)

    # Split train to train and validation
    total_size = len(mnist_train)
    valid_size = int(total_size * validation_rate)
    train_size = total_size - valid_size

    np.random.seed(random_seed)
    mnist_train, mnist_valid = random_split(mnist_train, [train_size, valid_size])
    
    # Data loader
    trainloader = DataLoader(dataset=mnist_train,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    
    validloader = DataLoader(dataset=mnist_valid,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)

    testloader = DataLoader(dataset=mnist_test,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=0)
    print('Data Complete')

    return trainloader, validloader, testloader


def cifar10_load(batch_size, validation_rate, shuffle=True, random_seed=223):
    # Transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])


    # Data Load
    cifar10_train = dset.CIFAR10(root='../dataset/CIFAR10',
                               train=True,
                               transform=transform_train,
                               download=True)
    cifar10_valid = dset.CIFAR10(root='../dataset/CIFAR10',
                               train=True,
                               transform=transform_test,
                               download=True)
    cifar10_test = dset.CIFAR10(root='../dataset/CIFAR10',
                              train=False,
                              transform=transform_test,
                              download=True)

    # Split train to train and validation
    total_size = len(cifar10_train)
    indices = list(range(total_size))
    split = int(validation_rate * total_size)

    np.random.seed(random_seed)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = sampler.SubsetRandomSampler(train_idx)
    valid_sampler = sampler.SubsetRandomSampler(valid_idx)
    
    # Data loader
    trainloader = DataLoader(dataset=cifar10_train,
                             batch_size=batch_size,
                             sampler=train_sampler,
                             num_workers=0)
    
    validloader = DataLoader(dataset=cifar10_valid,
                             batch_size=batch_size,
                             sampler=valid_sampler,
                             num_workers=0)

    testloader = DataLoader(dataset=cifar10_test,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=0)
    print('Data Complete')                            

    return trainloader, validloader, testloader