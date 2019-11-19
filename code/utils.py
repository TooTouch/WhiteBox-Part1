import torch

import random
import os
import numpy as np
import time
import datetime 

from dataload import mnist_load, cifar10_load
from model import SimpleCNN

import matplotlib.pyplot as plt

def seed_everything(seed=223):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class ModelTrain:
    def __init__(self, model, data, epochs, criterion, optimizer, device, model_name, savedir, monitor=None, mode=None, validation=None, verbose=0):
        '''
        params

        model: training model
        data: train dataset
        epochs: epochs
        criterion: critetion
        optimizer: optimizer
        device: [cuda:i, cpu], i=0, ...,n
        model_name: name of model to save
        savedir: directory to save
        monitor: metric name or loss to check [default: None]
        mode: [min, max] [default: None]
        validation: test set to evaluate in train [default: None]
        verbose: number of score print
        '''

        self.model = model
        self.train_set = data
        self.validation_set = validation
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.history = {}

        print('=====Train Mode======')
        ckp = CheckPoint(dirname=savedir,
                         model_name=model_name,
                         monitor=monitor,
                         mode=mode)
        # es = EarlyStopping(patience=20,
        #                    factor=0.01)

        # Training time check
        total_start = time.time()

        # Initialize history list
        train_acc_lst, train_loss_lst = [], []
        val_acc_lst, val_loss_lst = [], []
        epoch_time_lst = []

        for i in range(epochs):

            epoch_start = time.time()
            train_acc, train_loss = self.train()
            val_acc, val_loss = self.validation()        

            # Score check
            if i % verbose == 0:
                end = time.time()
                epoch_time = datetime.timedelta(seconds=end - epoch_start)
                print('\n[{0:}/{1:}] Train - Acc: {2:.4%}, Loss: {3:.5f} | Val - Acc: {4:.5%}, Loss: {5:.6f} | Time: {6:}\n'.format(i+1, epochs, train_acc, train_loss, val_acc, val_loss, epoch_time))

            # Model check
            ckp.check(epoch=i+1, model=self.model, score=val_acc)
            # es.check(val_loss)

            # Save history
            train_acc_lst.append(train_acc)
            train_loss_lst.append(train_loss)

            val_acc_lst.append(val_acc)
            val_loss_lst.append(val_loss)

            epoch_time_lst.append(str(epoch_time))

            # if es.nb_patience == es.patience:
            #     break

        end = time.time() - total_start
        total_time = datetime.timedelta(seconds=end)
        print('\nFinish Train: Training Time: {}\n'.format(total_time))

        # Make history 
        self.history = {}
        self.history['train'] = []
        self.history['train'].append({
            'acc':train_acc_lst,
            'loss':train_loss_lst,
        })
        self.history['validation'] = []
        self.history['validation'].append({
            'acc':val_acc_lst,
            'loss':val_loss_lst,
        })
        self.history['time'] = []
        self.history['time'].append({
            'epoch':epoch_time_lst,
            'total':str(total_time)
        })

    def train(self):
        self.model.train()
        train_acc = 0
        train_loss = 0
        total_size = 0
        
        for inputs, targets in self.train_set:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
        
            # Update
            loss.backward()
            self.optimizer.step()

            # Loss
            train_loss += loss.item() / len(self.train_set)

            # Accuracy
            _, predicted = outputs.max(1)
            train_acc += predicted.eq(targets).sum()
            total_size += targets.size(0)

        train_acc = train_acc.item() / total_size

        return train_acc, train_loss

    def validation(self):
        self.model.eval()
        val_acc = 0
        val_loss = 0
        total_size = 0

        with torch.no_grad():
            for inputs, targets in self.validation_set:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs).detach() 

                # Loss
                loss = self.criterion(outputs, targets)
                val_loss += loss.item() / len(self.validation_set)

                # Accuracy
                _, predicted = outputs.max(1)
                val_acc += predicted.eq(targets).sum()
                total_size += targets.size(0)
            
            val_acc = val_acc.item() / total_size
            
        return val_acc, val_loss



class ModelTest:
    def __init__(self, model, data, model_name, loaddir, device):
        '''
        params
        
        model: training model
        data: train dataset
        model_name: name of model to save
        loaddir: directory to load
        device: [cuda:i, cpu], i=0, ...,n
        '''

        self.model = model 
        self.data = data
        self.device = device

        load_file = torch.load(f'{loaddir}/{model_name}.pth')
        self.model.load_state_dict(load_file['model'])

        print('=====Test Mode======')
        print('Model load complete')
        del load_file['model']
        for k, v in load_file.items():
            print(f'{k}: {v}')

        start = time.time()
        test_acc = self.test()
        end = time.time() - start
        test_time = datetime.timedelta(seconds=end)
        print('Test Acc: {0:.4%} | Time: {1:}'.format(test_acc, test_time))


    def test(self):
        self.model.eval()
        test_acc = 0

        with torch.no_grad():
            for inputs, targets in self.data:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs).detach() 
                # Accuracy
                _, predicted = outputs.max(1)
                test_acc += predicted.eq(targets).sum().item()
            
            test_acc = test_acc / len(self.data.dataset)

        return test_acc
        


class CheckPoint:
    def __init__(self, dirname, model_name, monitor, mode):
        '''
        params
        dirname: save directory
        model_name: model name
        monitor: metric name or loss
        mode: [min, max]
        '''

        self.best = 0
        self.model_name = model_name
        self.save_dir = dirname
        self.monitor = monitor
        self.mode = mode
        
        # if save directory is not there, make it
        if not(os.path.isdir(self.save_dir)):
            os.mkdir(self.save_dir)

    def check(self, epoch, model, score):
        '''
        params
        epoch: current epoch
        model: training model
        score: current score
        '''

        if self.mode == 'min':
            if score < self.best:
                self.model_save(epoch, model, score)
        elif self.mode == 'max':
            if score > self.best:
                self.model_save(epoch, model, score)

        
    def model_save(self, epoch, model, score):
        '''
        params
        epoch: current epoch
        model: training model
        score: current score
        '''

        print('Save complete, epoch: {0:}: Best {1:} has changed from {2:.5f} to {3:.5f}'.format(epoch, self.monitor, self.best, score))
        self.best = score
        state = {
            'model':model.state_dict(),
            f'best_{self.monitor}':score,
            'best_epoch':epoch
        }
        torch.save(state, f'{self.save_dir}/{self.model_name}.pth')



class EarlyStopping:
    def __init__(self, patience, factor=0.001):
        self.patience = patience
        self.factor = factor
        self.best_loss = 0
        self.nb_patience = 0

    def check(self, loss):
        if self.best_loss < loss + self.factor:
            # Initialize best loss
            if self.best_loss == 0:
                self.best_loss = loss
            else:
                self.nb_patience += 1
                print(f'Current patience is [{self.nb_patience}/{self.patience}]: ', end='')
                print('Best loss: {0:.5f} | Current loss + {1:}: {2:.5f}'.format(self.best_loss, self.factor, loss + self.factor))

        elif self.best_loss > loss:
            self.best_loss = loss


def get_samples(target, nb_class=10, sample_index=0):
    '''
    params:
        target : [mnist, cifar10]
        nb_class : number of classes
        example_index : index of image by class

    returns:
        original_images (numpy array) : Original images, shape = (number of class, W, H, C)
        pre_images (torch array) : Preprocessing images, shape = (number of class, C, W, H)
        target_classes (dictionary) : keys = class index, values = class name
        model (pytorch model) : pretrained model
    '''

    if target == 'mnist':
        image_size = (28,28,1)
        
        _, _, testloader = mnist_load()
        testset = testloader.dataset

    elif target == 'cifar10':
        image_size = (32,32,3)

        _, _, testloader = cifar10_load()
        testset = testloader.dataset

    # idx2class
    target_class2idx = testset.class_to_idx
    target_classes = dict(zip(list(target_class2idx.values()), list(target_class2idx.keys())))

    # select images
    idx_by_class = [np.where(np.array(testset.targets)==i)[0][sample_index] for i in range(nb_class)]
    original_images = testset.data[idx_by_class]
    if not isinstance(original_images, np.ndarray):
        original_images = original_images.numpy()
    original_images = original_images.reshape((nb_class,)+image_size)
    # select targets
    if isinstance(testset.targets, list):
        original_targets = torch.LongTensor(testset.targets)[idx_by_class]
    else:
        original_targets = testset.targets[idx_by_class]

    # model load
    weights = torch.load('../checkpoint/simple_cnn_{}.pth'.format(target))
    model = SimpleCNN(target)
    model.load_state_dict(weights['model'])

    # image preprocessing
    pre_images = torch.zeros(original_images.shape)
    pre_images = np.transpose(pre_images, (0,3,1,2))
    for i in range(len(original_images)):
        pre_images[i] = testset.transform(original_images[i])
    
    return original_images, original_targets, pre_images, target_classes, model


def rescale_image(images):
    '''
    MinMax scaling

    Args:
        images : images (batch_size, C, H, W)
    '''
    mins = np.min(images, axis=(1,2,3)) # (batch_size, 1)
    mins = mins.reshape(mins.shape + (1,1,1,)) # (batch_size, 1, 1, 1)
    maxs = np.max(images, axis=(1,2,3))
    maxs = maxs.reshape(maxs.shape + (1,1,1,))

    images = (images - mins)/(maxs - mins)
    images = images.transpose(0,2,3,1)

    return images


def visualize_saliencys(origin_imgs, results, probs, preds, classes, names, target, **kwargs):
    # initialize
    row = kwargs['row']
    col = kwargs['col']
    size = kwargs['size']
    fontsize = kwargs['fontsize']
    labelsize = kwargs['labelsize']
    
    if target=='mnist':
        origin_imgs= origin_imgs.squeeze()
        for i in range(len(results)):
            results[i] = results[i].squeeze()
        color = 'gray'
    else:
        color = None
            
    f, ax = plt.subplots(row, col, figsize=size)
    # original images
    for i in range(row):
        ax[i,0].imshow(origin_imgs[i], color)
        ax[i,0].set_ylabel('True: {0:}\nPred: {1:} ({2:.2%})'.format(classes[i], int(preds[i]), probs[i]), size=labelsize)
        ax[i,0].set_xticks([])
        ax[i,0].set_yticks([])
        # set title
        if i == 0:
            ax[i,0].set_title('Original Image', size=fontsize)

    for i in range(row*(col-1)):
        r = i//(col-1)
        c = i%(col-1)
        ax[r,c+1].imshow(results[c][r], color)
        ax[r,c+1].axis('off')
        # set title
        if r == 0:
            ax[r,c+1].set_title(names[c], size=fontsize)

    plt.subplots_adjust(wspace=-0.5, hspace=0)
    plt.tight_layout()