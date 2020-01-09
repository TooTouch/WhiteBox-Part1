import torch

import numpy as np
import pandas as pd 
import random
import os
import time
import datetime 
import cv2
from PIL import Image

import h5py
from tqdm import tqdm

from dataload import mnist_load, cifar10_load
from models import SimpleCNN, RAN, WideResNetAttention

def seed_everything(seed=223):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class ModelTrain:
    '''
    Model Training and Validation
    '''
    def __init__(self, model, data, epochs, criterion, optimizer, device, model_name=None, savedir=None, monitor=None, mode=None, validation=None, verbose=0):
        '''
        args:
        - model: training model
        - data: train dataset
        - epochs: epochs
        - criterion: critetion
        - optimizer: optimizer
        - device: device to use [cuda:i, cpu], i=0, ...,n
        - model_name: name of model to save
        - savedir: directory to save
        - monitor: metric name or loss to check [default: None]
        - mode: [min, max] [default: None]
        - validation: test set to evaluate in train [default: None]
        - verbose: number of score print
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
            if savedir:
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
            loss = self.criterion(outputs, targets.long())
        
            # Update
            loss.backward()
            self.optimizer.step()

            # Loss
            train_loss += loss.item() / len(self.train_set)

            # Accuracy
            _, predicted = outputs.max(1)
            train_acc += predicted.eq(targets.long()).sum()
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
                loss = self.criterion(outputs, targets.long())
                val_loss += loss.item() / len(self.validation_set)

                # Accuracy
                _, predicted = outputs.max(1)
                val_acc += predicted.eq(targets.long()).sum()
                total_size += targets.size(0)
            
            val_acc = val_acc.item() / total_size
            
        return val_acc, val_loss



class ModelTest:
    '''
    Model Test
    '''
    def __init__(self, model, data, model_name, loaddir, device):
        '''
        args:
        - model: training model
        - data: train dataset
        - model_name: name of model to save
        - loaddir: directory to load
        - device: device to use [cuda:i, cpu], i=0, ...,n
        '''

        self.model = model 
        self.test_set = data
        self.device = device

        load_file = torch.load(f'{loaddir}/{model_name}.pth')
        self.model.load_state_dict(load_file['model'])

        print('=====Test Mode======')
        print('Model load complete')
        del load_file['model']
        for k, v in load_file.items():
            print(f'{k}: {v}')

        start = time.time()
        self.results = self.test()
        end = time.time() - start
        test_time = datetime.timedelta(seconds=end)
        print('Test Acc: {0:.4%} | Time: {1:}'.format(self.results, test_time))

        

    def test(self):
        self.model.eval()
        test_acc = 0

        with torch.no_grad():
            for inputs, targets in self.test_set:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs).detach() 
                # Accuracy
                _, predicted = outputs.max(1)
                test_acc += predicted.eq(targets).sum().item()
            
            test_acc = test_acc / len(self.test_set.dataset)

        return test_acc
        


class CheckPoint:
    def __init__(self, dirname, model_name, monitor, mode):
        '''
        args:
        - dirname: save directory
        - model_name: model name
        - monitor: metric name or loss
        - mode: [min, max]
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
        args:
        - epoch: current epoch
        - model: training model
        - score: current score
        '''

        if self.mode == 'min':
            if score < self.best:
                self.model_save(epoch, model, score)
        elif self.mode == 'max':
            if score > self.best:
                self.model_save(epoch, model, score)

        
    def model_save(self, epoch, model, score):
        '''
        args:
        - epoch: current epoch
        - model: training model
        - score: current score
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
    '''
    Model early stopping
    '''
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


def get_samples(target, nb_class=10, sample_index=0, attention=None, device='cpu'):
    '''
    Get samples : original images, preprocessed images, target class, trained model

    args:
    - target: [mnist, cifar10]
    - nb_class: number of classes
    - example_index: index of image by class

    return:
    - original_images (numpy array): Original images, shape = (number of class, W, H, C)
    - pre_images (torch array): Preprocessing images, shape = (number of class, C, W, H)
    - target_classes (dictionary): keys = class index, values = class name
    - model (pytorch model): pretrained model
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
    filename = f'simple_cnn_{target}'
    if attention in ['CAM','CBAM']:
        filename += f'_{attention}'
    elif attention in ['RAN','WARN']:
        filename = f'{target}_{attention}'
    print('filename: ',filename)
    weights = torch.load(f'../checkpoint/{filename}.pth')

    if attention == 'RAN':
        model = RAN(target).to(device)
    elif attention == 'WARN':
        model = WideResNetAttention(target).to(device)
    else:
        model = SimpleCNN(target, attention).to(device)
    model.load_state_dict(weights['model'])

    # image preprocessing
    pre_images = torch.zeros(original_images.shape)
    pre_images = np.transpose(pre_images, (0,3,1,2))
    for i in range(len(original_images)):
        pre_images[i] = testset.transform(original_images[i])
    
    return original_images, original_targets, pre_images, target_classes, model


def rescale_image(images, channel=True):
    '''
    MinMax scaling

    args:
    - images : images (batch size, C, H, W) if channel is True else (batch size, H, W)
    - channel : channel status (boolean)
    return:
    - images : rescaled images (batch_size, H, W, C)
    '''

    if channel:
        mins = np.min(images, axis=(1,2,3)) # (batch size, )
        mins = mins.reshape(mins.shape + (1,1,1,)) # (batch size, 1, 1, 1)
        maxs = np.max(images, axis=(1,2,3)) # (batch size, )
        maxs = maxs.reshape(maxs.shape + (1,1,1,)) # (batch size, 1, 1, 1)

        images = (images - mins)/(maxs - mins)
        images = images.transpose(0,2,3,1) # (batch size, H, W, C)

    else:
        mins = images.min(axis=(1,2)) # (batch size, )
        mins = mins.reshape(mins.shape + (1,1,)) # (batch size, 1, 1)
        maxs = images.max(axis=(1,2)) # (batch size, C)
        maxs = maxs.reshape(maxs.shape + (1,1,)) # (batch size, 1, 1)

        images = (images - mins)/(maxs - mins)
        images = np.uint8(images * 255) # (batch size, H, W)

    return images

def resize_image(image, origin_image, color):
    '''
    Resize input image to original image size

    args:
    - image: image. (H, W) 

    return:
    - img : resized image
    '''
    img = np.uint8(Image.fromarray(image).resize((origin_image.shape[-2:]), Image.ANTIALIAS))/255
    if color:
        # images (H,W) to (H,W,C)
        img = cv2.applyColorMap(np.uint8(img*255), cv2.COLORMAP_JET)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = np.expand_dims(img, axis=2)
    
    return img

def save_saliency_map(attr_method, dataloader, save_dir, name, **kwargs):
    '''
    Save saliency map extracted by attribution method

    args:
    - attr_method: attribution method
    - dataloader: dataset loader
    - name: tqdm description
    - save_dir: directory to save hdf5 file
    '''
    # config
    img_size = dataloader.dataset.data.shape[1:] 
    dim = len(img_size)
    if dim == 2:
        img_size = img_size + (1,)

    # initialize
    sal_maps = np.array([], dtype=np.float32).reshape((0,) + img_size)
    probs = np.array([], dtype=np.float32)
    preds = np.array([], dtype=np.uint8)

    # make saliency maps
    for img_b, target_b in tqdm(dataloader, desc=name):
        sal_map_b, prob_b, pred_b = attr_method(img_b, target_b, **kwargs)
        sal_maps = np.vstack([sal_maps, sal_map_b])
        probs = np.append(probs, prob_b)
        preds = np.append(preds, pred_b)

    # save saliency map to h5py file
    with h5py.File(save_dir, 'w') as hf:
        hf.create_dataset('saliencys',data=sal_maps)
        hf.create_dataset('probs',data=probs)
        hf.create_dataset('preds',data=preds)
        hf.close()
    print('Save saliency maps')

def calc_accuracy(model, dataset, device='cpu'):
    '''
    Calculate accuracy 

    args:
    - model: trained model
    - dataset: dataset to evaluate
    - idx2class: index and class dictionary
    - device: device to use [cuda:i, cpu], i=0, ...,n 

    return:
    - total_acc: total accuracy
    - acc_indice: accuracy by class
    '''
    # ture type must be numpy array
    true = np.array(dataset.dataset.targets)
    indices_by_idx = dict((idx, np.where(true==idx)) for idx in range(10))
    
    # test
    model.to(device)
    model.eval()
    pred_lst = []
    with torch.no_grad():
        for inputs, targets in dataset:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).detach()
            
            _, predicted = outputs.max(1)
            pred_lst.extend(predicted.cpu().numpy())
    
    pred = np.array(pred_lst)
    
    acc_indice = []
    for idx in range(10):
        indices = indices_by_idx[idx]
        true_idx = true[indices]
        pred_idx = pred[indices]
        correct = np.sum(true_idx == pred_idx)
        acc_idx = correct / true_idx.size
        acc_indice.append(acc_idx)
        
    total_acc = np.sum(true==pred) / true.shape[0]
    
    return total_acc, acc_indice


def count_params(model):
    '''
    Count nubmer of model parameters

    args:
    - model: model to count parameters

    return:
    - nb_params: number of model parameters
    '''
    nb_params = sum(np.prod(p.size()) for p in model.parameters())
    return nb_params

def acc_concat(acc_lst):
    '''
    concatenate total accuracy and accuracy by class

    args:
    - acc_lst: accuracy list. [total_accuracy, accuracy_by_class]
    
    return:
    - acc_lst: total accuracy list
    '''
    acc_lst[1].append(acc_lst[0])
    return acc_lst[1]

def compare_model_acc(model_lst, dataloader, model_names, device='cpu'):
    '''
    Comparison model accuracy

    args: 
    - model_lst: model list to compare
    - dataloader: data to evaluate
    - model_names: model names
    - device: device to use [cuda:i, cpu], i=0, ...,n
    
    return:
    - acc_df: dataframe accuracy of models. (number of model, number of class)
    '''
    assert len(model_lst) > 1, 'Model list must have at least two models'
    assert len(model_lst) == len(model_names), 'Model list must be the same length as the brist'

    acc_lst = []
    for model in model_lst:
        tacc, cnn_accs = calc_accuracy(model=model, dataset=dataloader, device=device)
        acc_lst.append([tacc, cnn_accs])

    acc_lst = list(map(acc_concat, acc_lst))

    cols = list(dataloader.dataset.class_to_idx.keys())
    cols.append('Total')
    acc_df = pd.DataFrame(acc_lst, columns=cols, index=model_names).round(3)

    return acc_df

