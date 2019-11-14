import numpy as np # array
import json # history
import os # directory

# pytorch 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F 
import torchvision.models as models

# defined function
from saliency.evaluation_methods import selecticity_evaluation
from dataload import mnist_load, cifar10_load
from model import SimpleCNN
from utils import seed_everything, ModelTrain, ModelTest, make_saliency_map

# arguments
import argparse

def main(args):
    # Config
    epochs = args.epochs
    batch_size = args.batch_size # mnist 128, cifar10 128
    valid_rate = args.valid_rate
    lr = args.lr # mnist 0.01, cifar10 0.01
    verbose = args.verbose

    target = args.target
    monitor = args.monitor
    mode = args.mode

    model_name = 'simple_cnn_{}'.format(target) # simple_cnn_{}
    savedir = '../checkpoint'
    logdir = '../logs'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('=====Setting=====')
    print('Epochs: ',epochs)
    print('Batch Size: ',batch_size)
    print('Validation Rate: ',valid_rate)
    print('Learning Rate: ',lr)
    print('Target: ',target)
    print('Monitor: ',monitor)
    print('Model Name: ',model_name)
    print('Mode: ',mode)
    print('Save Directory: ',savedir)
    print('Log Directory: ',logdir)
    print('Device: ',device)
    print('Verbose: ',verbose)
    print()
    print('Setting Random Seed')
    print()
    seed_everything() # seed setting
    
    # Data Load
    print('=====Data Load=====')
    if target == 'mnist':
        trainloader, validloader, testloader = mnist_load(batch_size=batch_size,
                                                          validation_rate=valid_rate,
                                                          shuffle=True)
                                    
    elif target == 'cifar10':
        trainloader, validloader, testloader = cifar10_load(batch_size=batch_size,
                                                            validation_rate=valid_rate,
                                                            shuffle=True)
    
    print('=====Model Load=====')
    # Load model
    net = SimpleCNN(target).to(device)
    print()


    # Model compile
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005) # MNIST
    criterion = nn.CrossEntropyLoss()

    # Train
    modeltrain = ModelTrain(model=net,
                            data=trainloader,
                            epochs=epochs,
                            criterion=criterion,
                            optimizer=optimizer,
                            device=device,
                            model_name=model_name,
                            savedir=savedir,
                            monitor=monitor,
                            mode=mode,
                            validation=validloader,
                            verbose=verbose)
    # Test
    ModelTest(model=net,
              data=testloader,
              loaddir=savedir,
              model_name=model_name,
              device=device)

    
    # History save as json file
    if not(os.path.isdir(logdir)):
        os.mkdir(logdir)
    with open(f'{logdir}/{model_name}_logs.txt','w') as outfile:
        json.dump(modeltrain.history, outfile)


    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # select mode
    parser.add_argument('--train', action='store_true', help='if train is true, train model')
    parser.add_argument('--eval', action='store_true', help='if eval is true, evaluate model')
    # train parameters
    parser.add_argument('--target', type=str, choices=['mnist','cifar10'], help='target data')
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='number of batch')
    parser.add_argument('--valid_rate', type=float, default=0.2, help='validation set ratio')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--verbose', type=bool, default=1, choices=range(1,11), help='model evalutation display period')
    parser.add_argument('--monitor', type=str, default='acc',choices=['acc','loss'], help='monitor value')
    parser.add_argument('--mode', type=str, default='max', choices=['max','min'], help='min or max')
    # make saliency map
    parser.add_argument('--make_saliency', action='store_true', help='if make saliency is true, make saliency map.')
    # attribution method
    parser.add_argument('--method', type=str, default=None, choices=['VBP','IB','IG','GB','GC','GB-GC','DeconvNet',None], help='select attribution method')
    # selectivity
    parser.add_argument('--steps', type=int, default=50, help='number of evaluation')
    parser.add_argument('--sample_pct', type=float, default=0.1, help='sample ratio')
    args = parser.parse_args()

    # TODO : Tensorboard Check
    if args.train:
        # python main.py --train --target=['mnist','cifar10']
        main(args=args)
    
    if args.make_saliency:
        # python main.py --make_saliency
        make_saliency_map()

    if args.eval:
        # python main.py --eval --target=['mnist','cifar10']
        selecticity_evaluation(args)


