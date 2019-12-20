import numpy as np # array
import json # history
import os # directory
import h5py # save files

# pytorch 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F 
import torchvision.models as models

# defined function
from saliency.evaluation_methods import Selectivity, adjust_image
from dataload import mnist_load, cifar10_load
from model import SimpleCNN
from utils import seed_everything, ModelTrain, ModelTest

# arguments
import argparse

def main(args, **kwargs):
    # Config
    epochs = args.epochs
    batch_size = args.batch_size 
    valid_rate = args.valid_rate
    lr = args.lr 
    verbose = args.verbose

    # checkpoint
    target = args.target
    monitor = args.monitor
    mode = args.mode

    # save name
    model_name = 'simple_cnn_{}'.format(target)
    if args.attention:
        model_name = f'{model_name}_{args.attention}'

    # save directory
    savedir = '../checkpoint'
    logdir = '../logs'

    # device setting cpu or cuda(gpu)
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

    # ROAR or KAR
    if (args.eval=='ROAR') or (args.eval=='KAR'):
        # saliency map load
        hf = h5py.File(f'../saliency_maps/[{args.target}]{args.method}_train.hdf5','r')
        sal_maps = np.array(hf['saliencys'])
        # adjust image 
        data_lst = adjust_image(kwargs['ratio'], trainloader, sal_maps, args.eval)
        # hdf5 close
        hf.close()                     
        # model name
        model_name = model_name + '_{0:}_{1:}{2:.1f}'.format(args.method, args.eval, kwargs['ratio'])
    
    print('=====Model Load=====')
    # Load model
    net = SimpleCNN(target, args.attention).to(device)
    print()

    # Model compile
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005) 
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
    modeltest = ModelTest(model=net,
                          data=testloader,
                          loaddir=savedir,
                          model_name=model_name,
                          device=device)
 
    modeltrain.history['test_result'] = modeltest.results

    # History save as json file
    if not(os.path.isdir(logdir)):
        os.mkdir(logdir)
    with open(f'{logdir}/{model_name}_logs.txt','w') as outfile:
        json.dump(modeltrain.history, outfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='Train or Evaluation')

    # Train
    parser.add_argument('--train', action='store_true', help='train mode')
    parser.add_argument('--target', type=str, choices=['mnist','cifar10'], help='target data')
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='number of batch')
    parser.add_argument('--valid_rate', type=float, default=0.2, help='validation set ratio')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--verbose', type=bool, default=1, choices=range(1,11), help='model evalutation display period')
    parser.add_argument('--monitor', type=str, default='acc',choices=['acc','loss'], help='monitor value')
    parser.add_argument('--mode', type=str, default='max', choices=['max','min'], help='min or max')
    parser.add_argument('--attention', type=str, default=None, choices=['CBAM','CAM'], help='choice attention method')
    
    # Evaluation
    parser.add_argument('--eval', default=None, type=str, choices=['coherence','selectivity','ROAR','KAR'], help='select evaluate methods')
    parser.add_argument('--method', type=str, default=None, choices=['VBP','IB','IG','GB','GC','GBGC','DeconvNet'], help='select attribution method')
    parser.add_argument('--steps', type=int, default=50, help='number of evaluation')
    parser.add_argument('--ratio', type=float, default=0.1, help='ratio whatever')
    args = parser.parse_args()

    # TODO: Tensorboard Check

    # python main.py --train --target=['mnist','cifar10'] --attention=['CBAM','CAM']
    if args.train:
        main(args=args)
        
    elif args.eval=='selectivity':
        # make evalutation directory
        if not os.path.isdir('../evaluation'):
            os.mkdir('../evaluation')

        # pretrained model load
        weights = torch.load('../checkpoint/simple_cnn_{}.pth'.format(args.target))
        model = SimpleCNN(args.target)
        model.load_state_dict(weights['model'])

        # selectivity evaluation
        selectivity_method = Selectivity(model=model, 
                                         target=args.target, 
                                         batch_size=args.batch_size,
                                         method=args.method, 
                                         sample_pct=args.ratio)
        # evaluation
        selectivity_method.eval(steps=args.steps, savedir='../evaluation')

    elif (args.eval=='ROAR') or (args.eval=='KAR'):
        # ratio
        ratio_lst = np.arange(0, 1, args.ratio)[1:] # exclude zero
        for ratio in ratio_lst:
            main(args=args, ratio=ratio)
        
    
        


