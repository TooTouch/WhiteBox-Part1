import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model import SimpleCNN
from dataload import mnist_load, cifar10_load
from saliency.attribution_methods import *
from saliency.ensembles import *
from utils import seed_everything, get_samples

import cv2
import os
import h5py
import numpy as np
from tqdm import tqdm


# TODO : Add ensemble methods 
class Selectivity(object):
    def __init__(self, model, target, batch_size, method, ensemble=None, sample_pct=0.1, deconv_model=None, nb_class=10, sample_index=0):
        '''
        Args
            model : pretrained model
            target : [mnist, cifar10]
            method : {
                VBP : Vanilla Backpropagation,
                IB : Input x Backpropagation,
                IG : Integrated Gradients,
                GB : Guided Backpropagation,
                GC : Grad CAM,
                GB-GC : Guided GradCAM,
                DeconvNet : DeconvNet,
            }
            ensemble : {
                SG : SmoothGrad,
                SG-SQ : SmoothGrad Square
                SG-VAR : SmoothGrad VAR
            }
            deconv_model : Deconvolution model. Only used if method is set to DeconvNet.
            nb_class : number of class
            sample_index : sample image index by class
        '''

        # data load
        if target == 'mnist':
            _, _, testloader = mnist_load()
        elif target == 'cifar10':
            _, _, testloader = cifar10_load()
        self.target = target
        self.testset = testloader.dataset 
        self.img_size = self.testset.data.shape[1:] # mnist : (28,28), cifar10 : (32,32,3)
        self.batch_size = batch_size

        # sampling
        seed_everything()
        sample_size = int(len(self.testset) * sample_pct)
        sample_idx = np.random.choice(len(self.testset), sample_size, replace=False)
        self.testset.data = self.testset.data[sample_idx]
        self.testset.targets = np.array(self.testset.targets)[sample_idx]
        self.sample_pct = sample_pct
        self.data_size = len(self.testset)

        # model setting
        self.model = model
        self.model.eval()
        self.deconv_model = deconv_model

        # saliency map
        self.method = method
        self.ensemble = ensemble
        self.saliency_map, self.layer, self.color = self.saliency_map_choice()

        # sample
        self.nb_class = nb_class
        self.nb_checkpoint = 5 
        self.idx_by_class = [np.where(np.array(self.testset.targets)==i)[0][sample_index] for i in range(self.nb_class)]
        
    def eval(self, steps):
        nb_ckp = ((steps+1)//self.nb_checkpoint) + 1
        acc_lst = np.zeros(steps+1, dtype=np.float32) 
        pred_lst = np.zeros((steps+1, self.data_size), dtype=np.uint8)
        score_lst = np.zeros((steps+1, self.data_size), dtype=np.float32)
        img_lst = np.zeros((nb_ckp, self.nb_class) + self.img_size, np.float32)
        saliency_lst = np.zeros((nb_ckp, self.nb_class) + self.img_size, dtype=np.float32)

        # print
        for s in range(steps+1):
            testloader = DataLoader(dataset=self.testset,
                                    batch_size=self.batch_size,
                                    shuffle=False,
                                    num_workers=0)

            scores, preds, acc, saliencys, indice = self.make_saliency(testloader)
            print('[{0:}/{1:}]: Total Accuracy: {2:.3%}'.format(s, steps, acc))

            # save
            score_lst[s] = scores
            pred_lst[s] = preds
            acc_lst[s] = acc
            
            # checkpoint
            if s % self.nb_checkpoint == 0:
                img_lst[s//self.nb_checkpoint] = self.testset.data[self.idx_by_class]
                saliency_lst[s//self.nb_checkpoint] = saliencys[self.idx_by_class]

            # remove
            if self.target=='mnist':
                self.testset.data[np.arange(self.data_size), indice[:,0], indice[:,1]] = 0
            elif self.target=='cifar10':
                self.testset.data[np.arange(self.data_size), indice[:,0], indice[:,1], indice[:,2]] = 0
            
                
        # save file to h5py
        self.save_file(acc_lst, score_lst, img_lst, saliency_lst, steps)

    def make_saliency(self, loader):
        correct = 0 
        preds = np.array([], dtype=np.uint8)
        scores = np.array([], dtype=np.float32)
        saliencys = np.array([], dtype=np.float32).reshape((0,) + self.img_size)
        indice = np.array([], dtype=np.uint8).reshape((0, len(self.img_size)))

        for idx, batch in enumerate(loader):
            imgs, targets = batch
            targets = torch.LongTensor(targets.numpy())

            b_size = imgs.shape[0]

            # predict
            outputs = self.model(imgs)
            prob_b, pred_b = outputs.max(1)
            score_b = outputs[np.arange(b_size), np.array(targets)].detach().numpy()

            # saliency map
            sal_maps_b, _, _ = self.saliency_map.generate_image(imgs, targets, layer=self.layer, color=self.color)
            sal_maps_b = sal_maps_b.reshape((b_size,) + self.img_size)

            # accuracy sum
            correct += (pred_b.detach().numpy() == np.array(targets)).sum()

            # replace maximum value to zero
            indice_b = np.array([np.unravel_index(sal_maps_b[i].argmax(), self.img_size) for i in range(b_size)])

            # save
            preds = np.append(preds, prob_b.detach().numpy())
            scores = np.append(scores, score_b)
            saliencys = np.vstack([saliencys, sal_maps_b])
            indice = np.vstack([indice,indice_b])

        acc = correct / self.data_size

        return scores, preds, acc, saliencys, indice


    def saliency_map_choice(self):
        color = True if self.target == 'cifar10' else False 

        if self.method == 'VBP':
            saliency_map = VanillaBackprop(self.model)
            layer = 0 
        elif self.method == 'IB':
            saliency_map = InputBackprop(self.model)
            layer = 0
        elif self.method == 'GB':
            saliency_map = GuidedBackprop(self.model)
            layer = 0 
        elif self.method == 'IG':
            saliency_map = IntegratedGradients(self.model)
            layer = 0 
        elif self.method == 'GC':
            saliency_map = GradCAM(self.model)
            layer = 8
        elif self.method == 'GB-GC':
            saliency_map = GuidedGradCAM(self.model)
            color = False
            layer = 8
        elif self.method == 'DeconvNet':
            saliency_map = DeconvNet(self.model, self.deconv_model)
            layer = 0       
        
        return saliency_map, layer, color

    
    def save_file(self, acc, score, img, saliency, steps):
        save_name = '../evaluation/{}_{}_steps{}_ckp{}_sample{}.hdf5'.format(self.target,self.method,steps,self.nb_checkpoint,self.sample_pct)
        with h5py.File(save_name, 'w') as hf:
            hf.create_dataset('acc', data=acc)
            hf.create_dataset('score', data=score)
            hf.create_dataset('image', data=img)
            hf.create_dataset('saliency', data=saliency)
            hf.close()



class ROAR(object):
    def __init__(self, args):
        # Config
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # data load
        if target == 'mnist':
            self.trainset, self.validset, self.testloader = mnist_load()
        elif target == 'cifar10':
            self.trainset, self.validset, self.testloader = cifar10_load()
        self.target = target
        self.img_size = self.trainset.dataset.data.shape[1:] # mnist : (28,28), cifar10 : (32,32,3)

        # model setting
        self.model = model
        self.model.eval()

        # saliency map
        self.method = method

        # save_name
        self.model_name = 'simple_cnn_{}_{}'.format(self.args.target, self.args.method) 
        self.savedir = '../checkpoint'
        self.logdir = '../logs'


        # remove percentage range
        self.remove_pct_lst = np.arange(0,1,0.1)

    def saliency_map(self):
        hf = h5py.File('../saliency_map/{}_{}.hdf5'.format(self.target, self.method))
        trainset_saliency_map = np.array(hf.get('trainset'))
        validset_saliency_map = np.array(hf.get('validset'))
        testset_saliency_map = np.array(hf.get('testset'))
        
        return (trainset_saliency_map, validset_saliency_map, testset_saliency_map)

    def remove_image(self, remove_pct, data_lst, saliency_map_lst):
        nb_pixel = np.prod(self.img_size)
        nb_remove = int(nb_pixel * remove_pct)

        for i in range(3):
            remove_idx = saliency_map_lst[i].argsort()[-nb_remove:]
            data_lst[i].dataset.data[remove_idx] = 0
        
        return data_lst

    def train(self):
        for remove_pct in self.remove_pct_lst:
            # data load
            data_lst = [self.trainset, self.validset, self.testset]
            # saliency map load
            saliency_map_lst = self.saliency_map()
            # remove importance values 
            data_lst = self.remove_image(remove_pct, data_lst, saliency_map_lst)

            # Load model
            net = SimpleCNN(self.args.target).to(device)
            print()


            # Model compile
            optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005) # MNIST
            criterion = nn.CrossEntropyLoss()

            # Train
            self.model_name = self._model_name + '_roar{}'.format(remove_pct)
            modeltrain = ModelTrain(model=net,
                                    data=trainloader,
                                    epochs=self.args.epochs,
                                    criterion=criterion,
                                    optimizer=optimizer,
                                    device=self.args.device,
                                    model_name=self.args.model_name,
                                    savedir=self.args.savedir,
                                    monitor=self.args.monitor,
                                    mode=self.args.mode,
                                    validation=validloader,
                                    verbose=self.args.verbose)           

    def test(self):
        pass



def selecticity_evaluation(args):
    # make directory
    if not os.path.isdir('../evaluation'):
        os.mkdir('../evaluation')

    # model load
    weights = torch.load('../checkpoint/simple_cnn_{}.pth'.format(args.target))
    model = SimpleCNN(args.target)
    model.load_state_dict(weights['model'])

    # evaluation method
    selectivity_method = Selectivity(model=model, 
                                     target=args.target, 
                                     batch_size=args.batch_size,
                                     method=args.method, 
                                     sample_pct=args.sample_pct)
    # evaluation
    selectivity_method.eval(args.steps)