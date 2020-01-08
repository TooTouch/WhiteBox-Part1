import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from models import SimpleCNN, SimpleCNNDeconv
from dataload import mnist_load, cifar10_load
from saliency.attribution_methods import *
from saliency.ensembles import *
from utils import seed_everything, get_samples, ModelTrain

import cv2
import os
import h5py
import numpy as np
from tqdm import tqdm


# TODO : Add ensemble methods 
class Selectivity(object):
    def __init__(self, model, target, batch_size, method, ensemble=None, sample_pct=0.1, nb_class=10, sample_index=0):
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
        self.deconv_model = None

        # saliency map
        self.method = method
        self.ensemble = ensemble
        self.saliency_map, self.layer, self.color = self.saliency_map_choice()

        # sample
        self.nb_class = nb_class
        self.nb_checkpoint = 5 
        self.idx_by_class = [np.where(np.array(self.testset.targets)==i)[0][sample_index] for i in range(self.nb_class)]
        
    def eval(self, steps, save_dir):
        nb_ckp = ((steps+1)//self.nb_checkpoint) + 1
        acc_lst = np.zeros(steps+1, dtype=np.float32) 
        pred_lst = np.zeros((steps+1, self.data_size), dtype=np.uint8)
        score_lst = np.zeros((steps+1, self.data_size), dtype=np.float32)
        img_lst = np.zeros((nb_ckp, self.nb_class) + self.img_size, dtype=np.float32)
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
        self.save_file(acc_lst, score_lst, img_lst, saliency_lst, steps, save_dir)

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
            self.deconv_model = SimpleCNNDeconv(self.target)
            saliency_map = DeconvNet(self.model, self.deconv_model)
            layer = 0       
        
        return saliency_map, layer, color
    
    def save_file(self, acc, score, img, saliency, steps, save_dir):
        save_name = f'{save_dir}/{self.target}_{self.method}_steps{steps}_ckp{self.nb_checkpoint}_sample{self.sample_pct}.hdf5'
        with h5py.File(save_name, 'w') as hf:
            hf.create_dataset('acc', data=acc)
            hf.create_dataset('score', data=score)
            hf.create_dataset('image', data=img)
            hf.create_dataset('saliency', data=saliency)
            hf.close()


def adjust_image(ratio, trainloader, saliency_maps, eval_method):
    # set threshold
    data = trainloader.dataset.data
    img_size = data.shape[1:] # mnist : (28,28), cifar10 : (32,32,3)
    nb_pixel = np.prod(img_size)
    threshold = int(nb_pixel * (1-ratio))
    # rank indice
    re_sal_maps = saliency_maps.reshape(saliency_maps.shape[0], -1)
    indice = re_sal_maps.argsort().argsort()
    # get mask
    if eval_method=='ROAR':
        mask = indice < threshold
    elif eval_method=='KAR':
        mask = indice >= threshold
    mask = mask.reshape(data.shape)
    # remove
    trainloader.dataset.data = (data * mask).reshape(data.shape)
    
    return trainloader       



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
                                     sample_pct=args.ratio)
    # evaluation
    selectivity_method.eval(args.steps)

