import torch
from torch.autograd import Variable

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
    def __init__(self, model, target, method, ensemble=None, sample_pct=0.1, deconv_model=None, nb_class=10, sample_index=0):
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
            deconv_model : Deconvolution model. Only used if method is set tot DeconvNet.
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

        # sampling
        seed_everything()
        sample_size = int(len(self.testset) * sample_pct)
        sample_idx = np.random.choice(len(self.testset), sample_size, replace=False)
        print(sample_idx.shape)
        self.testset.data = self.testset.data[sample_idx]
        self.testset.targets = np.array(self.testset.targets)[sample_idx]
        self.sample_pct = sample_pct
        self.data_size = len(self.testset)

        # model setting
        self.model = model
        self.model.eval()
        if deconv_model != None:
            self.deconv_model = deconv_model

        # saliency map
        self.method = method
        self.ensemble = ensemble
        self.saliency_map, self.layer = self.saliency_map_choice()

        # sample
        self.nb_class = nb_class
        self.term = 5
        self.nb_checkpoint = 1 # 0 to number of checkpoint
        self.idx_by_class = [np.where(np.array(self.testset.targets)==i)[0][sample_index] for i in range(self.nb_class)]
        
    def eval(self, steps):
        self.nb_checkpoint += int(steps / self.term) # 0 to number of checkpoint

        acc_lst = np.zeros(steps+1, dtype=np.float32)
        pred_lst = np.zeros((steps+1, self.data_size), dtype=np.uint8)
        score_lst = np.zeros((steps+1, self.data_size), dtype=np.float32)
        img_lst = np.zeros((self.nb_checkpoint, self.nb_class) + self.img_size, dtype=np.float32)
        saliency_lst = np.zeros((self.nb_checkpoint, self.nb_class) + self.img_size, dtype=np.float32)

        # print
        print('data size: ',self.data_size)
        print('acc_lst.shape: ',acc_lst.shape)
        print('pred_lst.shape: ',pred_lst.shape)
        print('score_lst.shape: ',score_lst.shape)
        print('img_lst.shape: ',img_lst.shape)
        print('saliency_lst.shape: ',saliency_lst.shape)

        for s in range(steps+1):
            for i in tqdm(range(self.data_size)):
                # load image and target
                img = self.testset.data[i]
                target = np.array(self.testset.targets[i])

                # preprocessing
                pre_img = self.testset.transform(np.array(img)).unsqueeze(0)
                pre_img = Variable(pre_img, requires_grad=True)

                # predict
                probs = self.model(pre_img)
                score = probs[0][target].item()
                pred = probs.argmax().item()

                score_lst[s,i] = score
                pred_lst[s,i] = pred

                # saliency map
                output, _, _ = self.saliency_map.generate_image(pre_img, self.layer, target)
                if (self.target=='cifar10') and (self.method=='GC'):
                    # GradCAM output shape is (W,H)
                    output = cv2.applyColorMap(np.uint8(output*255), cv2.COLORMAP_JET)
                    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                output = output.reshape(self.img_size)

                # replace maximum value to zero
                idx = np.unravel_index(output.argmax(), self.img_size)
                img[idx] = 0
                self.testset.data[i] = img
                
                # save score and 
                score_lst[s,i] = score

                # checkpoint
                if (i in self.idx_by_class) and (s % self.nb_checkpoint):
                    img_lst[s//self.nb_checkpoint, target] = img
                    saliency_lst[s//self.nb_checkpoint, target] = output

            # save accuracy
            correct = (pred_lst[s] == np.array(self.testset.targets)).sum()
            acc_lst[s] = correct / self.data_size

            # print
            print('[{}/{}]: Total Accuracy: {}'.format(s, steps, acc_lst[s]))

        # save file to h5py
        self.save_file(acc_lst, score_lst, img_lst, saliency_lst, steps)

    def saliency_map_choice(self):
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
            layer = 11
        elif self.method == 'GB-GC':
            saliency_map = GuidedGradCAM(self.model)
            layer = 0
        elif self.method == 'DeconvNet':
            saliency_map = DeconvNet(self.model, self.deconv_model)
            layer = 0       
        
        return saliency_map, layer

    
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
    _, _, _, model = get_samples(args.target)
    # evaluation method
    selectivity_method = Selectivity(model=model, 
                                     target=args.target, 
                                     method=args.method, 
                                     sample_pct=args.sample_pct)
    # evaluation
    selectivity_method.eval(args.steps)