import h5py
import numpy as np
import json

from saliency.attribution_methods import *
from model import SimpleCNNDeconv

import matplotlib.pyplot as plt

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


def visualize_selectivity(target, methods, steps, sample_pct, save_dir, **kwargs):
    # initialize
    fontsize = 10 if 'fontsize' not in kwargs.keys() else kwargs['fontsize']
    size = (5,5) if 'size' not in kwargs.keys() else kwargs['size']
    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'] if 'color' not in kwargs.keys() else kwargs['color']
    random_state = 223 if 'random_state' not in kwargs.keys() else kwargs['random_state']
    dpi = None if 'dpi' not in kwargs.keys() else kwargs['dpi']

    # set results dictionary by attribution methods
    attr_method_dict = {}
    for i in range(len(methods)):    
        attr_method_dict[methods[i]] = {'data':[]}

    # results load
    for attr_method in attr_method_dict.keys():
        hf = h5py.File(f'../evaluation/{target}_{attr_method}_steps{steps}_ckp5_sample{sample_pct}.hdf5', 'r')
        attr_method_dict[attr_method]['data'] = hf

    # Accuracy Change by Methods
    f, ax = plt.subplots(1,len(methods)+1, figsize=size)
    for i in range(len(methods)):
        method = methods[i]
        # results load
        hf = h5py.File(f'../evaluation/{target}_{method}_steps{steps}_ckp5_sample{sample_pct}.hdf5', 'r')
        # acc
        acc = np.array(hf['acc'])
        # plotting
        ax[0].plot(range(steps+1), acc, label=method, color=color[i])
        ax[0].legend()
        # close
        hf.close()
    # text
    ax[0].set_xlabel('# pixel removed', size=fontsize)
    ax[0].set_ylabel('Accuracy', size=fontsize)
    ax[0].set_title('[{}] Accuracy Change\nby Methods'.format(target.upper()), size=fontsize)
    ax[0].set_ylim([0,1])

    # Score Change by Methods
    for i in range(len(methods)):
        method = methods[i]
        # results load
        hf = h5py.File(f'../evaluation/{target}_{method}_steps{steps}_ckp5_sample{sample_pct}.hdf5', 'r')
        # score
        score = np.array(hf['score'])
        mean_score = np.mean(score, axis=1)
        # plotting average score
        ax[i+1].plot(range(steps+1), mean_score, label=method, color=color[i], linewidth=4)
        # sample index
        np.random.seed(random_state)
        sample_idx = np.random.choice(score.shape[1], 100, replace=False)
        sample_score = score[:,sample_idx]
        # plotting
        for j in range(100):
            ax[i+1].plot(range(steps+1), sample_score[:,j], color=color[i], linewidth=0.1)
        # text
        ax[i+1].set_xlabel('# pixel removed', size=fontsize)
        ax[i+1].set_ylabel('Score for correct class', size=fontsize)
        ax[i+1].set_title('[{}] {}\nScore Change'.format(target.upper(), method), size=fontsize)
        ax[i+1].set_ylim([0,1])
        # close
        hf.close()

    # figure adjust
    plt.subplots_adjust(wspace=-0.5, hspace=0)
    plt.tight_layout()
    
    # save
    plt.savefig(save_dir,dpi=kwargs['dpi'])


def visualize_ROARnKAR(targets, methods, ratio_lst, eval_method, savedir=None, **kwargs):
    # initialize
    fontsize = 10 if 'fontsize' not in kwargs.keys() else kwargs['fontsize']
    size = (5,5) if 'size' not in kwargs.keys() else kwargs['size']
    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'] if 'color' not in kwargs.keys() else kwargs['color']
    marker = ['o','v','^','s','x','*','p','d'] if 'marker' not in kwargs.keys() else kwargs['marker']
    random_state = 223 if 'random_state' not in kwargs.keys() else kwargs['random_state']
    dpi = None if 'dpi' not in kwargs.keys() else kwargs['dpi']

    # initialize methods acc by targets [mnist, cifar10]
    test_acc = {target: {} for target in targets}
    for target in targets:
        test_acc[target] = {m: [] for m in methods}

    # load logs
    for target in targets:
        f = open('../logs/simple_cnn_{}_logs.txt'.format(target), 'r')
        acc = json.load(f)['test_result']
        for m in methods:
            test_acc[target][m].append(acc)

    # load accuracy
    for target in targets:
        for m in methods:
            for ratio in ratio_lst[1:]:
                f = open('../logs/simple_cnn_{0:}_{1:}_{2:}{3:.1f}_logs.txt'.format(target, m, eval_method, ratio),'r')
                test_acc[target][m].append(json.load(f)['test_result'])

    # plotting
    f, ax = plt.subplots(1,2,figsize=size)
    for i in range(len(targets)):
        for j in range(len(methods)):
            ax[i].plot(ratio_lst, test_acc[targets[i]][methods[j]], label=methods[j], color=color[j], marker=marker[j])
        ax[i].set_title(f'{targets[i].upper()} {eval_method} score', size=fontsize)
        ax[i].set_ylabel('Accuracy', size=fontsize)
        ax[i].set_xlabel('# of remove ratio', size=fontsize)
        ax[i].set_xlim([0,1])
        ax[i].legend(loc='upper right')
    if savedir:
        plt.tight_layout()
        plt.savefig(savedir, dpi=dpi)



def visualize_coherence(dataset, images, pre_images, targets, idx2classes, model, methods, savedir=None, **kwargs):
    # initialize
    fontsize = 10 if 'fontsize' not in kwargs.keys() else kwargs['fontsize']
    size = (5,5) if 'size' not in kwargs.keys() else kwargs['size']
    random_state = 223 if 'random_state' not in kwargs.keys() else kwargs['random_state']
    dpi = None if 'dpi' not in kwargs.keys() else kwargs['dpi']

    # attribution methods
    attr_methods = []
    name_lst = []
    if 'VBP' in methods:
        VBP_attr = VanillaBackprop(model)
        attr_methods.append(VBP_attr)
        name_lst.append('Vanilla\nBackprop')
    if 'IB' in methods:
        IB_attr = InputBackprop(model)
        attr_methods.append(IB_attr)
        name_lst.append('Input\nBackprop')
    if 'DeconvNet' in methods:
        model_deconv = SimpleCNNDeconv(dataset)
        deconvnet_attr = DeconvNet(model, model_deconv)
        attr_methods.append(deconvnet_attr)
        name_lst.append('DeconvNet')
    if 'IG' in methods:
        IG_attr = IntegratedGradients(model)
        attr_methods.append(IG_attr)
        name_lst.append('Integrated\nGradients')
    if 'GB' in methods:
        GB_attr = GuidedBackprop(model)
        attr_methods.append(GB_attr)
        name_lst.append('Guided\nGradients')
    if 'GC' in methods:
        GC_attr = GradCAM(model)
        attr_methods.append(GC_attr)
        name_lst.append('Grad CAM')
    if 'GBGC':
        GBGC_attr = GuidedGradCAM(model)
        attr_methods.append(GBGC_attr)
        name_lst.append('Guided\nGrad CAM')
    
    # initialize results
    nb_class = 10
    nb_methods = len(attr_methods)
    sal_maps_lst = np.zeros((nb_methods, ) + images.shape, dtype=np.float32)

    # make saliency maps
    outputs = model(pre_images)
    probs, preds = outputs.detach().max(1)
    probs = probs.numpy()
    preds = preds.numpy()

    for m in range(nb_methods):
        sal_maps, _, _ = attr_methods[m].generate_image(pre_images, targets)
        sal_maps_lst[m] = sal_maps

    # plotting
    col = nb_methods + 1 # number of attribution methods + original image
    f, ax = plt.subplots(nb_class, col, figsize=size)
    # original images
    color = 'gray' if dataset == 'mnist' else None
    for i in range(nb_class):
        img = images[i].squeeze() if dataset == 'mnist' else images[i]
    
        ax[i,0].imshow(img, color)
        ax[i,0].set_ylabel('True: {0:}\nPred: {1:} ({2:.2%})'.format(idx2classes[i], int(preds[i]), probs[i]), size=fontsize-5)
        ax[i,0].set_xticks([])
        ax[i,0].set_yticks([])
        # set title
        if i == 0:
            ax[i,0].set_title('Original Image', size=fontsize)

    for i in range(nb_class*(col-1)):
        r = i//(col-1)
        c = i%(col-1)
        sal_map = sal_maps_lst[c,r].squeeze() if dataset == 'mnist' else sal_maps_lst[c,r]
        ax[r,c+1].imshow(sal_map, color)
        ax[r,c+1].axis('off')
        # set title
        if r == 0:
            ax[r,c+1].set_title(name_lst[c], size=fontsize)

    plt.subplots_adjust(wspace=0, hspace=0)
    if savedir:
        plt.tight_layout()
        plt.savefig(savedir,dpi=dpi)


def visualize_trainlogs(train, valid, title, savedir=None, **kwargs):
    # initialize
    fontsize = 10 if 'fontsize' not in kwargs.keys() else kwargs['fontsize']
    size = (5,5) if 'size' not in kwargs.keys() else kwargs['size']
    dpi = None if 'dpi' not in kwargs.keys() else kwargs['dpi']

    f, ax = plt.subplots(figsize=size)
    ax2 = ax.twinx()
    
    ax.plot(np.arange(len(train['acc'])), train['acc'], label='Train Acc', color='r')
    ax.plot(np.arange(len(valid['acc'])), valid['acc'], label='Valid Acc', color='c')
    ax2.plot(np.arange(len(train['loss'])), train['loss'], label='Train Loss', color='g')
    ax2.plot(np.arange(len(valid['loss'])), valid['loss'], label='Valid Loss', color='b')
    
    plt.title(title, size=fontsize)
    ax.legend(loc='upper right', fontsize=fontsize-2)
    ax2.legend(loc='lower right', fontsize=fontsize-2)

    if savedir:
        plt.tight_layout()
        plt.savefig(savedir, dpi=dpi)

