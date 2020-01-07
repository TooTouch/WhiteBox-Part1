import h5py
import numpy as np
import json

from saliency.attribution_methods import *
from models import SimpleCNNDeconv

import matplotlib.pyplot as plt

def visualize_saliencys(origin_imgs, results, probs, preds, classes, names, target, **kwargs):
    '''
    Visualize selectivity logs

    Args:
        origin_imgs: original images
        results: saliency maps
        probs: probability by class
        preds: predict class
        classes: target class
        names: attribution method names
        target: target dataset. ['mnist','cifar10']
    '''
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
    '''
    Visualize selectivity logs

    Args:
        target: target dataset. ['mnist','cifar10']
        methods: attribution methods 
        steps: number of step
        savedir: save path and save name
    '''
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


def visualize_ROARnKAR(targets, ratio_lst, eval_method, methods=None, attention=None, savedir=None, **kwargs):
    '''
    Visualize ROAR or KAR 

    Args:
        dataset: target dataset. ['mnist','cifar10']
        ratio_lst: pixel ratio list
        eval_method: ['ROAR','KAR']
        methods: attribution methods 
        attention: attention method
        savedir: save path and save name
    '''
    if methods==None:
        assert attention!=None, 'If methods is None, attention should not be None'
        methods = attention
    elif attention==None:
        assert methods!=None, 'If methods is None, attention should not be None'
    else:
        t_methods = methods + attention
        methods = t_methods.copy()

    # if attention is not None, define methods list
    for i in range(len(methods)):
        if methods[i] == 'CAM':
            methods[i] = 'CAM_CAM'
        elif methods[i] == 'CBAM':
            methods[i] = 'CBAM_GC'
        elif methods[i] == 'RAN':
            methods[i] = 'RAN_GC'
        elif methods[i] == 'WARN':
            methods[i] = 'WARN_GC'

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

    # load test accuracy
    for m in methods:
        for target in targets:
            if ('CAM' in m) or ('CBAM' in m):
                model_name = '{}_{}_{}'.format('simple_cnn', target,m.split('_')[0])
            elif ('RAN' in m) or ('WARN' in m):
                model_name = '{}_{}'.format(target, m.split('_')[0])
            else:
                model_name = '{}_{}'.format('simple_cnn', target)

            f = open('../logs/{}_logs.txt'.format(model_name),'r')
            acc = json.load(f)['test_result']
            test_acc[target][m].append(acc)

    # load roar/kar accuracy
    for target in targets:
        for m in methods:
            if ('RAN' in m) or ('WARN' in m):
                model_name = '{}_{}'.format(target, m)
            else:
                model_name = '{}_{}_{}'.format('simple_cnn', target, m)

            for ratio in ratio_lst[1:]:
                f = open('../logs/{0:}_{1:}{2:.1f}_logs.txt'.format(model_name, eval_method, ratio),'r')
                test_acc[target][m].append(json.load(f)['test_result'])

    # plotting
    f, ax = plt.subplots(1,2,figsize=size)
    for i in range(len(targets)):
        for j in range(len(methods)):
            ax[i].plot(ratio_lst, test_acc[targets[i]][methods[j]], label=methods[j], color=color[j], marker=marker[j])
        ax[i].set_title(f'{targets[i].upper()} {eval_method} score', size=fontsize)
        ax[i].set_ylabel('Accuracy', size=fontsize)
        r_k = 'remove' if eval_method=='ROAR' else 'keep'
        ax[i].set_xlabel(f'Pixel {r_k} ratio', size=fontsize)
        ax[i].set_xlim([0,1])
        ax[i].legend(loc='upper right')
    if savedir:
        plt.tight_layout()
        plt.savefig(savedir, dpi=dpi)


def make_saliency_map(dataset, model, methods, attr_method_lst, name_lst, **kwargs):
    '''
    Make sliency map

    Args:
        dataset: target dataset. ['mnist','cifar10']
        model: model to apply attribution method
        methods: attribution methods
        attr_method_lst: saliency map list
        name_lst: attribution method name list
    
    Return:

    '''
    if 'VBP' in methods:
        VBP_attr = VanillaBackprop(model, **kwargs)
        attr_method_lst.append(VBP_attr)
        name_lst.append('Vanilla\nBackprop')
    if 'IB' in methods:
        IB_attr = InputBackprop(model, **kwargs)
        attr_method_lst.append(IB_attr)
        name_lst.append('Input\nBackprop')
    if 'DeconvNet' in methods:
        model_deconv = SimpleCNNDeconv(dataset)
        deconvnet_attr = DeconvNet(model, model_deconv, **kwargs)
        attr_method_lst.append(deconvnet_attr)
        name_lst.append('DeconvNet')
    if 'IG' in methods:
        IG_attr = IntegratedGradients(model, **kwargs)
        attr_method_lst.append(IG_attr)
        name_lst.append('Integrated\nGradients')
    if 'GB' in methods:
        GB_attr = GuidedBackprop(model, **kwargs)
        attr_method_lst.append(GB_attr)
        name_lst.append('Guided\nBackprop')
    if 'GC' in methods:
        GC_attr = GradCAM(model, **kwargs)
        attr_method_lst.append(GC_attr)
        name_lst.append('Grad CAM')
    if 'GBGC' in methods:
        GBGC_attr = GuidedGradCAM(model, **kwargs)
        attr_method_lst.append(GBGC_attr)
        name_lst.append('Guided\nGrad CAM')

    return attr_method_lst, name_lst

def visualize_coherence_models(dataset, images, pre_images, targets, idx2classes, model, methods, model_names, savedir=None, **kwargs):
    '''
    Visualize coherence map that compare to attribution methods

    Args:
        dataset: target dataset. ['mnist','cifar10']
        images: original images
        pre_images: preprocessed images to evaluate
        target: targets to predict
        idx2classes: index and class dictionary
        model: model to apply attribution methods
        methods: attribution methods to extract saliency map
        savedir: save path and save name

    '''
    # initialize
    fontsize = 10 if 'fontsize' not in kwargs.keys() else kwargs['fontsize']
    size = (5,5) if 'size' not in kwargs.keys() else kwargs['size']
    random_state = 223 if 'random_state' not in kwargs.keys() else kwargs['random_state']
    dpi = None if 'dpi' not in kwargs.keys() else kwargs['dpi']
    wspace = 0 if 'wspace' not in kwargs.keys() else kwargs['wspace']
    hspace = 0 if 'hspace' not in kwargs.keys() else kwargs['hspace']

    params = {}
    n = 0
    for m in model_names:
        for i in range(len(methods)):
            if m == 'RAN':
                params[n] = {'layer':3}
            else:
                params[n] = {}
            n += 1

    # attribution methods
    attr_methods = []
    name_lst = []
    if isinstance(model, list):
        for i, m in enumerate(model):
            model_params = {'seq_name':'stages'} if model_names[i] == 'RAN' else {}
            attr_methods, name_lst = make_saliency_map(dataset, m, methods, attr_methods, name_lst, **model_params)
            name_lst[i] = name_lst[i] + f'\n{model_names[i]}'
    
    # initialize results
    nb_class = 10
    nb_methods = len(attr_methods)
    sal_maps_lst = np.zeros((nb_methods, ) + images.shape, dtype=np.float32)

    # make saliency maps
    for m in range(nb_methods):
        sal_maps, _, _ = attr_methods[m].generate_image(pre_images, targets, **params[m])
        sal_maps_lst[m] = sal_maps

    # plotting
    col = nb_methods + 1 # number of attribution methods + original image
    f, ax = plt.subplots(nb_class, col, figsize=size)
    # original images
    color = 'gray' if dataset == 'mnist' else None
    for i in range(nb_class):
        img = images[i].squeeze() if dataset == 'mnist' else images[i]
    
        ax[i,0].imshow(img, color)
        ax[i,0].set_ylabel('{0:}'.format(idx2classes[i]), size=fontsize)
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

    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    if savedir:
        plt.tight_layout()
        plt.savefig(savedir,dpi=dpi)


def visualize_coherence(dataset, images, pre_images, targets, idx2classes, model, methods, savedir=None, **kwargs):
    '''
    Visualize coherence map that compare to attribution methods

    Args:
        dataset: target dataset. ['mnist','cifar10']
        images: original images
        pre_images: preprocessed images to evaluate
        target: targets to predict
        idx2classes: index and class dictionary
        model: model to apply attribution methods
        methods: attribution methods to extract saliency map
        savedir: save path and save name

    '''
    # initialize
    fontsize = 10 if 'fontsize' not in kwargs.keys() else kwargs['fontsize']
    size = (5,5) if 'size' not in kwargs.keys() else kwargs['size']
    random_state = 223 if 'random_state' not in kwargs.keys() else kwargs['random_state']
    dpi = None if 'dpi' not in kwargs.keys() else kwargs['dpi']

    # attribution methods
    attr_methods = []
    name_lst = []
    attr_methods, name_lst = make_saliency_map(dataset, model, methods, attr_methods, name_lst, **kwargs)
    
    
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
    '''
    Visualize training log

    Args:
        train: training logs
        valid: validation logs
        title: graph title
        savedir: save path and save name
    '''
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



def visualize_models_log(log_lst, model_names, train_yn, savedir=None, **kwargs):
    '''
    Visualize logs of models

    Args:
        log_lst: log list of models
        model_names: model names
        train_yn: train or validation
        savedir: save path and save name
    '''
    # initialize
    fontsize = 10 if 'fontsize' not in kwargs.keys() else kwargs['fontsize']
    size = (5,5) if 'size' not in kwargs.keys() else kwargs['size']
    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'] if 'color' not in kwargs.keys() else kwargs['color']
    marker = ['o','v','^','s','x','*','p','d'] if 'marker' not in kwargs.keys() else kwargs['marker']
    random_state = 223 if 'random_state' not in kwargs.keys() else kwargs['random_state']
    dpi = None if 'dpi' not in kwargs.keys() else kwargs['dpi']
    nb_epoch = 30 if 'nb_epoch' not in kwargs.keys() else kwargs['nb_epoch']
    
    metrics = {'acc':'Accuracy', 'loss':'Loss'}
    
    f, ax = plt.subplots(1,2, figsize=size)

    for i, (k, v) in enumerate(metrics.items()):
        for j in range(len(log_lst)):
            m_logs = log_lst[j][train_yn][0][k]
            ax[i].plot(np.arange(nb_epoch), m_logs[:nb_epoch], label=model_names[j], color=color[j])
        ax[i].set_title('Comparison of mode {}'.format(train_yn), size=fontsize)
        ax[i].set_ylabel(v, size=fontsize)
        ax[i].set_xlabel('Epochs', size=fontsize)
        ax[i].legend()
    if savedir:
        plt.tight_layout()
        plt.savefig(savedir, dpi=dpi)