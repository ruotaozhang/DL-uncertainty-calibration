import os
import platform
import logging
import copy
import torch
import torchvision.models as models
import numpy as np
from sklearn import metrics
from utils.customNetworks import DANN, FeatureExtractor, LabelClassifier, DomainClassifier


def msg(message: str):
    """
    Print message and add it to log
    :param message: a string of message to be added to log
    :return: None
    """
    logging.info(message)
    print(message)


def get_path(device_id=None) -> dict:
    """
    get paths to useful directories
    :param device_id: specify gpu id on Athena
    :return: a dictionary containing relevant paths
    """
    try:
        current_file_path = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
    except NameError:
        current_file_path = os.getcwd().replace('\\', '/')

    current_parent_path = os.path.dirname(current_file_path)
    current_parent_parent_path = os.path.dirname(current_parent_path)

    system = platform.system()
    if system == 'Linux':
        if 'centos' in platform.platform():
            data_path = current_parent_parent_path + '/datasets'
            save_path = current_parent_parent_path + '/saved_files'
            if device_id is not None:
                assert device_id in ['0', '1'], 'invalid argument device_id'
                os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                os.environ["CUDA_VISIBLE_DEVICES"] = device_id
        else:
            data_path = '/gpfs/data/jsteingr/rzhang63/datasets'
            save_path = os.path.dirname(data_path) + '/saved_files'
    else:
        data_path = 'D:/ML/datasets'
        save_path = current_parent_parent_path + '/saved_files'

    drd_img_path = data_path + '/diabetic-retinopathy-detection/train/'
    drd_label_path = data_path + '/diabetic-retinopathy-detection/'
    aptos_img_path = data_path + '/aptos2019-blindness-detection/train/'
    aptos_label_path = data_path + '/aptos2019-blindness-detection/'
    model_save_path = save_path + '/model_files/DR/'
    evaldict_save_path = save_path + '/eval_dicts/'
    tensorboard_path = save_path + '/tensorboard_summary/'
    return {'drd_img': drd_img_path, 'drd_label': drd_label_path, 'aptos_img': aptos_img_path,
            'aptos_label': aptos_label_path, 'model_save': model_save_path, 'evaldict_save': evaldict_save_path,
            'tfboard': tensorboard_path, 'current': current_file_path, 'current_parent': current_parent_path,
            'current_parent_parent': current_parent_parent_path}


def auc_estimation(y: np.ndarray, s: np.ndarray, bootstrap=False) -> dict:
    """
    Estimate of AUC and bootstrap estimate of variance
    :param y: true 0/1 response. np array with shape=(N,)
    :param s: predicted score. np array with shape=(N,)
    :param bootstrap: whether to compute 95% boostrap CI
    :return: a dictionary containing the estimated AUC and 95% CI
    """
    assert isinstance(y, np.ndarray) and len(y.shape) == 1, 'y: invalid input'
    assert isinstance(s, np.ndarray) and len(s.shape) == 1 and s.shape[0] == y.shape[0], 's: invalid input'
    assert set(y) == {0, 1}, 'y: invalid input'
    n = y.shape[0]
    auc = metrics.roc_auc_score(y, s)
    if bootstrap:
        b_mu_list = []
        for b in range(1000):
            np.random.seed(b)
            b_idx = np.random.choice(range(n), n)
            b_y = y[b_idx]
            b_s = s[b_idx]
            b_mu = metrics.roc_auc_score(b_y, b_s)
            b_mu_list.append(b_mu)
        return {'mu': auc, 'b_var': np.var(b_mu_list, ddof=1), 'b_ci': np.quantile(b_mu_list, (0.025, 0.975))}
    return {'mu': auc}


def add_dropout(model, drop_p: float):
    """
    Add dropout to every Conv2d layer in the network
    :param model: the neural net to which we wish to add dropout
    :param drop_p: the dropout rate for the added dropout layers
    :return: a model with dropout added after every Conv2d layer
    """
    layers_to_change = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            # print('found ', name)
            layers_to_change.append(name)

    # Iterate all layers to change
    for layer_name in layers_to_change:
        # Check if name is nested
        *parent, child = layer_name.split('.')
        # Nested
        if len(parent) > 0:
            # Get parent modules
            m = model.__getattr__(parent[0])
            for p in parent[1:]:
                m = m.__getattr__(p)
            # Get the conv layer
            orig_layer = m.__getattr__(child)
        else:
            m = model.__getattr__(child)
            orig_layer = copy.deepcopy(m)  # deepcopy, otherwise you'll get an infinite recusrsion
        # Add dropout layer here
        m.__setattr__(child, torch.nn.Sequential(orig_layer, torch.nn.Dropout(p=drop_p)))
    return model


def prob2uncertainty(prob_array: torch.Tensor) -> dict:
    """
    Compute predictive entropy based on predicted probability.
    :param prob_array: tensor array of shape=(batch_size,T,num_class)
    :return: a dict containing prob_array, mean_prob_array and predictive entropy
    """
    assert prob_array.dim() == 3, 'prob_array must have dim=3'

    prob_array = prob_array.cpu()
    assert np.argwhere(np.isnan(prob_array.numpy())).size == 0, 'prob array contains nan'

    # mean predicted probability averaged over T
    mean_prob_array = torch.mean(prob_array, dim=1)  # shape=(batch_size,num_class)

    # predictive entropy
    pred_entropy_array = torch.distributions.categorical.Categorical(mean_prob_array).entropy()  # shape=(batch_size)
    assert np.argwhere(np.isnan(pred_entropy_array.numpy())).size == 0, 'entropy contains nan'

    return {'prob': prob_array, 'mean_prob': mean_prob_array, 'entropy': pred_entropy_array}


def pred_batch(model_list: list, img_batch: torch.Tensor, T=10, mc=False):
    """
    Given a list of models and a batch of images, compute logit and probability predictions
    :param model_list: a list of neural network models
    :param img_batch: tensor representing a batch of images
    :param T: number of MC predictions to make if mc=True. If mc=False, T is set to 1
    :param mc: whether to do MC predictions
    :return: a tuple of logit array and probability array
    """
    assert isinstance(model_list, list), 'argument model must be a list of network instances'
    assert len(model_list) > 0, 'model list must be non-empty'

    M = len(model_list)  # M is the size of ensemble

    is_dann_v2 = isinstance(model_list[0], dict)

    with torch.no_grad():
        if is_dann_v2:
            # model must have fe,lc and dc
            assert {'fe', 'lc', 'dc'} == set(model_list[0].keys()), 'invalid input model'
            device = next(model_list[0]['lc'].parameters()).device
        else:
            assert len(set([next(m.parameters()).device for m in model_list])) == 1, 'models on different devices'
            device = next(model_list[0].parameters()).device
    if not mc:
        T = 1

    prob_list = []
    yhat_list = []
    with torch.no_grad():
        for m in range(M):
            if is_dann_v2:
                fe = model_list[m]['fe']
                lc = model_list[m]['lc']
                fe.eval()
                lc.eval()
                if mc:
                    fe.train()  # set dropout to train mode, but not BatchNorm
                    for l in fe.modules():
                        if isinstance(l, torch.nn.BatchNorm2d): l.eval()
            else:
                nnet = model_list[m]
                nnet.eval()
                if mc:
                    nnet.train()  # set dropout to train mode, but not BatchNorm
                    for l in nnet.modules():
                        if isinstance(l, torch.nn.BatchNorm2d): l.eval()
            for t in range(T):
                if is_dann_v2:
                    yhat = lc(fe(img_batch.to(device)))
                elif 'domain_classifier' in dict(nnet.named_children()).keys():
                    yhat, _ = nnet(img_batch.to(device), alpha=0.5)
                else:
                    yhat = nnet(img_batch.to(device))
                yhat = yhat.cpu()  # shape=(batch_size,output_dim)
                num_class = 2 if yhat.shape[1] == 1 else yhat.shape[1]
                prob = torch.nn.Softmax(dim=1)(yhat) if num_class > 2 else torch.cat(
                    (1.0 - torch.nn.Sigmoid()(yhat), torch.nn.Sigmoid()(yhat)), dim=1)
                yhat_list.append(yhat.unsqueeze(0))  # list of cpu tensor with shape=(1,batch_size,output_dim)
                prob_list.append(prob.unsqueeze(0))  # list of cpu tensor with shape=(1,batch_size,num_class)

        yhat_array = torch.cat(yhat_list, dim=0)  # cpu tensor of shape=(T*M,batch_size,output_dim)
        yhat_array = torch.transpose(yhat_array, 0, 1)  # shape=(batch_size,T*M,output_dim)

        prob_array = torch.cat(prob_list, dim=0)  # cpu tensor of shape=(T*M,batch_size,num_class)
        prob_array = torch.transpose(prob_array, 0, 1)  # shape=(batch_size,T*M,num_class)

        return yhat_array, prob_array


def pred_acc(model_list: list, data_loader, T=10, mc=False):
    """
    Given a list of models and a data loader, compute predictions, predictive entropy and AUC
    :param model_list: a list of neural network models
    :param data_loader:
    :param T: number of MC predictions to make if mc=True. If mc=False, T is set to 1
    :param mc: whether to do MC predictions
    :return: a dict containing predictions with entropy and AUC
    """
    assert isinstance(model_list, list), 'argument model must be a list of network instances'
    assert len(model_list) > 0, 'model list must be non-empty'

    M = len(model_list)  # M is the size of ensemble

    if not mc:
        T = 1

    n_samples = len(data_loader.sampler)
    label_list = []
    prob_list = []
    yhat_list = []
    with torch.no_grad():
        for idx, data_list in enumerate(data_loader):
            img_batch = data_list[0]
            label_batch = data_list[1]  # shape=(batch_size)
            yhat_batch, prob_batch = pred_batch(model_list=model_list, img_batch=img_batch, T=T, mc=mc)
            label_list.append(label_batch)  # list of tensors with shape=(batch_size)
            prob_list.append(prob_batch)  # list of tensors with shape=(batch_size,T*M,num_class)
            yhat_list.append(yhat_batch)  # list of tensors with shape=(batch_size,T*M,output_dim)

    label_array = torch.cat(label_list, dim=0)  # tensor of shape=(n_samples)
    prob_array = torch.cat(prob_list, dim=0)  # tensor of shape=(n_samples,T*M,num_class)
    yhat_array = torch.cat(yhat_list, dim=0)  # tensor of shape=(n_samples,T*M,output_dim)
    result_dict = prob2uncertainty(prob_array)
    _, predicted = torch.max(result_dict['mean_prob'], 1)

    num_class = prob_array.shape[2]
    assert num_class == 2, 'num_class must be 2'
    auc = round(metrics.roc_auc_score(label_array, result_dict['mean_prob'][:, 1]), 4)

    if M > 1:
        if mc:
            print('AUC of MC Ensemble on the {} images: auc={}'.format(n_samples, auc))
        else:
            print('AUC of Ensemble on the {} images: auc={}'.format(n_samples, auc))
    else:
        if mc:
            print('AUC of MC Dropout on the {} images: auc={}'.format(n_samples, auc))
        else:
            print('AUC of Regular Network on the {} images: auc={}'.format(n_samples, auc))

    return {'auc': auc, 'original_logits': yhat_array, 'prob': prob_array, 'mean_prob': result_dict['mean_prob'],
            'entropy': result_dict['entropy'], 'labels': label_array}


def make_net(output_dim: int, architecture='shuffle05', drop_p=0.0, pretrained=True, add_noise=False):
    """
    Construct a neural net based on the specification.
    :param output_dim: output dimension of the neural net. When num_class=2, output_dim=1, otherwise output_dim=num_class
    :param architecture: architecture of the network
    :param drop_p: the dropout rate to be added after every Conv2d layer in the network
    :param pretrained: whether to start with pre-trained weights
    :param add_noise: whether to add small Gaussian noise to pre-trained weights
    :return: a neural network model
    """
    assert architecture in ['shuffle05', 'shuffle10', 'mobile', 'resnet50'], 'wrong network architecture choice'
    if architecture == 'shuffle05':
        nnet = models.shufflenet_v2_x0_5(pretrained=pretrained)
        nnet.fc = torch.nn.Linear(nnet.fc.in_features, output_dim)
    elif architecture == 'shuffle10':
        nnet = models.shufflenet_v2_x1_0(pretrained=pretrained)
        nnet.fc = torch.nn.Linear(nnet.fc.in_features, output_dim)
    elif architecture == 'mobile':
        nnet = models.mobilenet_v2(pretrained=pretrained)
        nnet.classifier[-1] = torch.nn.Linear(nnet.classifier[-1].in_features, output_dim)
    else:
        nnet = models.resnet50(pretrained=pretrained)
        nnet.fc = torch.nn.Linear(nnet.fc.in_features, output_dim)

    if drop_p > 0:
        nnet = add_dropout(model=nnet, drop_p=drop_p)
        print('inserting dropout layers after all conv2d layers with drop_p={}'.format(drop_p))

    print('number of param: {}'.format(sum([param.nelement() for param in nnet.parameters()])))

    if add_noise:
        assert pretrained, 'can only add noise to pretrained weights'
        print('adding noise to pretrained weights')
        with torch.no_grad():
            for param in nnet.parameters():
                param.add_(torch.randn(param.size()) * 0.01)

    return torch.nn.DataParallel(nnet)


def make_DANN(output_dim, architecture='shuffle05', drop_p=0.0, pretrained=True, add_noise=False):
    """
    Construct a DANN model based on the specification. This particular implementation is based on
    https://github.com/fungtion/DANN
    :param output_dim: output dimension of the neural net. When num_class=2, output_dim=1, otherwise output_dim=num_class
    :param architecture: architecture of the network
    :param drop_p: the dropout rate to be added after every Conv2d layer in the network
    :param pretrained: whether to start with pre-trained weights
    :param add_noise: whether to add small Gaussian noise to pre-trained weights
    :return: a DANN model
    """
    nnet = DANN(architecture=architecture, output_dim=output_dim, pretrained=pretrained)

    if drop_p > 0:
        nnet = add_dropout(model=nnet, drop_p=drop_p)
        print('inserting dropout layers after all conv2d layers with drop_p={}'.format(drop_p))

    print('number of param: {}'.format(sum([param.nelement() for param in nnet.parameters()])))

    if add_noise:
        assert pretrained, 'can only add noise to pretrained weights'
        print('adding noise to pretrained weights')
        with torch.no_grad():
            for param in nnet.parameters():
                param.add_(torch.randn(param.size()) * 0.01)

    return torch.nn.DataParallel(nnet)


def make_DANN_v2(output_dim, architecture='shuffle05', drop_p=0.0, pretrained=True, add_noise=False):
    """
    Construct a DANN model based on the specification. This particular implementation is based on
    https://github.com/Yangyangii/DANN-pytorch
    :param output_dim: output dimension of the neural net. When num_class=2, output_dim=1, otherwise output_dim=num_class
    :param architecture: architecture of the network
    :param drop_p: the dropout rate to be added after every Conv2d layer in the network
    :param pretrained: whether to start with pre-trained weights
    :param add_noise: whether to add small Gaussian noise to pre-trained weights
    :return: a DANN model
    """
    fe = FeatureExtractor(architecture=architecture, pretrained=pretrained)
    with torch.no_grad():
        a = fe.to('cpu')(torch.rand(2, 3, 512, 512))
    input_dim = a.view(2, -1).shape[1]
    lc = LabelClassifier(input_dim=input_dim, output_dim=output_dim)
    dc = DomainClassifier(input_dim=input_dim)

    if drop_p > 0:
        fe = add_dropout(model=fe, drop_p=drop_p)
        print('inserting dropout layers after all conv2d layers with drop_p={}'.format(drop_p))

    print('number of param in FeatureExtractor: {}'.format(sum([param.nelement() for param in fe.parameters()])))
    print('number of param in LabelClassifier: {}'.format(sum([param.nelement() for param in lc.parameters()])))
    print('number of param in DomainClassifier: {}'.format(sum([param.nelement() for param in dc.parameters()])))

    if add_noise:
        assert pretrained, 'can only add noise to pretrained weights'
        print('adding noise to pretrained weights in FeatureExtractor')
        with torch.no_grad():
            for param in fe.parameters():
                param.add_(torch.randn(param.size()) * 0.01)

    return torch.nn.DataParallel(fe), torch.nn.DataParallel(lc), torch.nn.DataParallel(dc)


def entropy_based_referral(pred_prob:torch.Tensor, truth, uncert):
    """
    Perform entropy-based referral based on a set of predictions
    :param pred_prob: the predicted probabilities
    :param truth: the groud-truth response/outcome/label
    :param uncert: the uncertainty (entropy) associated with each prediction
    :return: a tensor of auc values
    """
    if isinstance(pred_prob, np.ndarray):
        pred_prob = torch.tensor(pred_prob)
    assert pred_prob.dim() == 2, 'pred_prob should have shape=(n_samples,num_class)'
    assert pred_prob.shape[1] == 2, 'num_class must be 2'
    _, preds = torch.max(pred_prob, 1)
    n_samples = pred_prob.shape[0]
    auc_array = torch.zeros((51))
    for i in range(51):
        p = 1 - i / 100  # percentage of retained cases
        k = int(n_samples * p)
        # take out the k smallest uncert values and corresponding indices
        uncert_sorted, uncert_sorted_idx = torch.topk(uncert, k=k, largest=False)
        auc_array[i] = metrics.roc_auc_score(truth[uncert_sorted_idx], pred_prob[uncert_sorted_idx, 1])
    return auc_array


def oracle_referral(pred_prob, truth, binning_scheme='quantile', n_bins=50):
    """
    Perform oracle referral based on the set of predictions.
    :param pred_prob: the predicted probabilities
    :param truth: the groud-truth response/outcome/label
    :param binning_scheme: how to bin the predicted probabilities
    :param n_bins: number of bins
    :return: a sequence of auc values
    """
    assert binning_scheme in ['uniform', 'quantile'], 'invalid binning scheme'
    prob_of_pos = pred_prob[:, 1]
    if binning_scheme == 'quantile':  # Determine bin edges by distribution of data
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(prob_of_pos, quantiles * 100)
        bins[-1] = bins[-1] + 1e-8
    else:
        bins = np.linspace(0., 1. + 1e-8, n_bins + 1)

    binids = np.digitize(prob_of_pos, bins) - 1
    unique_bin_ids = np.unique(binids)
    frac_of_pos = np.array([np.sum(truth[binids == id].numpy()) / np.sum(binids == id) for id in unique_bin_ids])

    auc0 = metrics.roc_auc_score(truth, prob_of_pos)
    sort_idx = np.argsort(np.abs(frac_of_pos - 0.5))
    p = 0  # percentage of removed cases
    auc_list = [(p, auc0)]
    i = 0
    bin_ids_2_remove = []
    while p <= 0.5:
        bin_ids_2_remove.append(sort_idx[i])
        retained_cases_preds = prob_of_pos[np.isin(binids, bin_ids_2_remove, invert=True)]
        retained_cases_truth = truth[np.isin(binids, bin_ids_2_remove, invert=True)]
        p = 1 - len(retained_cases_preds) / len(prob_of_pos)
        auc_p = metrics.roc_auc_score(retained_cases_truth, retained_cases_preds)
        auc_list.append((p, auc_p))
        i += 1

    return {'auc': auc_list, 'auc_rand': None}
