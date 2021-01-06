import torch
import torchvision.models as models
import numpy as np
from scipy.stats import multinomial
from sklearn import metrics
import copy
from utils.customNetworks import DANN, FeatureExtractor, LabelClassifier, DomainClassifier
import dalib.adaptation
from dalib.modules.domain_discriminator import DomainDiscriminator
import os
import platform
import logging


def multinomial_cov(array):
    # array must be of shape: (num_class,)
    c = np.outer(array, array)
    np.fill_diagonal(c, 0)
    return np.diag(array * (1 - array)) - c


def multinomial_cov_highdim(array):
    # array must be of shape: (batch_size,T,num_class)
    out = np.random.rand(array.shape[0], array.shape[1], array.shape[2], array.shape[2])
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            out[i, j, :, :] = multinomial_cov(array[i, j, :])
    return out


def add_dropout(model, drop_p, type='1d'):
    assert type in ['1d', '2d'], 'invalid type value'
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
        # Add your layer here
        if type == '1d':
            m.__setattr__(child, torch.nn.Sequential(orig_layer, torch.nn.Dropout(p=drop_p)))
        else:
            m.__setattr__(child, torch.nn.Sequential(orig_layer, torch.nn.Dropout2d(p=drop_p)))
    return model


def make_net(output_dim, architecture='shuffle05', drop_p=0.0, pretrained=True, add_noise=False):
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


def make_dalib(output_dim, da_type, architecture='shuffle05', drop_p=0.0, pretrained=True, add_noise=False):
    assert da_type in ['dann', 'cdan'], 'type: invalid input'
    assert architecture in ['shuffle05', 'shuffle10', 'mobile', 'resnet50'], 'network architecture: invalid'
    if architecture == 'shuffle05':
        backbone = models.shufflenet_v2_x0_5(pretrained=pretrained)
        backbone.fc = torch.nn.Identity()
    elif architecture == 'shuffle10':
        backbone = models.shufflenet_v2_x1_0(pretrained=pretrained)
        backbone.fc = torch.nn.Identity()
    elif architecture == 'mobile':
        backbone = models.mobilenet_v2(pretrained=pretrained)
        backbone.classifier = torch.nn.Identity()
    else:
        backbone = models.resnet50(pretrained=pretrained)
        backbone.fc = torch.nn.Identity()

    with torch.no_grad():
        a = backbone.to('cpu')(torch.rand(2, 3, 512, 512))
    backbone.out_features = a.view(2, -1).shape[1]

    if drop_p > 0:
        backbone = add_dropout(model=backbone, drop_p=drop_p)
        print('inserting dropout layers after all conv2d layers in backbone with drop_p={}'.format(drop_p))

    print('number of param: {}'.format(sum([param.nelement() for param in backbone.parameters()])))

    if add_noise:
        assert pretrained, 'can only add noise to pretrained weights'
        print('adding noise to pretrained weights in backbone')
        with torch.no_grad():
            for param in backbone.parameters():
                param.add_(torch.randn(param.size()) * 0.01)
    if da_type == 'dann':
        classifier = dalib.adaptation.dann.ImageClassifier(backbone, output_dim)  # bottleneck_dim = 256 default
        domain_discriminator = DomainDiscriminator(in_feature=classifier.features_dim, hidden_size=128)
        domain_adv = dalib.adaptation.dann.DomainAdversarialLoss(domain_discriminator)
    else:
        classifier = dalib.adaptation.cdan.ImageClassifier(backbone, output_dim)  # bottleneck_dim = 256 default
        domain_discriminator = DomainDiscriminator(in_feature=classifier.features_dim, hidden_size=128)
        domain_adv = dalib.adaptation.cdan.ConditionalDomainAdversarialLoss(domain_discriminator)

    return classifier, domain_adv


def refer_performance2(pred_prob, truth, binning_scheme='quantile', n_bins=50):
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


def refer_performance(pred_prob, truth, uncert):
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


def prob2uncertainty(prob_array):
    # prob_array must be tensor array of shape=(batch_size,T,num_class)
    # input prob_array can be on any device, but all function returns are on cpu
    # T=1 is ok, but uncertainty estimates do not make sense
    assert prob_array.dim() == 3, 'prob_array must have dim=3'

    prob_array = prob_array.cpu()
    assert np.argwhere(np.isnan(prob_array.numpy())).size == 0, 'prob array contains nan'

    mean_prob_array = torch.mean(prob_array, dim=1)  # shape=(batch_size,num_class)

    aleatoric = multinomial.cov(n=1, p=prob_array)  # shape=(batch_size,T,num_class,num_class)
    if np.argwhere(np.isnan(aleatoric)).size != 0:
        aleatoric = multinomial_cov_highdim(prob_array)
        assert np.argwhere(np.isnan(aleatoric)).size == 0, 'aleatoric contains nan'
    aleatoric = np.mean(aleatoric, axis=1)  # shape=(batch_size,num_class,num_class)
    aleatoric = torch.from_numpy(aleatoric)

    epistemic = [np.cov(x, rowvar=False, bias=True) for x in prob_array]
    assert np.argwhere(np.isnan(epistemic)).size == 0, 'epistemic contains nan'
    epistemic = np.stack(epistemic, axis=0)  # shape=(batch_size,num_class,num_class)
    epistemic = torch.from_numpy(epistemic)
    # cov_y = aleatoric + epistemic
    # cov_y = multinomial.cov(n=1,p=mean_prob_array) # shape=(batch_size,num_class,num_class)
    # assert cov_y == multinomial.cov(n=1,p=mean_prob_array),'something is wrong'

    pred_entropy_array = torch.distributions.categorical.Categorical(mean_prob_array).entropy()  # shape=(batch_size)
    assert np.argwhere(np.isnan(pred_entropy_array.numpy())).size == 0, 'entropy contains nan'

    pred_entropy_array2 = torch.distributions.categorical.Categorical(prob_array).entropy()  # shape=(batch_size,T)
    aleatoric2 = torch.mean(pred_entropy_array2, dim=1)  # shape=(batch_size)
    epistemic2 = pred_entropy_array - aleatoric2

    assert np.argwhere(np.isnan(aleatoric2.numpy())).size == 0, 'aleatoric2 contains nan'
    assert np.argwhere(np.isnan(epistemic2.numpy())).size == 0, 'epistemic2 contains nan'

    return {'prob': prob_array, 'mean_prob': mean_prob_array, 'aleatoric': aleatoric, 'epistemic': epistemic,
            'entropy': pred_entropy_array, 'aleatoric2': aleatoric2, 'epistemic2': epistemic2}


def pred_batch(model_list, img_batch, T=10, mc=False):
    # T is the number of MC samples
    assert isinstance(model_list, list), 'argument model must be a list of network instances'
    assert len(model_list) > 0, 'model list must be non-empty'

    M = len(model_list)  # M is the size of ensemble

    domain_adapted = isinstance(model_list[0], dict)
    with torch.no_grad():
        if domain_adapted:
            # model must have lc and dd
            assert {'lc', 'dd'} == set(model_list[0].keys()), 'invalid input model'
            device = next(model_list[0]['lc'].parameters()).device
        else:
            assert len(
                set(list(map(lambda x: next(x.parameters()).device, model_list)))) == 1, 'models on different devices'
            device = next(model_list[0].parameters()).device

    if not mc:
        T = 1

    with torch.no_grad():
        prob_list = []
        yhat_list = []
        for m in range(M):
            if domain_adapted:
                lc = model_list[m]['lc']
                lc.eval()
                if mc:
                    lc.train()  # set dropout to train mode, but not BatchNorm
                    for l in lc.modules():
                        if isinstance(l, torch.nn.BatchNorm2d): l.eval()
            else:
                nnet = model_list[m]
                nnet.eval()
                if mc:
                    nnet.train()  # set dropout to train mode, but not BatchNorm
                    for l in nnet.modules():
                        if isinstance(l, torch.nn.BatchNorm2d): l.eval()
            for t in range(T):
                if domain_adapted:
                    yhat, _ = lc(img_batch.to(device))
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


def pred_acc(model_list, data_loader, T=10, mc=False):
    # T is the number of MC samples
    assert isinstance(model_list, list), 'argument model must be a list of network instances'
    assert len(model_list) > 0, 'model list must be non-empty'

    M = len(model_list)  # M is the size of ensemble

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
    acc = round((predicted == label_array).type(torch.float).mean().item(), 4)

    num_class = prob_array.shape[2]
    if num_class == 2:
        auc = round(metrics.roc_auc_score(label_array, result_dict['mean_prob'][:, 1]), 4)
    else:
        auc = round(metrics.roc_auc_score(label_array, result_dict['mean_prob'], average='weighted', multi_class='ovo'),
                    4)

    if M > 1:
        if mc:
            print('Acc and AUC of MC Ensemble on the {} images: acc={} and auc={}'.format(n_samples, acc, auc))
        else:
            print('Acc and AUC of Ensemble on the {} images: acc={} and auc={}'.format(n_samples, acc, auc))
    else:
        if mc:
            print('Acc and AUC of MC Dropout on the {} images: acc={} and auc={}'.format(n_samples, acc, auc))
        else:
            print('Acc and AUC of Regular Network on the {} images: acc={} and auc={}'.format(n_samples, acc, auc))

    return {'acc': acc, 'auc': auc, 'original_logits': yhat_array, 'prob': prob_array,
            'mean_prob': result_dict['mean_prob'],
            'entropy': result_dict['entropy'], 'aleatoric': result_dict['aleatoric'],
            'epistemic': result_dict['epistemic'], 'labels': label_array,
            'aleatoric2': result_dict['aleatoric2'], 'epistemic2': result_dict['epistemic2']}




def meanprob2logit(mean_probs):
    mean_probs = torch.tensor(
        list(map(lambda x: x - 1e-5 if x == 1.0 else (x + 1e-5 if x == 0.0 else x), list(mean_probs.numpy()))))
    mean_prob_logits = torch.log(mean_probs / (1 - mean_probs))
    assert np.argwhere(np.isnan(mean_prob_logits.numpy())).size == 0, 'logit has nan'
    assert len(list(filter(lambda i: list(mean_prob_logits.numpy())[i] in [-np.inf, np.inf],
                           range(mean_prob_logits.shape[0])))) == 0, 'logit has inf'
    return mean_prob_logits


def auc_estimation(y: np.ndarray, s: np.ndarray, bootstrap=False) -> dict:
    """
    Estimate of AUC and estimate of variance
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


def get_dir():
    try:
        current_file_dir = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
    except NameError:
        current_file_dir = os.getcwd().replace('\\', '/')

    current_parent_dir = os.path.dirname(current_file_dir)
    current_parent_parent_dir = os.path.dirname(current_parent_dir)

    system = platform.system()
    if system == 'Linux':
        if 'centos' in platform.platform():
            data_dir = current_parent_parent_dir + '/datasets'
            save_dir = current_parent_parent_dir + '/saved_files'
        else:
            data_dir = '/gpfs/data/jsteingr/rzhang63/datasets'
            save_dir = os.path.dirname(data_dir) + '/saved_files'
    else:
        data_dir = 'D:/ML/datasets'
        save_dir = current_parent_parent_dir + '/saved_files'

    drd_dir = data_dir + '/diabetic-retinopathy-detection/train/'
    aptos_dir = data_dir + '/aptos2019-blindness-detection/train/'
    model_save_dir = save_dir + '/model_files/DR/'
    evaldict_save_dir = save_dir + '/eval_dicts/'
    tensorboard_dir = save_dir + '/tensorboard_summary/'
    return {'drd': drd_dir, 'aptos': aptos_dir, 'model_save': model_save_dir, 'evaldict_save': evaldict_save_dir,
            'tfboard': tensorboard_dir, 'current': current_file_dir, 'current_parent': current_parent_dir,
            'current_parent_parent': current_parent_parent_dir}


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def msg(message):
    logging.info(message)
    print(message)

