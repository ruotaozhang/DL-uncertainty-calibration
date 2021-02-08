import time
from typing import Dict
from datetime import datetime
import ast
import argparse
from utils.customFunc import *
from utils.datasetLoader import *


def nn_train(nnet, loss_fn, optimizer, train_loader) -> list:
    device = next(nnet.parameters()).device
    loss_list = []
    for x_batch, y_batch in train_loader:
        y_batch = y_batch.to(device)
        # set nnet to train mode and compute forward pass
        nnet = nnet.train()
        yhat = nnet(x_batch.to(device))
        # make it compatible with BCEWithLogitsLoss
        if yhat.shape[1] == 1: y_batch = y_batch.type(torch.float).view(-1, 1)
        # compute loss
        loss = loss_fn(yhat, y_batch.to(device)).mean()
        # Zeroes gradients, computes gradients, update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
    return loss_list


def dann_train(nnet, label_loss_fn, domain_loss_fn, optimizer, source_loader, target_loader, epoch, n_epoch):
    device = next(nnet.parameters()).device
    label_loss_list = []
    domain_loss_list = []
    len_dataloader = min(len(source_loader), len(target_loader))
    source_data_iter = iter(source_loader)
    target_data_iter = iter(target_loader)
    for batch_idx in range(len_dataloader):
        p = float(batch_idx + epoch * len_dataloader) / n_epoch / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # train model using source data
        source_img, source_label = next(source_data_iter)
        batch_size = source_img.shape[0]
        source_domain = torch.zeros(batch_size).type(torch.float).view(-1, 1)
        class_output, domain_output = nnet(x=source_img.to(device), alpha=alpha)
        if class_output.shape[1] == 1: source_label = source_label.type(torch.float).view(-1, 1)
        source_label_loss = label_loss_fn(class_output, source_label.to(device)).mean()
        source_domain_loss = domain_loss_fn(domain_output, source_domain.to(device)).mean()

        # train model using target data
        target_img, _ = next(target_data_iter)
        batch_size = target_img.shape[0]
        target_domain = torch.ones(batch_size).type(torch.float).view(-1, 1)
        _, domain_output = nnet(x=target_img.to(device), alpha=alpha)
        target_domain_loss = domain_loss_fn(domain_output, target_domain.to(device)).mean()

        total_loss = source_label_loss + source_domain_loss + target_domain_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        label_loss_list.append(source_label_loss.cpu().item())
        domain_loss_list.append(source_domain_loss.cpu().item() + target_domain_loss.cpu().item())
        if (batch_idx + 1) % 100 == 0 or batch_idx == 0:
            msg('Epoch: {}/{}, iter: {}/{}, alpha={}, s_label_loss={}, s_domain_loss={}, t_domain_loss={}'.format(
                epoch + 1,
                n_epoch,
                batch_idx + 1,
                len_dataloader,
                alpha,
                source_label_loss,
                source_domain_loss,
                target_domain_loss))

    return label_loss_list, domain_loss_list


def dann_train_v2(nnet_dict: Dict, loss_fn_dict: Dict, opt_dict: Dict, source_loader: DataLoader,
                  target_loader: DataLoader, epoch: int, n_epoch: int):
    # optimizer must match nnet parameters
    # for domain classification: source domain=1, target domain=0
    fe = nnet_dict['fe']
    lc = nnet_dict['lc']
    dc = nnet_dict['dc']
    label_loss_fn = loss_fn_dict['label']
    domain_loss_fn = loss_fn_dict['domain']
    fe_opt = opt_dict['fe']
    lc_opt = opt_dict['lc']
    dc_opt = opt_dict['dc']

    assert next(fe.parameters()).device == next(lc.parameters()).device == next(
        dc.parameters()).device, 'different parts on different device'
    device = next(fe.parameters()).device

    label_loss_list = []
    domain_loss_list = []
    source_loader_n_batch = len(source_loader)
    target_data_iter = iter(target_loader)
    for batch_idx, (source_images, source_labels) in enumerate(source_loader):
        try:
            target_images, _ = next(target_data_iter)
        except StopIteration:
            target_data_iter = iter(target_loader)
            target_images, _ = next(target_data_iter)

        # train domain classifier
        feat = fe(torch.cat([source_images, target_images], dim=0).to(device))
        domain_pred = dc(feat.detach())
        domain_truth = torch.cat([torch.ones(source_images.shape[0], 1), torch.zeros(target_images.shape[0], 1)],
                                 dim=0).to(device)
        dc_loss = domain_loss_fn(domain_pred, domain_truth).mean()
        dc_opt.zero_grad()
        dc_loss.backward()
        dc_opt.step()

        # train feature extractor and label classifier
        p = float(batch_idx + epoch * source_loader_n_batch) / n_epoch / source_loader_n_batch
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        label_pred_4_source_images = lc(feat[:source_images.shape[0], :])
        domain_pred = dc(feat)
        if label_pred_4_source_images.shape[1] == 1:
            source_labels = source_labels.type(torch.float).view(-1, 1).to(device)
        lc_loss = label_loss_fn(label_pred_4_source_images, source_labels).mean()
        dc_loss = domain_loss_fn(domain_pred, domain_truth).mean()
        total_loss = lc_loss - alpha * dc_loss
        fe_opt.zero_grad()
        lc_opt.zero_grad()
        dc_opt.zero_grad()
        total_loss.backward()
        lc_opt.step()
        fe_opt.step()

        label_loss_list.append(lc_loss.cpu().item())
        domain_loss_list.append(dc_loss.cpu().item())

        if (batch_idx + 1) % 100 == 0 or batch_idx == 0:
            msg('Epoch: {}/{}, iter: {}/{}, alpha={}, label_loss={}, domain_loss={}'.format(epoch + 1, n_epoch,
                                                                                            batch_idx + 1,
                                                                                            source_loader_n_batch,
                                                                                            alpha,
                                                                                            lc_loss,
                                                                                            dc_loss))

    return label_loss_list, domain_loss_list


def train_evaluate(train_config: Dict, drd_loader_dict: Dict, aptos_loader_dict: Dict, path_dict: Dict):
    # define dataloaders
    if train_config['bootstrap']:
        train_loader = drd_loader_dict['train_b_aug']
    elif train_config['domain_adapted'] == 0:
        train_loader = drd_loader_dict['train_aug']
    else:
        source_train_loader = drd_loader_dict['train_aug']
        target_train_loader = aptos_loader_dict['all_aug']

    # setup model and hyperparameters
    learning_rate = train_config['learning_rate']
    weight_decay = train_config['weight_decay']
    conv_drop_p = train_config['conv_drop_p']
    n_epoch = train_config['n_epoch']
    device = train_config['device']

    if train_config['num_class'] == 2:
        output_dim = 1
        label_loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none').to(device)
    else:
        output_dim = train_config['num_class']
        label_loss_fn = torch.nn.CrossEntropyLoss(reduction='none').to(device)

    if train_config['domain_adapted'] == 1:
        nnet = make_DANN(output_dim=output_dim, architecture=train_config['architecture'], drop_p=conv_drop_p,
                         pretrained=train_config['pretrained'], add_noise=train_config['add_noise']).to(device)
        domain_loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none').to(device)
        optimizer = torch.optim.Adam(nnet.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif train_config['domain_adapted'] == 2:
        fe, lc, dc = make_DANN_v2(output_dim=output_dim, architecture=train_config['architecture'], drop_p=conv_drop_p,
                                  pretrained=train_config['pretrained'], add_noise=train_config['add_noise'])
        fe = fe.to(device)
        lc = lc.to(device)
        dc = dc.to(device)
        domain_loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none').to(device)
        fe_opt = torch.optim.Adam(fe.parameters(), lr=learning_rate, weight_decay=weight_decay)
        lc_opt = torch.optim.Adam(lc.parameters(), lr=learning_rate, weight_decay=weight_decay)
        dc_opt = torch.optim.Adam(dc.parameters(), lr=learning_rate, weight_decay=weight_decay)
        nnet = {'fe': fe, 'lc': lc, 'dc': dc}
        opt_dict = {'fe': fe_opt, 'lc': lc_opt, 'dc': dc_opt}
        loss_fn_dict = {'label': label_loss_fn, 'domain': domain_loss_fn}
    else:
        nnet = make_net(output_dim=output_dim, architecture=train_config['architecture'], drop_p=conv_drop_p,
                        pretrained=train_config['pretrained'], add_noise=train_config['add_noise']).to(device)
        optimizer = torch.optim.Adam(nnet.parameters(), lr=learning_rate, weight_decay=weight_decay)
        domain_loss_fn = None

    # start training
    net_para_dict = {}
    label_losses = [0]
    domain_losses = [0]
    epoch_time = []
    for epoch in range(n_epoch):
        # torch.cuda.synchronize()
        tic = time.time()
        if train_config['domain_adapted'] == 0:
            epoch_label_losses = nn_train(nnet, loss_fn=label_loss_fn, optimizer=optimizer, train_loader=train_loader)
            epoch_domain_losses = [0]
        elif train_config['domain_adapted'] == 1:
            epoch_label_losses, epoch_domain_losses = dann_train(nnet, label_loss_fn=label_loss_fn,
                                                                 domain_loss_fn=domain_loss_fn,
                                                                 optimizer=optimizer,
                                                                 source_loader=source_train_loader,
                                                                 target_loader=target_train_loader,
                                                                 n_epoch=n_epoch,
                                                                 epoch=epoch)
        else:
            # note: nnet here is actually a dictionary
            epoch_label_losses, epoch_domain_losses = dann_train_v2(nnet_dict=nnet, loss_fn_dict=loss_fn_dict,
                                                                    opt_dict=opt_dict,
                                                                    source_loader=source_train_loader,
                                                                    target_loader=target_train_loader,
                                                                    epoch=epoch, n_epoch=n_epoch)
        label_losses += epoch_label_losses
        domain_losses += epoch_domain_losses
        # torch.cuda.synchronize()
        epoch_time.append((time.time() - tic) / 60)
        if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == n_epoch:
            avg_epoch_time = round(np.array(epoch_time).mean(), 1)
            msg('Epoch {}/{}: mean epoch time={}mins, label_loss={}, domain_loss={}'.format(epoch + 1, n_epoch,
                                                                                            avg_epoch_time,
                                                                                            label_losses[-1],
                                                                                            domain_losses[-1]))
            if train_config['domain_adapted'] <= 1:
                net_para_dict['net_para_epoch_{}'.format(epoch + 1)] = nnet.to('cpu').state_dict()
                nnet = nnet.to(device)
            else:
                net_para_dict['net_para_epoch_{}'.format(epoch + 1)] = {'fe': fe.to('cpu').state_dict(),
                                                                        'lc': lc.to('cpu').state_dict(),
                                                                        'dc': dc.to('cpu').state_dict()}
                fe = fe.to(device)
                lc = lc.to(device)
                dc = dc.to(device)

    # evaluate on drd_train, drd_test and aptos
    mc = (conv_drop_p > 0)
    method = 'MC Dropout' if mc else 'regular'
    msg('evaluation method is {} since conv_drop_p={}'.format(method, conv_drop_p))
    drd_train_eval = pred_acc(model_list=[nnet], data_loader=drd_loader_dict['train'], T=100, mc=mc)
    drd_test_eval = pred_acc(model_list=[nnet], data_loader=drd_loader_dict['test'], T=100, mc=mc)
    aptos_whole_eval = pred_acc(model_list=[nnet], data_loader=aptos_loader_dict['all'], T=100, mc=mc)

    # define model file name
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%H%M%S")
    model_name_string0 = '{}_subsetsize-{}_bootstrap-{}_addnoise-{}/'.format(train_config['architecture'],
                                                                             train_config['subset_size'],
                                                                             train_config['bootstrap'],
                                                                             train_config['add_noise'])
    model_name_string1 = model_name_string0[:-1] + '_nepoch{}_convP-{}_time-{}_{}_{}_{}.tar'.format(
        train_config['n_epoch'], conv_drop_p, dt_string, drd_train_eval['auc'], drd_test_eval['auc'],
        aptos_whole_eval['auc'])

    try:
        os.mkdir(path_dict['model_save'] + model_name_string0[:-1] + '_' + train_config['array_job_id'])
    except FileExistsError:
        print('folder already exists')
    torch.save({'net_para_dict': net_para_dict, 'label_loss': label_losses, 'domain_loss': domain_losses,
                'train_config': train_config},
               path_dict['model_save'] + model_name_string0[:-1] + '_' + train_config[
                   'array_job_id'] + '/' + model_name_string1)
    msg(model_name_string1 + ' ' + 'saved')
    return {'model': nnet, 'drd_train_eval': drd_train_eval, 'drd_test_eval': drd_test_eval,
            'aptos_whole_eval': aptos_whole_eval, 'label_loss': label_losses, 'domain_loss': domain_losses}


def main():
    device_id = '0'
    path_dict = get_path(device_id)

    parser = argparse.ArgumentParser()
    parser.add_argument("-job", default='000000')
    parser.add_argument("-arrayjob", default='000000')
    args = parser.parse_args()

    job_id = str(args.job)
    array_job_id = str(args.arrayjob)
    logging.basicConfig(filename=path_dict['current_parent'] + '/log_' + array_job_id + '.log', level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', filemode='a')

    msg('job array id = {}, job id = {}'.format(array_job_id, job_id))
    msg(torch.__version__)
    msg(torch.version.cuda)
    msg('device id={}'.format(device_id))
    msg('current file/working dir: {}'.format(path_dict['current']))
    msg('parent dir of current file/working dir: {}'.format(path_dict['current_parent']))

    config_file = open(path_dict['current_parent'] + '/model_config.txt', 'r')
    contents = config_file.read()
    train_config = ast.literal_eval(contents)
    config_file.close()

    assert train_config['label_type'] == 'binary', 'model can only deal with binary labels'
    train_config['num_class'] = 2
    train_config['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    train_config['n_gpu'] = torch.cuda.device_count()
    train_config['array_job_id'] = array_job_id
    train_config['job_id'] = job_id
    assert train_config['domain_adapted'] in [0, 1, 2], 'domain_adapted: invalid input'
    if train_config['domain_adapted'] > 0:
        train_config['bootstrap'] = False
        msg('Since training DANN, setting bootstrap=False')
    msg(train_config)

    msg('-------------------- Start --------------------')

    # load DRD data
    drd_loader_dict = DR_data_loader(img_path=path_dict['drd_img'], label_path=path_dict['drd_label'],
                                     subset_size=train_config['subset_size'], batch_size=train_config['batch_size'])

    # load APTOS2019 data
    aptos_loader_dict = Aptos_data_loader(img_path=path_dict['aptos_img'], label_path=path_dict['aptos_label'],
                                          batch_size=train_config['batch_size'])

    msg('Start training of final model')
    out_dict = train_evaluate(train_config, drd_loader_dict, aptos_loader_dict, path_dict)
    msg('-------------------- End --------------------')
    return out_dict


if __name__ == '__main__':
    result_dict = main()
