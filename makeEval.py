import pickle
from typing import Dict
from utils.customFunc import *
from utils.datasetLoader import *


def make_eval(model_folder_name: str, path_dict: Dict, model_folder_name_2_job_id_dict: Dict) -> dict:
    """
    Evaluate a given model.
    :param model_folder_name: folder name of the model to be evaluated
    :param path_dict: a dict containing relevant paths
    :param model_folder_name_2_job_id_dict: a mapping between model folder name and job id
    :return: a dict containing evaluation results
    """
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'

    # check if this model has already been evaluated
    eval_list = os.listdir(path_dict['evaldict_save'])
    job_id = model_folder_name_2_job_id_dict[model_folder_name]
    if len([x for x in eval_list if job_id in x]) > 0:
        msg('Eval dict for {} already exists. Skip. '.format(job_id))
        return 1

    # get the number of base models in the folder
    msg('Evaluating {}'.format(model_folder_name))
    model_list = sorted(os.listdir(path_dict['model_save'] + model_folder_name))
    num_models = len(model_list)

    if num_models != 1 and num_models != 5:
        msg('{} has wrong number of models. Skip. '.format(job_id))
        return 1
    msg('Number of models for {}: {}'.format(job_id, num_models))

    # load the models with trained weights
    net_list = []
    domain_loss_list = []
    label_loss_list = []
    for i in range(num_models):
        model_name = model_list[i]
        model_dict = torch.load(path_dict['model_save'] + model_folder_name + '/' + model_name)
        # print('train_loss' in model_dict.keys())
        if 'train_loss' in model_dict.keys():
            domain_loss = [0]
            label_loss = model_dict['train_loss']
        else:
            domain_loss = model_dict['domain_loss']
            label_loss = model_dict['label_loss']

        domain_loss_list.append(domain_loss)
        label_loss_list.append(label_loss)
        train_config = model_dict['train_config']

        if 'domain_adapted' not in train_config.keys():
            domain_adapted = 0
        elif isinstance(train_config['domain_adapted'], bool):
            domain_adapted = 1 if train_config['domain_adapted'] else 0
        else:
            domain_adapted = train_config['domain_adapted']

        output_dim = 1 if train_config['num_class'] == 2 else train_config['num_class']
        architecture = train_config['architecture']
        drop_p = train_config['conv_drop_p']

        if domain_adapted == 1:
            nnet = make_DANN(output_dim=output_dim, architecture=train_config['architecture'], drop_p=drop_p)
            nnet.load_state_dict(model_dict['net_para_dict']['net_para_epoch_100'])
            if isinstance(nnet, torch.nn.DataParallel):
                nnet = nnet.module
            net_list.append(nnet.to(device))
        elif domain_adapted == 2:
            fe, lc, dc = make_DANN_v2(output_dim, train_config['architecture'], drop_p=drop_p)
            fe.load_state_dict(model_dict['net_para_dict']['net_para_epoch_100']['fe'])
            lc.load_state_dict(model_dict['net_para_dict']['net_para_epoch_100']['lc'])
            dc.load_state_dict(model_dict['net_para_dict']['net_para_epoch_100']['dc'])
            if isinstance(fe, torch.nn.DataParallel):
                fe = fe.module
                lc = lc.module
                dc = dc.module
            fe = fe.to(device)
            lc = lc.to(device)
            dc = dc.to(device)
            nnet = {'fe': fe, 'lc': lc, 'dc': dc}
            net_list.append(nnet)
        else:
            nnet = make_net(output_dim=output_dim, architecture=architecture, drop_p=drop_p)
            nnet.load_state_dict(model_dict['net_para_dict']['net_para_epoch_100'])
            if isinstance(nnet, torch.nn.DataParallel):
                nnet = nnet.module
            net_list.append(nnet.to(device))

    msg('{}: {}'.format(job_id, train_config))

    # load DRD and APTOS
    drd_loader_dict = DR_data_loader(img_path=path_dict['drd_img'], label_path=path_dict['drd_label'],
                                     subset_size=train_config['subset_size'], batch_size=100, num_workers=0)

    aptos_loader_dict = Aptos_data_loader(img_path=path_dict['aptos_img'], label_path=path_dict['aptos_label'],
                                          batch_size=100, num_workers=0)

    # start evaluation
    mc = (train_config['conv_drop_p'] > 0)
    eval_dict = {}
    eval_dict['domain_loss'] = domain_loss_list
    eval_dict['label_loss'] = label_loss_list
    eval_dict['train_config'] = train_config
    eval_dict['model_list'] = net_list
    eval_dict['drd_train_eval'] = pred_acc(model_list=net_list, data_loader=drd_loader_dict['train'], T=100, mc=mc)
    eval_dict['drd_val_eval'] = pred_acc(model_list=net_list, data_loader=drd_loader_dict['val'], T=100, mc=mc)
    eval_dict['drd_test_eval'] = pred_acc(model_list=net_list, data_loader=drd_loader_dict['test'], T=100, mc=mc)
    eval_dict['aptos_whole_eval'] = pred_acc(model_list=net_list, data_loader=aptos_loader_dict['all'], T=100, mc=mc)

    # write to output and save
    output = open(path_dict['evaldict_save'] + job_id + '_eval_dict.pkl', 'wb')
    pickle.dump(eval_dict, output)
    output.close()
    msg('{} eval dict saved.'.format(job_id))
    return eval_dict


def main():
    device_id = '0'
    path_dict = get_path(device_id)

    model_folder_list = [x for x in os.listdir(path_dict['model_save']) if 'archive' not in x]
    model_folder_list = random.sample(model_folder_list, len(model_folder_list))
    job_id_list = [x[([i for i, c in enumerate(x) if c == '_'][-1] + 1):] for x in model_folder_list]
    assert len(job_id_list) == len(set(job_id_list)), 'duplicates in job id list'
    model_folder_name_2_job_id_dict = dict(zip(model_folder_list, job_id_list))

    logging.basicConfig(filename=path_dict['current_parent'] + '/log_makeEval.log', level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', filemode='a')

    for m in model_folder_list:
        make_eval(m, path_dict, model_folder_name_2_job_id_dict)


if __name__ == '__main__':
    main()
