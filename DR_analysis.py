import utils.customFunc
import torch
import logging
import os
import pickle
import pandas as pd
import calibration as cal
import pymannkendall as mk
from scipy.integrate import trapz
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from utils.datasetLoader import Aptos_data_loader


def make_tb(eval_dict_name, path_dict, pd_dict):
    # get job id
    job_id = eval_dict_name[:[i for i, c in enumerate(eval_dict_name) if c == '_'][0]]
    utils.customFunc.msg('job id = {}'.format(job_id))

    # load pickle file
    pkl_file = open(path_dict['evaldict_save'] + eval_dict_name, 'rb')
    eval_dict = pickle.load(pkl_file)
    pkl_file.close()

    # check if model was trained with domain adaptation
    if 'domain_adapted' not in eval_dict['train_config'].keys():
        domain_adapted = False
    elif isinstance(eval_dict['train_config']['domain_adapted'], bool):
        domain_adapted = eval_dict['train_config']['domain_adapted']
    elif eval_dict['train_config']['domain_adapted'] == 0:
        domain_adapted = False
    else:
        domain_adapted = True
    # assert not domain_adapted, 'Do not evaluate models trained with DA'
    # if domain_adapted:
    #    return 1

    # number of models trained in this job. Should be either 1 or 5
    num_models = len(eval_dict['model_list'])
    assert num_models in [1, 5], 'invalid num_models: {}'.format(num_models)

    # add basic model info to pd
    pd_dict['JobID'].append(job_id)
    pd_dict['Network'].append(eval_dict['train_config']['architecture'])
    if num_models == 1:
        pd_dict['Method'].append('MC Dropout')
    elif domain_adapted:
        pd_dict['Method'].append('DANN Ensemble')
    elif eval_dict['train_config']['conv_drop_p'] > 0:
        pd_dict['Method'].append('MC Ensemble')
    else:
        pd_dict['Method'].append('Deep Ensemble')

    if eval_dict['train_config']['subset_size'] == 0:
        pd_dict['Train Size'].append('30126')
    else:
        pd_dict['Train Size'].append('4000')

    if eval_dict['train_config']['bootstrap']:
        pd_dict['Bootstrap'].append('1')
    else:
        pd_dict['Bootstrap'].append('0')

    # split APTOS predictions into two sets: val + test
    aptos_loader_dict = Aptos_data_loader(img_path=path_dict['aptos_img'], label_path=path_dict['aptos_label'])
    aptos_val_idx = aptos_loader_dict['test_idx']
    aptos_test_idx = aptos_loader_dict['train_idx']

    # validation data for calibration
    drd_val_mean_prob = eval_dict['drd_val_eval']['mean_prob']
    # drd_val_logits = utils.customFunc.probs2logits(drd_val_mean_prob)
    drd_val_labels = eval_dict['drd_val_eval']['labels']
    aptos_val_mean_prob = eval_dict['aptos_whole_eval']['mean_prob'][aptos_val_idx]
    # aptos_val_logits = utils.customFunc.probs2logits(aptos_val_mean_prob)
    aptos_val_labels = eval_dict['aptos_whole_eval']['labels'][aptos_val_idx]

    # train 2 calibrators, one for DRD and one for APTOS
    # drd_calibrator = utils.customFunc.temp_scaling_calibrator()
    # drd_calibrator.train_calibration(drd_val_mean_prob, drd_val_labels)
    # aptos_calibrator = utils.customFunc.temp_scaling_calibrator()
    # aptos_calibrator.train_calibration(aptos_val_mean_prob, aptos_val_labels)
    drd_calibrator = cal.PlattCalibrator(drd_val_labels.shape[0], 15)
    drd_calibrator.train_calibration(drd_val_mean_prob.numpy()[:, 1], drd_val_labels.numpy())
    aptos_calibrator = cal.PlattCalibrator(aptos_val_labels.shape[0], 15)
    aptos_calibrator.train_calibration(aptos_val_mean_prob.numpy()[:, 1], aptos_val_labels.numpy())

    # iterate over the 3 datasets that we want to evaluate
    for which_eval in ['drd_train', 'drd_test', 'aptos_whole']:
        which_eval_dict = eval_dict['{}_eval'.format(which_eval)]
        if which_eval == 'drd_train':
            prefix = 'DRD Train'
        elif which_eval == 'drd_test':
            prefix = 'DRD Test'
        else:
            prefix = 'APTOS Test'

        # get predictions and labels
        if which_eval == 'aptos_whole':
            mean_prob = which_eval_dict['mean_prob'][aptos_test_idx]
            # logits = utils.customFunc.probs2logits(mean_prob)
            true_outcome = which_eval_dict['labels'][aptos_test_idx].type(torch.LongTensor)
            # calibrated_mean_prob = aptos_calibrator.calibrate(mean_prob)
            calibrated_prob_of_pos = aptos_calibrator.calibrate(mean_prob.numpy()[:, 1])
            calibrated_mean_prob = np.column_stack((1 - calibrated_prob_of_pos, calibrated_prob_of_pos))
        else:
            mean_prob = which_eval_dict['mean_prob']
            # logits = utils.customFunc.probs2logits(mean_prob)
            true_outcome = which_eval_dict['labels'].type(torch.LongTensor)
            # calibrated_mean_prob = drd_calibrator.calibrate(mean_prob)
            calibrated_prob_of_pos = drd_calibrator.calibrate(mean_prob.numpy()[:, 1])
            calibrated_mean_prob = np.column_stack((1 - calibrated_prob_of_pos, calibrated_prob_of_pos))
        assert mean_prob.shape[1] == 2, 'mean_prob second dim should be 2'

        # compute AUC
        auc_est = utils.customFunc.auc_estimation(y=true_outcome.numpy(), s=mean_prob[:, 1].numpy(), bootstrap=True)
        auc = auc_est['mu']
        auc_b_ci = auc_est['b_ci']
        pd_dict['{} AUC'.format(prefix)].append('{:.3f} ({:.3f}, {:.3f})'.format(auc, auc_b_ci[0], auc_b_ci[1]))

        # compute ECE
        # ece_before = cal.get_ece(probs=mean_prob[:, 1], labels=true_outcome)
        # ece_after = cal.get_ece(probs=calibrated_mean_prob[:, 1], labels=true_outcome)
        ece_before = cal.lower_bound_scaling_ce(probs=mean_prob[:, 1], labels=true_outcome, p=1, debias=False,
                                                mode='top-label', binning_scheme=cal.get_equal_bins)
        ece_after = cal.lower_bound_scaling_ce(probs=calibrated_mean_prob[:, 1], labels=true_outcome, p=1, debias=False,
                                               mode='top-label', binning_scheme=cal.get_equal_bins)
        pd_dict['{} ECE1'.format(prefix)].append('{:.3f}'.format(ece_before))
        pd_dict['{} ECE2'.format(prefix)].append('{:.3f}'.format(ece_after))

        # entropy-based referral
        entropy_before = torch.distributions.categorical.Categorical(mean_prob).entropy()
        entropy_after = torch.distributions.categorical.Categorical(torch.tensor(calibrated_mean_prob)).entropy()
        auc_array_en1 = utils.customFunc.refer_performance(pred_prob=mean_prob, truth=true_outcome,
                                                           uncert=entropy_before)
        # oracle referral
        refer_dict2 = utils.customFunc.refer_performance2(pred_prob=mean_prob, truth=true_outcome, n_bins=50)

        # entropy-based referral after calibration
        auc_array_en3 = utils.customFunc.refer_performance(pred_prob=calibrated_mean_prob, truth=true_outcome,
                                                           uncert=entropy_after)

        # compute Mann-Kendall Tau for entropy-based referral
        trend1, h1, p1, z1, Tau1, s1, var_s1, slope1, intercept1 = mk.original_test(auc_array_en1)
        pd_dict['{} Tau1'.format(prefix)].append('{:.3f}'.format(Tau1))

        # compute Mann-Kendall Tau for oracle referral
        auc_array_en2 = [i[1] for i in refer_dict2['auc']]
        trend2, h2, p2, z2, Tau2, s2, var_s2, slope2, intercept2 = mk.original_test(auc_array_en2)
        pd_dict['{} Tau2'.format(prefix)].append('{:.3f}'.format(Tau2))

        # compute Mann-Kendall Tau for entropy-based referral after calibration
        trend3, h3, p3, z3, Tau3, s3, var_s3, slope3, intercept3 = mk.original_test(auc_array_en3)
        pd_dict['{} Tau3'.format(prefix)].append('{:.3f}'.format(Tau3))

        # compute A for entropy-based referral
        auc0 = auc_array_en1[0].numpy().item()
        area = trapz(y=(auc_array_en1 - auc0).numpy(), dx=0.01)
        a_measure = area / (0.5 * (1 - auc0))
        pd_dict['{} A1'.format(prefix)].append('{:.3f}'.format(a_measure))

        # compute A for oracle referral
        auc0 = auc_array_en2[0]
        p_list = [i[0] for i in refer_dict2['auc']]
        p = max(list(filter(lambda x: x >= 0.5, p_list)))
        area = trapz(y=(np.array(auc_array_en2) - auc0), dx=0.01)
        a_measure = area / (p * (1 - auc0))
        pd_dict['{} A2'.format(prefix)].append('{:.3f}'.format(a_measure))

        # compute A for entropy-based referral after calibration
        auc0 = auc_array_en3[0].numpy().item()
        area = trapz(y=(auc_array_en3 - auc0).numpy(), dx=0.01)
        a_measure = area / (0.5 * (1 - auc0))
        pd_dict['{} A3'.format(prefix)].append('{:.3f}'.format(a_measure))

        # calibration plots before and after calibration
        title = 'Model Calibration ({}, {})'.format(job_id, prefix)
        save = path_dict['current_parent_parent'] + '/results/images/{}_{}_calibration.png'.format(job_id, prefix)
        fraction_of_pos_before, mean_pred_value_before = calibration_curve(y_true=true_outcome, y_prob=mean_prob[:, 1],
                                                                           n_bins=15)
        fraction_of_pos_after, mean_pred_value_after = calibration_curve(y_true=true_outcome,
                                                                         y_prob=calibrated_mean_prob[:, 1], n_bins=15)
        plt.figure(figsize=(10, 5))
        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        ax2 = plt.subplot2grid((3, 1), (2, 0))

        ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        ax1.plot(mean_pred_value_before, fraction_of_pos_before, "s-",
                 label="Before Calibration (ECE={})".format(round(ece_before, 3)))
        ax1.plot(mean_pred_value_after, fraction_of_pos_after, "s-",
                 label="After Calibration (ECE={})".format(round(ece_after, 3)))
        # ax1.text(x=0.5, y=0, s='ECE = {}'.format(round(ece, 4)))
        ax1.set_ylabel("Fraction of positives")
        ax1.set_ylim([-0.05, 1.05])
        ax1.legend(loc="lower right")
        ax1.set_title(title)

        ax2.hist(mean_prob[:, 1], range=(0, 1), bins=10, label='Before Calibration', histtype="step", lw=2,
                 density=False)
        ax2.hist(calibrated_mean_prob[:, 1], range=(0, 1), bins=10, label='After Calibration', histtype="step", lw=2,
                 density=False)
        ax2.set_xlabel("Predicted probability")
        ax2.set_ylabel("Count")
        ax2.legend(loc="upper center", ncol=2)
        plt.tight_layout()
        plt.savefig(save)
        plt.close()

        # plot AUC vs referral
        title = 'AUC vs Referral ({}, {})'.format(job_id, prefix)
        save = path_dict['current_parent_parent'] + '/results/images/{}_{}_auc_vs_referral.png'.format(job_id, prefix)
        ytop = torch.max(torch.cat((auc_array_en1, torch.tensor(auc_array_en2), auc_array_en3))).item()
        ytop = ytop + 0.005 if ytop == 1 else ytop + 0.01
        ybottom = torch.min(torch.cat((auc_array_en1, torch.tensor(auc_array_en2), auc_array_en3))).item() - 0.005
        plt.figure(figsize=(12, 9))
        fontsize = 12
        plt.ylim(top=ytop, bottom=ybottom)
        plt.plot(np.arange(0, 0.51, 0.01), auc_array_en1, color='orange',
                 label='Entropy-based referral (before calibration)')
        plt.plot(p_list, auc_array_en2, color='red', label='Oracle referral')
        plt.plot(np.arange(0, 0.51, 0.01), auc_array_en3, color='blue',label='Entropy-based referral (after calibration)')
        plt.xlabel('Fraction of removed cases', fontsize=fontsize)
        plt.ylabel('Model AUC evaluated on the retained cases', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.legend(loc="lower right")
        plt.title(title, fontsize=fontsize)
        plt.savefig(save)
        plt.close()


def main():
    # get a dictionary of useful directories
    path_dict = utils.customFunc.get_path()

    logging.basicConfig(filename=path_dict['current_parent'] + '/log_DR_analysis.log', level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', filemode='a')

    eval_list = sorted(filter(lambda x: 'archive' not in x, os.listdir(path_dict['evaldict_save'])))

    # eval_dict_name = eval_list[0]
    pd_dict = {'Method': [], 'Network': [], 'Train Size': [], 'Bootstrap': [], 'JobID': [],
               'DRD Train AUC': [],
               'DRD Train ECE1': [],
               'DRD Train ECE2': [],
               'DRD Train Tau1': [],
               'DRD Train Tau2': [],
               'DRD Train Tau3': [],
               'DRD Train A1': [],
               'DRD Train A2': [],
               'DRD Train A3': [],
               'DRD Test AUC': [],
               'DRD Test ECE1': [],
               'DRD Test ECE2': [],
               'DRD Test Tau1': [],
               'DRD Test Tau2': [],
               'DRD Test Tau3': [],
               'DRD Test A1': [],
               'DRD Test A2': [],
               'DRD Test A3': [],
               'APTOS Test AUC': [],
               'APTOS Test ECE1': [],
               'APTOS Test ECE2': [],
               'APTOS Test Tau1': [],
               'APTOS Test Tau2': [],
               'APTOS Test Tau3': [],
               'APTOS Test A1': [],
               'APTOS Test A2': [],
               'APTOS Test A3': []
               }

    for i in eval_list:
        make_tb(eval_dict_name=i, path_dict=path_dict, pd_dict=pd_dict)

    df = pd.DataFrame(pd_dict)
    df.to_csv('{}/results/pd_results.csv'.format(path_dict['current_parent_parent']), index=False, mode='w', header=True)


if __name__ == '__main__':
    main()
