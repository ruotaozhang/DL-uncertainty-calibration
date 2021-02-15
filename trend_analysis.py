import pymannkendall as mk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import seaborn as sns
import utils.customFunc
from scipy import stats

dir_dict = utils.customFunc.get_path()
df = pd.read_csv('{}/results/pd_results.csv'.format(dir_dict['current_parent_parent']))


def set_fontsize(fontsize):
    plt.rc('font', size=fontsize)  # controls default text sizes
    plt.rc('axes', titlesize=fontsize)  # fontsize of the axes title
    plt.rc('axes', labelsize=fontsize)  # fontsize of the x and y labels
    plt.rc('legend', fontsize=fontsize)  # legend fontsize
    plt.rc('figure', titlesize=fontsize)  # fontsize of the figure title
    plt.rc('xtick', labelsize=fontsize)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=fontsize)  # fontsize of the tick labels


#####################
save = dir_dict['current_parent_parent'] + '/results/images/ece_vs_ar_drd_train.png'
plt.figure(figsize=(12, 9))
set_fontsize(42)
plt.scatter(df['DRD Train A1'], df['DRD Train ECE1'], s=80)
plt.xlabel('Area Ratio (AR) Statistic')
plt.ylabel('ECE')
plt.tight_layout()
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
plt.savefig(save)
not_na_mask = pd.notna(df['DRD Train A1'])
stats.pearsonr(df['DRD Train A1'][not_na_mask], df['DRD Train ECE1'][not_na_mask])

save = dir_dict['current_parent_parent'] + '/results/images/ece_vs_ar_drd_test.png'
plt.figure(figsize=(12, 9))
set_fontsize(42)
plt.scatter(df['DRD Test A1'], df['DRD Test ECE1'], s=80)
plt.xlabel('Area Ratio (AR) Statistic')
plt.ylabel('ECE')
plt.tight_layout()
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
plt.savefig(save)
not_na_mask = pd.notna(df['DRD Test A1'])
stats.pearsonr(df['DRD Test A1'][not_na_mask], df['DRD Test ECE1'][not_na_mask])

save = dir_dict['current_parent_parent'] + '/results/images/ece_vs_ar_aptos_test.png'
plt.figure(figsize=(12, 9))
set_fontsize(42)
plt.scatter(df['APTOS Test A1'], df['APTOS Test ECE1'], s=80)
plt.xlabel('Area Ratio (AR) Statistic')
plt.ylabel('ECE')
plt.tight_layout()
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
plt.savefig(save)
not_na_mask = pd.notna(df['APTOS Test A1'])
stats.pearsonr(df['APTOS Test A1'][not_na_mask], df['APTOS Test ECE1'][not_na_mask])

#####################
fontsize = 42
save = dir_dict['current_parent_parent'] + '/results/images/aptos_test_tau_comparison.png'
set_fontsize(fontsize)
plt.figure(figsize=(12, 9))
sns.kdeplot(df['APTOS Test Tau1'], shade=True, color="r", clip=(-1, 1),
            label='Before calibration', legend=False)
sns.kdeplot(df['APTOS Test Tau2'], shade=True, color="b", clip=(-1, 1), label='Oracle referral', legend=False)
sns.kdeplot(df['APTOS Test Tau3'], shade=True, color="orange", clip=(-1, 1),
                       label='After calibration', legend=False)
plt.xlabel('MK')
plt.ylabel('Density')
plt.tight_layout()
plt.savefig(save)

save = dir_dict['current_parent_parent'] + '/results/images/drd_test_tau_comparison.png'
set_fontsize(fontsize)
plt.figure(figsize=(12, 9))
sns.kdeplot(df['DRD Test Tau1'], shade=True, color="r", clip=(-1, 1),
            label='Before calibration', legend=False)
sns.kdeplot(df['DRD Test Tau2'], shade=True, color="b", clip=(-1, 1), label='Oracle referral', legend=False)
sns.kdeplot(df['DRD Test Tau3'], shade=True, color="orange", clip=(-1, 1),
                       label='After calibration', legend=False)
plt.xlabel('MK')
plt.ylabel('Density')
plt.tight_layout()
plt.savefig(save)

save = dir_dict['current_parent_parent'] + '/results/images/drd_train_tau_comparison.png'
set_fontsize(fontsize)
plt.figure(figsize=(12, 9))
sns.kdeplot(df['DRD Train Tau1'], shade=True, color="r", clip=(-1, 1),
            label='Before calibration', legend=False)
sns.kdeplot(df['DRD Train Tau2'], shade=True, color="b", clip=(-1, 1), label='Oracle referral', legend=False)
sns_plot = sns.kdeplot(df['DRD Train Tau3'], shade=True, color="orange", clip=(-1, 1),
                       label='After calibration', legend=False)
plt.xlabel('MK')
plt.ylabel('Density')
plt.tight_layout()
plt.savefig(save)

#####################
save = dir_dict['current_parent_parent'] + '/results/images/drd_train_ar_comparison.png'
set_fontsize(fontsize)
plt.figure(figsize=(12, 9))
sns.kdeplot(df['DRD Train A1'], shade=True, color="r", clip=(-np.inf, 1),
            label='Before calibration', legend=False)
sns.kdeplot(df['DRD Train A2'], shade=True, color="b", clip=(-np.inf, 1), label='Oracle referral', legend=False)
sns.kdeplot(df['DRD Train A3'], shade=True, color="orange", clip=(-np.inf, 1),
                       label='After calibration', legend=False)
plt.xlabel('AR')
plt.ylabel('Density')
plt.tight_layout()
plt.savefig(save)

save = dir_dict['current_parent_parent'] + '/results/images/drd_test_ar_comparison.png'
set_fontsize(fontsize)
plt.figure(figsize=(12, 9))
sns.kdeplot(df['DRD Test A1'], shade=True, color="r", clip=(-np.inf, 1),
            label='Before calibration', legend=False)
sns.kdeplot(df['DRD Test A2'], shade=True, color="b", clip=(-np.inf, 1), label='Oracle referral', legend=False)
sns.kdeplot(df['DRD Test A3'], shade=True, color="orange", clip=(-np.inf, 1),
                       label='After calibration', legend=False)
plt.xlabel('AR')
plt.ylabel('Density')
plt.tight_layout()
plt.savefig(save)

save = dir_dict['current_parent_parent'] + '/results/images/aptos_test_ar_comparison.png'
set_fontsize(fontsize)
plt.figure(figsize=(12, 9))
sns.kdeplot(df['APTOS Test A1'], shade=True, color="r", clip=(-np.inf, 1),
            label='Before calibration', legend=False)
sns.kdeplot(df['APTOS Test A2'], shade=True, color="b", clip=(-np.inf, 1), label='Oracle referral', legend=False)
sns.kdeplot(df['APTOS Test A3'], shade=True, color="orange", clip=(-np.inf, 1),
                       label='After calibration', legend=False)
plt.xlabel('AR')
plt.ylabel('Density')
plt.tight_layout()
plt.savefig(save)

#####################

drd_train_aucs = [float(a[:5]) for a in df['DRD Train AUC'].to_list()]
drd_test_aucs = [float(a[:5]) for a in df['DRD Test AUC'].to_list()]
aptos_test_aucs = [float(a[:5]) for a in df['APTOS Test AUC'].to_list()]

stats.pearsonr(drd_train_aucs, df['DRD Train ECE1'])
stats.pearsonr(drd_test_aucs, df['DRD Test ECE1'])
stats.pearsonr(aptos_test_aucs, df['APTOS Test ECE1'])

networks = df['Network'].to_list() * 3
eces = df['DRD Train ECE1'].to_list() + df['DRD Test ECE1'].to_list() + df['APTOS Test ECE1'].to_list()
df2 = pd.DataFrame({'Network': networks, 'ECE': eces})
df2.boxplot(column=['ECE'], by=['Network'])
df.boxplot(column=['DRD Test ECE1'], by=['Network'])

#####################

sns.kdeplot(df['DRD Test A1'], shade=True, color="r", clip=(-np.inf, 1))
sns.kdeplot(df['DRD Test A2'], shade=True, color="b", clip=(-np.inf, 1))
sns.kdeplot(df['DRD Test A3'], shade=True, color="orange", clip=(-np.inf, 1))

sns.kdeplot(df['DRD Train A1'], shade=True, color="r", clip=(-np.inf, 1))
sns.kdeplot(df['DRD Train A2'], shade=True, color="b", clip=(-np.inf, 1))
sns.kdeplot(df['DRD Train A3'], shade=True, color="orange", clip=(-np.inf, 1))

###########################
drd_train_eval_dict = eval_dict['drd_train_eval']
drd_train_mean_prob = drd_train_eval_dict['mean_prob']
drd_train_true_outcome = drd_train_eval_dict['labels'].type(torch.LongTensor)
drd_train_entropy_before = torch.distributions.categorical.Categorical(drd_train_mean_prob).entropy()
drd_train_auc_array_en1 = utils.customFunc.refer_performance(pred_prob=drd_train_mean_prob,
                                                             truth=drd_train_true_outcome,
                                                             uncert=drd_train_entropy_before)

drd_test_eval_dict = eval_dict['drd_test_eval']
drd_test_mean_prob = drd_test_eval_dict['mean_prob']
drd_test_true_outcome = drd_test_eval_dict['labels'].type(torch.LongTensor)
drd_test_entropy_before = torch.distributions.categorical.Categorical(drd_test_mean_prob).entropy()
drd_test_auc_array_en1 = utils.customFunc.refer_performance(pred_prob=drd_test_mean_prob, truth=drd_test_true_outcome,
                                                            uncert=drd_test_entropy_before)

aptos_test_eval_dict = eval_dict['aptos_whole_eval']
aptos_test_mean_prob = aptos_test_eval_dict['mean_prob'][aptos_test_idx]
aptos_test_true_outcome = aptos_test_eval_dict['labels'][aptos_test_idx].type(torch.LongTensor)
aptos_test_entropy_before = torch.distributions.categorical.Categorical(aptos_test_mean_prob).entropy()
aptos_test_auc_array_en1 = utils.customFunc.refer_performance(pred_prob=aptos_test_mean_prob,
                                                              truth=aptos_test_true_outcome,
                                                              uncert=aptos_test_entropy_before)

# plot AUC vs referral
title = 'Uncertainty-based Referral'
save = dir_dict['current_parent'] + '/results/images/{}_referral_drd_test_aptos_test.png'.format(job_id)
ytop = torch.max(torch.cat((drd_test_auc_array_en1, aptos_test_auc_array_en1))).item()
ytop = ytop + 0.005 if ytop == 1 else ytop + 0.01
ybottom = torch.min(torch.cat((drd_test_auc_array_en1, aptos_test_auc_array_en1))).item() - 0.005
fig = plt.figure(figsize=(12, 12))
fontsize = 28
plt.ylim(top=ytop, bottom=ybottom)
# plt.plot(np.arange(0, 0.51, 0.01), drd_train_auc_array_en1, color='orange',label='DRD Train')
plt.plot(np.arange(0, 0.51, 0.01), drd_test_auc_array_en1, color='blue',
         label='DRD Test (Tau=0.853, AR=0.12)')
plt.plot(np.arange(0, 0.51, 0.01), aptos_test_auc_array_en1, color='red',
         label='APTOS Test (Tau=-0.94, AR=-0.143)')
plt.xlabel('Fraction of removed cases', fontsize=fontsize)
plt.ylabel('Model AUC evaluated on the retained cases', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.legend(loc="upper right", fontsize=fontsize)
plt.title(title, fontsize=fontsize)
plt.savefig(save)
plt.close()

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('xtick', labelsize=fontsize)  # fontsize of the tick labels
plt.rc('ytick', labelsize=fontsize)  # fontsize of the tick labels

title = 'Model Calibration'
save = dir_dict['current_parent'] + '/results/images/{}_aptos_testcalibration.png'.format(job_id)
fraction_of_pos_before, mean_pred_value_before = calibration_curve(y_true=aptos_test_true_outcome,
                                                                   y_prob=aptos_test_mean_prob[:, 1], n_bins=15)
plt.figure(figsize=(12, 12))
fontsize = 28
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((3, 1), (2, 0))
ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
ax1.plot(mean_pred_value_before, fraction_of_pos_before, "s-", label="ECE=0.121")
# ax1.text(x=0.5, y=0, s='ECE = {}'.format(round(ece, 4)))
ax1.set_ylabel("Fraction of positives", fontsize=fontsize)
ax1.set_ylim([-0.05, 1.05])
ax1.legend(loc="lower right", fontsize=26)
ax1.set_title(title, fontsize=fontsize)
ax2.hist(aptos_test_mean_prob[:, 1], range=(0, 1), bins=10, histtype="step", lw=2,
         density=False)
ax2.set_xlabel("Predicted probability", fontsize=fontsize)
ax2.set_ylabel("Count", fontsize=fontsize)
# ax2.legend(loc="upper center", ncol=2)
plt.tight_layout()
plt.savefig(save)

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
        logits = utils.customFunc.probs2logits(mean_prob)
        true_outcome = which_eval_dict['labels'][aptos_test_idx].type(torch.LongTensor)
        # calibrated_mean_prob = aptos_calibrator.calibrate(mean_prob)
        calibrated_prob_of_pos = aptos_calibrator.calibrate(mean_prob.numpy()[:, 1])
        calibrated_mean_prob = np.column_stack((1 - calibrated_prob_of_pos, calibrated_prob_of_pos))
    else:
        mean_prob = which_eval_dict['mean_prob']
        logits = utils.customFunc.probs2logits(mean_prob)
        true_outcome = which_eval_dict['labels'].type(torch.LongTensor)
        # calibrated_mean_prob = drd_calibrator.calibrate(mean_prob)
        calibrated_prob_of_pos = drd_calibrator.calibrate(mean_prob.numpy()[:, 1])
        calibrated_mean_prob = np.column_stack((1 - calibrated_prob_of_pos, calibrated_prob_of_pos))
    assert mean_prob.shape[1] == 2, 'mean_prob second dim should be 2'

    # compute AUC
    auc_est = utils.customFunc.auc_estimation(y=true_outcome.numpy(), s=mean_prob[:, 1].numpy(), bootstrap=True)
    auc = auc_est['mu']
    auc_b_ci = auc_est['b_ci']
    pd_dict['{} AUC'.format(prefix)].append(
        '{} ({}, {})'.format(round(auc, 3), round(auc_b_ci[0], 3), round(auc_b_ci[1], 3)))

    # compute ECE
    # ece_before = cal.get_ece(probs=mean_prob[:, 1], labels=true_outcome)
    # ece_after = cal.get_ece(probs=calibrated_mean_prob[:, 1], labels=true_outcome)
    ece_before = cal.lower_bound_scaling_ce(probs=mean_prob[:, 1], labels=true_outcome, p=1, debias=False,
                                            mode='top-label', binning_scheme=cal.get_equal_bins)
    ece_after = cal.lower_bound_scaling_ce(probs=calibrated_mean_prob[:, 1], labels=true_outcome, p=1, debias=False,
                                           mode='top-label', binning_scheme=cal.get_equal_bins)
    pd_dict['{} ECE1'.format(prefix)].append(round(ece_before, 3))
    pd_dict['{} ECE2'.format(prefix)].append(round(ece_before, 3))

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
    pd_dict['{} Tau1'.format(prefix)].append('{}'.format(round(Tau1, 3)))

    # compute Mann-Kendall Tau for oracle referral
    auc_array_en2 = [i[1] for i in refer_dict2['auc']]
    trend2, h2, p2, z2, Tau2, s2, var_s2, slope2, intercept2 = mk.original_test(auc_array_en2)
    pd_dict['{} Tau2'.format(prefix)].append('{}'.format(round(Tau2, 3)))

    # compute Mann-Kendall Tau for entropy-based referral after calibration
    trend3, h3, p3, z3, Tau3, s3, var_s3, slope3, intercept3 = mk.original_test(auc_array_en3)
    pd_dict['{} Tau3'.format(prefix)].append('{}'.format(round(Tau3, 3)))

    # compute A for entropy-based referral
    auc0 = auc_array_en1[0].numpy().item()
    area = trapz(y=(auc_array_en1 - auc0).numpy(), dx=0.01)
    a_measure = area / (0.5 * (1 - auc0))
    pd_dict['{} A1'.format(prefix)].append('{}'.format(round(a_measure, 3)))

    # compute A for oracle referral
    auc0 = auc_array_en2[0]
    p_list = [i[0] for i in refer_dict2['auc']]
    p = max(list(filter(lambda x: x >= 0.5, p_list)))
    area = trapz(y=(np.array(auc_array_en2) - auc0), dx=0.01)
    a_measure = area / (p * (1 - auc0))
    pd_dict['{} A2'.format(prefix)].append('{}'.format(round(a_measure, 3)))

    # compute A for entropy-based referral after calibration
    auc0 = auc_array_en3[0].numpy().item()
    area = trapz(y=(auc_array_en3 - auc0).numpy(), dx=0.01)
    a_measure = area / (0.5 * (1 - auc0))
    pd_dict['{} A3'.format(prefix)].append('{}'.format(round(a_measure, 3)))

    # calibration plots before and after calibration
    save = path_dict['current_parent_parent'] + '/results/images/{}_{}_calibration.png'.format(job_id, prefix)
    fraction_of_pos_before, mean_pred_value_before = calibration_curve(y_true=true_outcome, y_prob=mean_prob[:, 1],
                                                                       n_bins=15)
    plt.figure(figsize=(12, 9))
    set_fontsize(40)
    plt.plot([0, 1], [0, 1], "k:", label="Perfect calibration")
    plt.plot(mean_pred_value_before, fraction_of_pos_before, "s-", label="ECE={}".format(round(ece_before, 2)))
    # ax1.text(x=0.5, y=0, s='ECE = {}'.format(round(ece, 4)))
    plt.xlabel("Predicted probability")
    plt.ylabel('Fraction of positives')
    plt.ylim(bottom=-0.05, top=1.05)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(save)
    plt.close()

    # plot AUC vs referral
    save = path_dict['current_parent_parent'] + '/results/images/{}_{}_auc_vs_referral.png'.format(job_id, prefix)
    ytop = torch.max(auc_array_en1).item()
    ytop = ytop + 0.005 if ytop == 1 else ytop + 0.01
    ybottom = torch.min(auc_array_en1).item() - 0.005
    plt.figure(figsize=(12, 9))
    set_fontsize(42)
    plt.ylim(top=ytop, bottom=ybottom)
    plt.plot(np.arange(0, 0.51, 0.01), auc_array_en1, color='black')
    plt.xlabel('Fraction of removed cases')
    plt.ylabel('AUC')
    plt.tight_layout()
    plt.savefig(save)
    plt.close()
