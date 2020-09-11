
# Script to compute spectra and run FOOOF toolbox to get frequency and amplitude of theta
# author: Mehdi Senoussi
# date: 02/02/2020


import mne
import numpy as np
from matplotlib import pyplot as pl
from scipy import signal as sig
from scipy import stats
from statsmodels.stats.anova import AnovaRM

from mne.time_frequency import tfr_morlet, psd_multitaper, psd_welch
from fooof import FOOOF, FOOOFGroup
import pandas as pd

from mne.channels import find_ch_connectivity
from mne.stats import permutation_cluster_test, spatio_temporal_cluster_test
from scipy import stats


# loads the channel file
data_path = '/Volumes/mehdimac/ghent/mystinfo/gitcleandata/'
n_elec = 64

insttxts = np.array(['LL', 'LR', 'RL', 'RR'])
inst_diff_order = np.array([3, 0, 2, 1], dtype=np.int)

#### load epochs

obs_all = np.arange(1, 40)
## excluded participants
# obs 5 and 15 have less than 5 blocks, obs 9 left-handed
# obs 16, 23 and 33 have less than 200 trials after rejection based on EyeT
obs_all = obs_all[np.array([obs_i not in [5, 9, 15, 16, 23, 33] for obs_i in obs_all])]

eps = mne.read_epochs(fname = data_path + 'obs_1/eeg/obs1_allclean_peri-stim_pres_data_filt-None-48-epo.fif.gz',
	proj = False, verbose= 50, preload=True)

# obs_all = obs_all[:-4]

# get data and group by instruction, correct/incorrect, etc.
n_obs = len(obs_all)

avg_theta_peak_byInstCorr = np.zeros(shape = [n_obs, 4, 2, 64])
avg_theta_amp_byInstCorr = np.zeros(shape = [n_obs, 4, 2, 64])

theta_par_all = []; acc_all = [];inst_all = []
for obs_ind, obs_i in enumerate(obs_all):
	print('loading obs %i' % obs_i)
	obs_eegpath = data_path + 'obs_%i/eeg/' % obs_i
	z = np.load(obs_eegpath +\
			'obs_%i_theta_fooof_freqlim_2-20Hz.npz' % obs_i,
			allow_pickle=True)['arr_0'][..., np.newaxis][0]

	theta_par_all.append(z['theta_params_clean'].astype(np.float).copy())
	acc_all.append(z['respcorrect'].astype(np.int).copy())
	inst_all.append(z['instr_type'].astype(np.int).copy())

	mask_corr = acc_all[obs_ind]==1
	for instr in np.arange(4):
		mask_instr = inst_all[obs_ind] == instr

		for elec_n in np.arange(64):
			theta_there_mask = theta_par_all[obs_ind][elec_n, :, 0] != -1
			theta_params_theta_there = theta_par_all[obs_ind][elec_n, theta_there_mask, :]
			instr_theta_there = mask_instr[theta_there_mask]
			corr_theta_there_instr =\
				acc_all[obs_ind][theta_there_mask][instr_theta_there].astype(np.int)
			avg_theta_peak_byInstCorr[obs_ind, instr, 0, elec_n] =\
				theta_params_theta_there[instr_theta_there, 0][corr_theta_there_instr == 1].mean()
			avg_theta_peak_byInstCorr[obs_ind, instr, 1, elec_n] =\
				theta_params_theta_there[instr_theta_there, 0][corr_theta_there_instr == 0].mean()

			amp_temp = theta_par_all[obs_ind][elec_n, :, 1].copy()
			amp_temp[amp_temp==-1] = 0
			avg_theta_amp_byInstCorr[obs_ind, instr, 0, elec_n] =\
				amp_temp[mask_instr & (acc_all[obs_ind] == 1)].mean()
			avg_theta_amp_byInstCorr[obs_ind, instr, 1, elec_n] =\
				amp_temp[mask_instr & (acc_all[obs_ind] == 0)].mean()


toplot1 = avg_theta_amp_byInstCorr[:, :, 0, :]
toplot1 = (toplot1 - toplot1.mean(axis=2)[:, :, np.newaxis])\
			/toplot1.std(axis=2)[:, :, np.newaxis]
toplot2 = avg_theta_amp_byInstCorr[:, :, 1, :]
toplot2 = (toplot2 - toplot2.mean(axis=2)[:, :, np.newaxis])\
		/toplot2.std(axis=2)[:, :, np.newaxis]
toplot = (toplot1 - toplot2).mean(axis=1)


ch_idx = eps.info['ch_names'].index('AFz')
connectivity, ch_names = find_ch_connectivity(eps.info, ch_type='eeg')

p_accept = .05
threshold = stats.distributions.t.ppf(1 - .05, n_obs - 1)

# compute clusters
T_obs, clusters, p_values, _ =\
	mne.stats.permutation_cluster_1samp_test(X=toplot, n_permutations=10000,
		threshold=threshold, tail=1, n_jobs=-1, buffer_size=None,
		connectivity=connectivity)
good_cluster_inds = np.where(p_values < p_accept)[0]
mask = np.array(clusters)[good_cluster_inds].squeeze()

fig, axs = pl.subplots(1, 3)
t, p = stats.ttest_1samp(toplot, popmean=0, axis=0)
im1, cn = mne.viz.plot_topomap(t, eps.info, cmap='RdBu_r',#'coolwarm',#'Blues_r',
	sensors=False, outlines='head', extrapolate ='local',
	head_pos = {'scale':[1.3, 1.7]}, axes=axs[0], vmin=-3, vmax=3,
	contours=2, mask=mask)
axs[0].set_title('Topo of common elecs\nacross inst clusters', fontsize=10)

fig.colorbar(im1, ax = axs[0])
zscore_theta_peak_byInstCorr = avg_theta_peak_byInstCorr.copy() -\
	np.nanmean(avg_theta_peak_byInstCorr, axis=1)[:, np.newaxis, :, :]
std_temp = np.nanstd(avg_theta_peak_byInstCorr, axis=1)[:, np.newaxis, :, :]
std_temp[np.isnan(std_temp) | (std_temp==0)] = 1
zscore_theta_peak_byInstCorr /= std_temp

ch_inds = np.array([eps.info['ch_names'].index(ch_name)\
	for ch_name in np.array(ch_names)[mask]])

corr_names = ['Correct', 'Incorrect']
for corr_ind, corr_i in enumerate([0, 1]):
	xs = np.arange(4)+(corr_i-1)/8
	toplot = np.nanmean(zscore_theta_peak_byInstCorr[:, inst_diff_order,corr_i,:][:,:, ch_inds],
		axis=-1)
	# for obs_ind in np.arange(n_obs): ax.plot(xs, toplot[obs_ind, :], 'o-', color=[.6, .6, .6, .6])
	axs[1].errorbar(x=xs, y=np.nanmean(toplot, axis=0), yerr = np.nanstd(toplot, axis=0)/n_obs**.5,
		fmt=['o', '^'][corr_ind], ms=10, color=['g', 'r'][corr_i], label=corr_names[corr_i])
axs[1].set_xticks(np.arange(4)); axs[1].set_xticklabels(insttxts[inst_diff_order])
axs[1].legend(loc='best'); axs[1].grid()

bla = np.nanmean(zscore_theta_peak_byInstCorr[:, inst_diff_order, 0, :][:,:, ch_inds], axis=-1)
subjs = np.repeat(obs_all, 4)
hand = np.tile(np.array([1, 2, 1, 2]), n_obs)
hemi = np.tile(np.array([1, 2, 2, 1]), n_obs)
df = pd.DataFrame({'obs':subjs, 'hand':hand, 'hemi':hemi,
			'theta_peak':bla.flatten()})

aovrm = AnovaRM(data=df, depvar='theta_peak', subject='obs',
	within=['hand', 'hemi'], aggregate_func=np.mean)
res = aovrm.fit()
print(res.anova_table)
fvals = res.anova_table.values[:, 0]
pvals = res.anova_table.values[:, -1]

axs[1].set_title('Theta Peak - %s\nF=%s\np=%s)'%\
	(data_format, np.str(fvals), np.str(pvals)), fontsize=8)
axs[1].hlines(y=0, xmin=-.5, xmax=4.5)
axs[1].set_xlim(-1, 4)


# plot diff correct - incorrect
toplot = np.nanmean(zscore_theta_peak_byInstCorr[:, inst_diff_order, 0, :][:,:, ch_inds]\
	-zscore_theta_peak_byInstCorr[:, inst_diff_order, 1, :][:, :, ch_inds], axis=-1)
axs[2].errorbar(x=np.arange(4), y=np.nanmean(toplot, axis=0), yerr = np.nanstd(toplot, axis=0)/n_obs**.5,
	fmt='^', ms=10, color='k')

subjs = np.repeat(obs_all, 4)
hand = np.tile(np.array([1, 2, 1, 2]), n_obs)
hemi = np.tile(np.array([1, 2, 2, 1]), n_obs)
df = pd.DataFrame({'obs':subjs, 'hand':hand, 'hemi':hemi,
			'theta_peak':toplot.flatten()})
aovrm = AnovaRM(data=df, depvar='theta_peak', subject='obs', within=['hand', 'hemi'], aggregate_func=np.mean)
res = aovrm.fit()
fvals = res.anova_table.values[:, 0]
pvals = res.anova_table.values[:, -1]
axs[2].set_title('Theta Peak diff (corr-incorr) - %s\nF=%s\np=%s)'%\
	(data_format, np.str(fvals), np.str(pvals)), fontsize=8)
axs[2].grid(); axs[2].hlines(y=0, xmin=-.5, xmax=4.5)
axs[2].set_xlim(-1, 4)

fig.set_size_inches(10, 4)
pl.tight_layout()









###### SLOPE OF THETA SHIFT
corr_all_inst = np.load(data_path + 'corr_byInst_all_forEEG.npy')

overall_acc_per_inst = corr_all_inst.mean(axis=0)[inst_diff_order]*100

lin_rel_peak_inst = np.zeros(shape = [n_obs, 2, 2])
for obs_ind in np.arange(n_obs):
	for corr_ind in np.arange(2):
		theta_peaks = zscore_theta_peak_byInstCorr[:, inst_diff_order, corr_ind, :][obs_ind, :, ch_inds]
		theta_peaks = np.nanmean(theta_peaks, axis=0)
		lin_rel_peak_inst[obs_ind, corr_ind, :] = np.polyfit(overall_acc_per_inst, theta_peaks, 1)

# inter-individual differences in accuracy linked to theta peak shift?
corr_all = np.load(data_path + 'corr_all_forEEG.npy')

xs = np.array([-.6, .6])
x = lin_rel_peak_inst[:, 0, 0]
y = corr_all*100
correl_eeg_peak_acc = np.polyfit(x, y, 1)
pl.figure()
pl.plot(x, y, 'o', ms=10, mec=[1,1,1], mfc=[.7, .7, .7], mew=1.5)
pl.plot(xs, correl_eeg_peak_acc[0]*xs + correl_eeg_peak_acc[1], 'k--')
rho, pval = stats.spearmanr(x, y)
pl.title('Correlation: accuracy - theta peak slope across instructions\nrho=%.2f, p=%.4f' % (rho, pval), fontsize=10)
pl.xlim([-.22, .22]); pl.ylim([54, 86])
pl.gcf().set_size_inches([4.66, 4.82])
pl.xlabel('Slope of theta increase with task "easiness"')
pl.ylabel('Overall accuracy (%)')



