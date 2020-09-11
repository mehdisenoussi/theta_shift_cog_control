# Script to compute spectra and run FOOOF toolbox to get frequency and amplitude of theta
# author: Mehdi Senoussi
# date: 02/02/2020


import mne
import numpy as np
from matplotlib import pyplot as pl
from scipy import signal as sig
from scipy import stats

from mne.time_frequency import tfr_morlet, psd_multitaper, psd_welch
from fooof import FOOOF, FOOOFGroup
import pandas as pd


# loads the channel file
data_path = '/Volumes/mehdimac/ghent/mystinfo/gitcleandata/'
montage = mne.channels.read_montage(data_path + 'chanlocs_66elecs_noCz.loc')


insttxts = np.array(['LL', 'LR', 'RL', 'RR'])
inst_diff_order = np.array([3, 0, 2, 1], dtype=np.int)

#### load epochs

obs_all = np.arange(1, 40)
## excluded participants
# obs 5 and 15 have less than 5 blocks, obs 9 left-handed
# obs 16, 23 and 33 have less than 200 trials after rejection based on EyeT
obs_all = obs_all[np.array([obs_i not in [5, 9, 15, 16, 23, 33] for obs_i in obs_all])]

# obs_all = np.array([ \
# 	#1,  2,  4,  6,  7,  8, 10, 12, 13, 14, 17, 18, 19, 20, 22, 24,
#        # 25, 26, 29, 31, 32, 34, 35, 37, 38])
# 		36, 39])

n_elec = 64

eps_all = []
psds_welch_mean_all = np.zeros(shape = [len(obs_all), 40, 64])
for obs_ind, obs_i in enumerate(obs_all):
	print('obs %i'%obs_i)
	eps = mne.read_epochs(fname = data_path + 'obs_%i/eeg/obs%i_allclean_peri-stim_pres_data_filt-None-48-epo.fif.gz' % (obs_i, obs_i), proj = False, verbose= 50, preload=True)
	eps_all.append(eps)


# z = np.load('/Users/mehdi/work/ghent/mystinfo/results/theta_peak_zscore_ddm_35obs.npz', allow_pickle=True)['arr_0'][..., np.newaxis][0]
# theta_peak_zscore_ddm = z['theta_peak_zscore_ddm']
# z = np.load('/Users/mehdi/work/ghent/mystinfo/results/theta_peak_classic_35obs.npz', allow_pickle=True)['arr_0'][..., np.newaxis][0]
# theta_peak_classic = z['theta_peak_classic']
# z = np.load('/Users/mehdi/work/ghent/mystinfo/results/overall_driftrate_35obs.npz', allow_pickle=True)['arr_0'][..., np.newaxis][0]
# v = z['overall_driftrate']


for obs_ind, obs_i in enumerate(obs_all):
	# create observer's fit results' list
	all_fits_obs_n = []
	obs_eegpath = data_path + 'obs_%i/eeg/' % obs_i
			
	print(obs_i)
	eps = eps_all[obs_ind]
	freqs_welch, amp_welch =  sig.welch(x = eps.get_data(), axis = -1,
		fs = 200, window='hann', average = 'mean', nperseg=200,
		nfft=400, return_onesided=True)
	theta_params_clean = np.zeros(shape = [64, len(eps), 3]) - 1
	all_fits = []
	for elec_n in np.arange(64):
		if np.logical_not(elec_n % 10): print('\telec : %s' % eps.info['ch_names'][elec_n])
		fg = FOOOFGroup(verbose = False, max_n_peaks = 4, peak_width_limits = [.5, 2])
		fg.fit(freqs_welch, amp_welch[:, elec_n, :], [2, 20], n_jobs=-1)
		all_fits.append(fg.get_results())

		theta_params =\
			np.array([fg_res_n.peak_params[(fg_res_n.peak_params[:,0]>3)\
				& (fg_res_n.peak_params[:,0]<8)] for fg_res_n in fg.get_results()])
		for trial_n in np.arange(len(theta_params)):
			if len(theta_params[trial_n]) == 1:
				theta_params_clean[elec_n, trial_n, :] = theta_params[trial_n].squeeze()
			elif len(theta_params[trial_n]) > 1:
				max_amp_ind = np.argmax(theta_params[trial_n][:, 1]).squeeze()
				theta_params_clean[elec_n, trial_n, :] = theta_params[trial_n][max_amp_ind, :].squeeze()
	np.savez(obs_eegpath +\
		'obs_%i_theta_fooof_freqlim_2-20Hz.npz' % obs_i,
		{'theta_params_clean':theta_params_clean, 'instr_type':eps.metadata['instr_type'].values,
		'respcorrect':eps.metadata['respcorrect'].values, 'resptime':eps.metadata['resptime'].values})
	


eps = mne.read_epochs(fname = data_path + 'obs_1/eeg/obs1_allclean_peri-stim_pres_data_filt-None-48-epo.fif.gz',
	proj = False, verbose= 50, preload=True)

# obs_all = obs_all[:-4]

# get data and group by instruction, correct/incorrect, etc.
n_obs = len(obs_all)

avg_theta_peak_byInstCorr = np.zeros(shape = [n_obs, 4, 3, 64])
std_theta_peak_byInstCorr = np.zeros(shape = [n_obs, 4, 3, 64])

avg_theta_amp_byInstCorr = np.zeros(shape = [n_obs, 4, 3, 64])
prop_theta_peak_found = np.zeros(shape = [n_obs, 4, 3, 64])

theta_par_all = []; acc_all = []; rt_all = []; inst_all = []
for obs_ind, obs_i in enumerate(obs_all):
	obs_eegpath = data_path + 'obs_%i/eeg/' % obs_i
	z = np.load(obs_eegpath +\
			'obs_%i_theta_fooof_freqlim_2-20Hz.npz' % obs_i,
			allow_pickle=True)['arr_0'][..., np.newaxis][0]

	theta_par_all.append(z['theta_params_clean'].astype(np.float).copy())
	acc_all.append(z['respcorrect'].astype(np.int).copy())
	rt_all.append(z['resptime'].astype(np.float).copy())
	inst_all.append(z['instr_type'].astype(np.int).copy())

	mask_corr = acc_all[obs_ind]==1
	for instr in np.arange(4):
		mask_instr = inst_all[obs_ind] == instr
		prop_theta_peak_found[obs_ind, instr, :, :] =\
			(theta_par_all[obs_ind][:, mask_instr, 0] > 0).astype(np.int).mean(axis=1),\
			(theta_par_all[obs_ind][:, mask_instr&mask_corr, 0] > 0).astype(np.int).mean(axis=1),\
			(theta_par_all[obs_ind][:, mask_instr&np.logical_not(mask_corr), 0] > 0).astype(np.int).mean(axis=1)

		for elec_n in np.arange(64):
			theta_there_mask = theta_par_all[obs_ind][elec_n, :, 0] != -1
			theta_params_theta_there = theta_par_all[obs_ind][elec_n, theta_there_mask, :]
			instr_theta_there = mask_instr[theta_there_mask]
			corr_theta_there_instr =\
				acc_all[obs_ind][theta_there_mask][instr_theta_there].astype(np.int)
			avg_theta_peak_byInstCorr[obs_ind, instr, 0, elec_n] =\
				theta_params_theta_there[instr_theta_there, 0].mean()
			avg_theta_peak_byInstCorr[obs_ind, instr, 1, elec_n] =\
				theta_params_theta_there[instr_theta_there, 0][corr_theta_there_instr == 1].mean()
			avg_theta_peak_byInstCorr[obs_ind, instr, 2, elec_n] =\
				theta_params_theta_there[instr_theta_there, 0][corr_theta_there_instr == 0].mean()

			
			std_theta_peak_byInstCorr[obs_ind, instr, 0, elec_n] =\
				theta_params_theta_there[instr_theta_there, 0].std()
			std_theta_peak_byInstCorr[obs_ind, instr, 1, elec_n] =\
				theta_params_theta_there[instr_theta_there, 0][corr_theta_there_instr == 1].std()
			std_theta_peak_byInstCorr[obs_ind, instr, 2, elec_n] =\
				theta_params_theta_there[instr_theta_there, 0][corr_theta_there_instr == 0].std()


			amp_temp = theta_par_all[obs_ind][elec_n, :, 1].copy()
			amp_temp[amp_temp==-1] = 0
			avg_theta_amp_byInstCorr[obs_ind, instr, 0, elec_n] = amp_temp[mask_instr].mean()
			avg_theta_amp_byInstCorr[obs_ind, instr, 1, elec_n] =\
				amp_temp[mask_instr & (acc_all[obs_ind] == 1)].mean()
			avg_theta_amp_byInstCorr[obs_ind, instr, 2, elec_n] =\
				amp_temp[mask_instr & (acc_all[obs_ind] == 0)].mean()

prop_theta_peak_found[np.isnan(prop_theta_peak_found)] = 0





############## 		PROPORTION OF THETA PEAKS FOUND 	###############
scale_it = True
if scale_it:
	std_temp = np.nanstd(prop_theta_peak_found, axis=1)[:, np.newaxis, :, :]
	std_temp[np.isnan(std_temp) | (std_temp==0)] = 1
else: std_temp = 1
zscore_prop_theta_perInstCorr = (prop_theta_peak_found\
							- prop_theta_peak_found.mean(axis=1)[:, np.newaxis, :, :])\
							/ std_temp

# zscore_prop_theta_perInstCorr = prop_theta_peak_found

ch_name = 'Fz'
ch_idx = eps.info['ch_names'].index(ch_name)

fig, axs = pl.subplots(1, 2)
axs[0].set_xticks(np.arange(4)); axs[0].set_xticklabels(insttxts[inst_diff_order])
toplot = zscore_prop_theta_perInstCorr[:, inst_diff_order, 0, ch_idx]
axs[0].errorbar(x=np.arange(4), y=toplot.mean(axis=0),
	yerr = toplot.std(axis=0)/np.sqrt(n_obs), fmt='o', ms=10, color='k')

toplot = zscore_prop_theta_perInstCorr[:, inst_diff_order, 2, ch_idx]
axs[0].errorbar(x=np.arange(4)-.2, y=toplot.mean(axis=0),
	yerr = toplot.std(axis=0)/np.sqrt(n_obs), fmt='o', ms=10, color='r')

toplot = zscore_prop_theta_perInstCorr[:, inst_diff_order, 1, ch_idx]
axs[0].errorbar(x=np.arange(4)+.2, y=toplot.mean(axis=0),
	yerr = toplot.std(axis=0)/np.sqrt(n_obs), fmt='o', ms=10, color='g')


toplot = zscore_prop_theta_perInstCorr[:, :, 1:, ch_idx].mean(axis=1)
axs[1].errorbar(x=np.arange(2), y=toplot.mean(axis=0),
	yerr = toplot.std(axis=0)/np.sqrt(n_obs), fmt='o', ms=10, color='k')
axs[1].set_xticks(np.arange(2)); axs[1].set_xticklabels(['Corr.', 'Incorr.'])
pl.suptitle(data_name)






################################################################################
###########			WHERE (topo) DO WE FIND MOST THETA PEAKS 			########
################################################################################


centered_prop_theta_peak_found = prop_theta_peak_found\
	- prop_theta_peak_found.mean(axis=1)[:, np.newaxis, :,:]

fig, axs = pl.subplots(3, 4)
for corr_i in np.arange(3):
	for inst in inst_diff_order:
		im, cn = mne.viz.plot_topomap(\
			prop_theta_peak_found[:, inst, corr_i, :].mean(axis=0),
			eps.info, cmap='hot', sensors=True, outlines='head',
			extrapolate ='local', head_pos = {'scale':[1.3, 1.7]},
			axes=axs[corr_i, inst],
			# vmin=-.05, vmax=.05)
			vmin=.4, vmax=.64)
pl.colorbar(im, ax=axs[-1,-1])
pl.suptitle(data_name)



fig, axs = pl.subplots(3, 4)
for corr_i in np.arange(3):
	for inst in inst_diff_order:
		toplot = np.array([stats.binom_test((prop_theta_peak_found[:, inst, corr_i, elec_n]>.3).sum(),
			n=n_obs, p=0.5, alternative='greater') for elec_n in np.arange(n_elec)])
		im, cn = mne.viz.plot_topomap(-np.log10(toplot),
			eps.info, cmap='afmhot', sensors=True, outlines='head',
			extrapolate ='local', head_pos = {'scale':[1.3, 1.7]},
			axes=axs[corr_i, inst], vmin=-np.log10(.05), vmax=5)
pl.colorbar(im, ax=axs[-1,-1])
pl.suptitle(data_name)


diff_correct = prop_theta_peak_found[:,:,1,:] - prop_theta_peak_found[:,:,2,:]
t, p = stats.ttest_1samp(diff_correct, popmean=0, axis=0)
fig, axs = pl.subplots(2, 4)
for ind, inst in enumerate(inst_diff_order):
	im1, cn = mne.viz.plot_topomap(t[inst], eps.info, cmap='RdBu_r', sensors=True,
		outlines='head', extrapolate ='local', head_pos = {'scale':[1.3, 1.7]},
		axes=axs[0, ind], vmin=-3, vmax=3)
	poneside = p[inst].copy()
	poneside[t[inst]<0]=1 
	im2, cn = mne.viz.plot_topomap(-np.log10(poneside), eps.info, cmap='afmhot', sensors=True,
		outlines='head', extrapolate ='local', head_pos = {'scale':[1.3, 1.7]},
		axes=axs[1, ind], vmin=0, vmax=2)
	axs[0, ind].set_title(insttxts[inst])
pl.colorbar(im1, ax=axs[0,-1])
pl.colorbar(im2, ax=axs[-1,-1])
pl.suptitle(data_name)

################################################################################



################################################################################
###########			THETA PEAK AMPLITUDE BY INSTRUCTIONS???????? 		########
################################################################################

scale_it = False
if scale_it:
	std_temp = np.nanstd(avg_theta_amp_byInstCorr, axis=1)[:, np.newaxis, :, :]
	std_temp[np.isnan(std_temp) | (std_temp==0)] = 1
else: std_temp = 1
zscore_theta_amp_byInstCorr =\
	(avg_theta_amp_byInstCorr -\
		np.nanmean(avg_theta_amp_byInstCorr, axis=1)[:, np.newaxis, :, :]) \
		/ std_temp

zscore_theta_amp_byInstCorr = avg_theta_amp_byInstCorr

ch_name = 'Fz'
ch_idx = eps.info['ch_names'].index(ch_name)

corr_names = ['All', 'Correct', 'Incorrect']
fig, ax = pl.subplots(1, 1)
for corr_ind, corr_i in enumerate([2, 0, 1]):
	xs = np.arange(4)+(corr_i-1)/8
	toplot = zscore_theta_amp_byInstCorr[:, inst_diff_order, corr_i, ch_idx]
	ax.errorbar(x=xs, y=toplot.mean(axis=0), yerr = toplot.std(axis=0)/n_obs**.5,
		fmt='o', ms=10, color=['k', 'g', 'r'][corr_i], label=corr_names[corr_i])
	# for obs_ind in np.arange(n_obs): ax.plot(xs, toplot[obs_ind, :], 'o', color=[.6, .6, .6, .6])
ax.set_xticks(np.arange(4)); ax.set_xticklabels(insttxts[inst_diff_order])
ax.set_title('Amplitude at %s\nby instruction by correctness' % ch_name)
ax.legend(loc='best'); ax.grid()

from mne.channels import find_ch_connectivity
from mne.stats import permutation_cluster_test, spatio_temporal_cluster_test
from scipy import stats
connectivity, ch_names = find_ch_connectivity(eps.info, ch_type='eeg')

p_accept = .05
threshold = stats.distributions.t.ppf(1 - .05, n_obs - 1)

# topographies
fig, axs = pl.subplots(3, 4)
for ind, inst in enumerate(inst_diff_order):
	axs[0, ind].set_title(insttxts[inst])
	for corr_i in np.arange(2):
		toplot = zscore_theta_amp_byInstCorr[:, inst, corr_i+1, :]
		toplot = (toplot - toplot.mean(axis=1)[:, np.newaxis])\
					/toplot.std(axis=1)[:, np.newaxis]
		if corr_i >= 0:
			cluster_stats = mne.stats.permutation_cluster_1samp_test(X=toplot, n_permutations=1000,
											 threshold=threshold, tail=1,
											 n_jobs=-1, buffer_size=None,
											 connectivity=connectivity)
			T_obs, clusters, p_values, _ = cluster_stats
			good_cluster_inds = np.where(p_values < p_accept)[0]
			mask = np.array(clusters)[good_cluster_inds].squeeze().astype(np.bool)
		else:
			mask = np.zeros(shape=64, dtype=np.bool)
		im1, cn = mne.viz.plot_topomap(toplot.mean(axis=0), eps.info, cmap='Greens_r',
			sensors=True, outlines='head', extrapolate ='local',
			head_pos = {'scale':[1.3, 1.7]}, axes=axs[corr_i, ind],
			vmin=-.5, vmax=.5, contours=3, mask=mask)

	toplot1 = zscore_theta_amp_byInstCorr[:, inst, 1, :]
	toplot1 = (toplot1 - toplot1.mean(axis=1)[:, np.newaxis])\
				/toplot1.std(axis=1)[:, np.newaxis]
	toplot2 = zscore_theta_amp_byInstCorr[:, inst, 2, :]
	toplot2 = (toplot2 - toplot2.mean(axis=1)[:, np.newaxis])\
			/toplot2.std(axis=1)[:, np.newaxis]
	toplot = toplot1 - toplot2
	
	# compute clusters
	T_obs, clusters, p_values, _ =\
		mne.stats.permutation_cluster_1samp_test(X=toplot, n_permutations=1000,
			threshold=threshold, tail=1, n_jobs=-1, buffer_size=None,
			connectivity=connectivity)
	good_cluster_inds = np.where(p_values < p_accept)[0]
	mask = np.array(clusters)[good_cluster_inds].squeeze()

	t, p = stats.ttest_1samp(toplot, popmean=0, axis=0)
	im2, cn = mne.viz.plot_topomap(t, eps.info, cmap='RdBu_r',
		sensors=True, outlines='head', extrapolate ='local',
		head_pos = {'scale':[1.3, 1.7]}, axes=axs[2, ind],
		vmin=-2, vmax=2, contours=3, mask = mask)
pl.colorbar(im1, ax=axs[1,-1])
pl.colorbar(im2, ax=axs[-1,-1])
axs[1, 0].set_ylabel('Incorrect trials', fontsize=8)
axs[0, 0].set_ylabel('Correct trials', fontsize=8)
axs[2, 0].set_ylabel('Correct-Incorrect\nT-value', fontsize=8)
pl.suptitle('Z-scored (per obs., across elecs) theta peak amp.')




fig, ax = pl.subplots(1,1)
toplot1 = zscore_theta_amp_byInstCorr[:, :, 1, :]
toplot1 = (toplot1 - toplot1.mean(axis=2)[:, :, np.newaxis])\
			/toplot1.std(axis=2)[:, :, np.newaxis]
toplot2 = zscore_theta_amp_byInstCorr[:, :, 2, :]
toplot2 = (toplot2 - toplot2.mean(axis=2)[:, :, np.newaxis])\
		/toplot2.std(axis=2)[:, :, np.newaxis]
toplot = (toplot1 - toplot2).mean(axis=1)

# compute clusters
T_obs, clusters, p_values, _ =\
	mne.stats.permutation_cluster_1samp_test(X=toplot, n_permutations=10000,
		threshold=threshold, tail=1, n_jobs=-1, buffer_size=None,
		connectivity=connectivity)
good_cluster_inds = np.where(p_values < p_accept)[0]
mask = np.array(clusters)[good_cluster_inds].squeeze()

t, p = stats.ttest_1samp(toplot, popmean=0, axis=0)
im1, cn = mne.viz.plot_topomap(t, eps.info, cmap='RdBu_r',
	sensors=True, outlines='head', extrapolate ='local',
	head_pos = {'scale':[1.3, 1.7]}, axes=ax,
	vmin=-2, vmax=2, contours=3, mask = mask)

pl.colorbar(im1, ax=ax)
ax.set_ylabel('Correct-Incorrect\nT-value', fontsize=8)
pl.suptitle('Z-scored (per obs., across elecs) theta peak amp.')



ztoplot = (avg_theta_amp_byInstCorr - avg_theta_amp_byInstCorr.mean(axis=-1)[..., np.newaxis])\
				/ avg_theta_amp_byInstCorr.std(axis=-1)[..., np.newaxis]
fig, axs = pl.subplots(1, 4)
for inst_i in np.arange(4):
	for obs_ind in np.arange(n_obs):
		toplot = ztoplot[obs_ind, inst_i, 1:, 1]
		axs[inst_i].plot([.5, 1], toplot, 'o-', color = 'grey', alpha = .5)
	axs[inst_i].errorbar(x=[.5, 1], y=ztoplot[:, inst_i, 1:, 1].mean(axis=0),
		yerr=ztoplot[:, inst_i, 1:, 1].std(axis=0)/(n_obs**.5), fmt='ko-', ms=10, linewidth=3)
	axs[inst_i].set_xlim([0,1.5])



################################################################################
###########			THETA PEAK SHIFT BY INSTRUCTIONS ???????? 			########
################################################################################

scale_it = False
if scale_it:
	std_temp = np.nanstd(avg_theta_peak_byInstCorr, axis=1)[:, np.newaxis, :, :]
	std_temp[np.isnan(std_temp) | (std_temp==0)] = 1
else: std_temp = 1
zscore_theta_peak_byInstCorr = (avg_theta_peak_byInstCorr -\
		np.nanmean(avg_theta_peak_byInstCorr, axis=1)[:, np.newaxis, :, :]) \
		/ std_temp

zscore_theta_peak_byInstCorr = avg_theta_peak_byInstCorr


ch_name = 'AFz'
ch_idx = eps.info['ch_names'].index(ch_name)

corr_names = ['All', 'Correct', 'Incorrect']
fig, ax = pl.subplots(1, 1)
for corr_ind, corr_i in enumerate([2, 0, 1]):
	xs = np.arange(4)+(corr_i-1)/8
	toplot = zscore_theta_peak_byInstCorr[:, inst_diff_order, corr_i, ch_idx]
	ax.errorbar(x=xs, y=np.nanmean(toplot, axis=0), yerr = np.nanstd(toplot, axis=0)/n_obs**.5,
		fmt='o', ms=10, color=['k', 'g', 'r'][corr_i], label=corr_names[corr_i])
	# for obs_ind in np.arange(n_obs): ax.plot(xs, toplot[obs_ind, :], 'o', color=[.6, .6, .6, .6])
ax.set_xticks(np.arange(4)); ax.set_xticklabels(insttxts[inst_diff_order])
ax.set_title('Theta Peak at %s\nby instruction by correctness' % ch_name)
ax.legend(loc='best'); ax.grid()



fig, axs = pl.subplots(3, 4)
for corr_i in np.arange(3):
	for ind, inst in enumerate(inst_diff_order):
		toplot1 = zscore_theta_peak_byInstCorr[:, inst, corr_i, :]# - zscore_theta_peak_byInstCorr[:, 3, corr_i, :]
		# toplot1 = (toplot1 - np.nanmean(toplot1, axis=1)[:, np.newaxis])\
		# 	/ np.nanstd(toplot1, axis=1)[:,np.newaxis]

		# toplot1[np.isnan(t)] = 0
		im, cn = mne.viz.plot_topomap(np.nanmean(toplot1, axis=0), eps.info,
			cmap='RdBu_r', sensors=True, outlines='head', extrapolate ='local',
			head_pos = {'scale':[1.3, 1.7]}, axes=axs[corr_i, ind],
			vmin=-.05, vmax=.05)#, mask = p<.05)
		axs[0, ind].set_title(insttxts[inst])
	axs[corr_i, 0].set_ylabel(corr_names[corr_i])
	pl.colorbar(im, ax=axs[corr_i, -1])
# axs[0].set_ylabel('theta peak')



##############################################################
#### Difference in peak between correct and incorrect trials?

centered_avg_theta_peak_byInstCorr = (avg_theta_peak_byInstCorr -\
		np.nanmean(avg_theta_peak_byInstCorr, axis=1)[:, np.newaxis, :, :])

# zscored_avg_theta_peak_byInstCorr = centered_avg_theta_peak_byInstCorr \
# 	/ avg_theta_peak_byInstCorr.std(axis=2).mean(axis=1)[:, np.newaxis, np.newaxis, :]

data = avg_theta_peak_byInstCorr
data = centered_avg_theta_peak_byInstCorr
# data = zscored_avg_theta_peak_byInstCorr

diff_corr_incorr_thetaPeakByInst = data[:, :, 1, :] - data[:, :, 2, :]

ch_name = 'Fz'
ch_idx = eps.info['ch_names'].index(ch_name)

fig, axs = pl.subplots(1, 2)
axs[0].set_xticks(np.arange(4)); axs[0].set_xticklabels(insttxts[inst_diff_order])
axs[0].errorbar(x=np.arange(4),
	y=diff_corr_incorr_thetaPeakByInst[:, inst_diff_order, ch_idx].mean(axis=0),
	yerr = diff_corr_incorr_thetaPeakByInst[:, inst_diff_order, ch_idx].std(axis=0)/np.sqrt(len(obs_all)),
	fmt='o', ms=10, color='k')
axs[0].set_title('Correct-Incorrect difference in theta peak')
pl.suptitle(data_name)

fig, axs = pl.subplots(1, 3)
for corr_i in np.arange(3):
	toplot1 = np.nanmean(zscore_theta_peak_byInstCorr[:, [0, 3], corr_i, :], axis=1)
	toplot1 = (toplot1 - np.nanmean(toplot1, axis=1)[:, np.newaxis]) / np.nanstd(toplot1, axis=1)[:,np.newaxis]
	toplot2 = np.nanmean(zscore_theta_peak_byInstCorr[:, [1, 2], corr_i, :], axis=1)
	toplot2 = (toplot2 - np.nanmean(toplot2, axis=1)[:, np.newaxis]) / np.nanstd(toplot2, axis=1)[:,np.newaxis]

	diff = toplot1-toplot2
	t, p = stats.ttest_1samp(diff, axis=0, popmean=0)
	t[np.isnan(t)] = 0
	im, cn = mne.viz.plot_topomap(t, eps.info, cmap='RdBu_r', sensors=True,
		outlines='head', extrapolate ='local', head_pos = {'scale':[1.3, 1.7]},
		axes=axs[corr_i], vmin=-4, vmax=4, mask = p<.05)

	axs[corr_i].set_title(corr_names[corr_i])
axs[0].set_ylabel('Tvalue diff [RR, LL] - [RL, LR]')
pl.colorbar(im, ax=axs[-1])






# plot topography of frequency shift Easy minus Difficult conditions
fig, axs = pl.subplots(1, 3)
for corr_i in np.arange(3):
	toplot1 = np.nanmean(zscore_theta_peak_byInstCorr[:, [0, 3], corr_i, :], axis=1)
	toplot1 = (toplot1 - np.nanmean(toplot1, axis=1)[:, np.newaxis]) / np.nanstd(toplot1, axis=1)[:,np.newaxis]
	toplot2 = np.nanmean(zscore_theta_peak_byInstCorr[:, [1, 2], corr_i, :], axis=1)
	toplot2 = (toplot2 - np.nanmean(toplot2, axis=1)[:, np.newaxis]) / np.nanstd(toplot2, axis=1)[:,np.newaxis]

	diff = toplot1-toplot2
	t, p = stats.ttest_1samp(diff, axis=0, popmean=0)
	t[np.isnan(t)] = 0
	im, cn = mne.viz.plot_topomap(t, eps.info, cmap='RdBu_r', sensors=True,
		outlines='head', extrapolate ='local', head_pos = {'scale':[1.3, 1.7]},
		axes=axs[corr_i], vmin=-4, vmax=4, mask = p<.05)

	axs[corr_i].set_title(corr_names[corr_i])
axs[0].set_ylabel('Tvalue diff [RR, LL] - [RL, LR]')
pl.colorbar(im, ax=axs[-1])

t, p = stats.ttest_1samp(zscore_theta_peak_byInstCorr[:, [0, 3], 1, :].mean(axis=1)\
	- zscore_theta_peak_byInstCorr[:, [1,2], 1, :].mean(axis=1), popmean=0, axis=0)
fig, ax = pl.subplots(1, 1)
im, cn = mne.viz.plot_topomap(t, eps.info, cmap='RdBu_r', sensors=True, outlines='head',
	head_pos = {'scale':[1.3, 1.7]}, axes=ax, vmin=-3, vmax=3, show_names=True, names=eps.ch_names)#,
	#mask = p<.05)
pl.colorbar(im, ax=ax)


# correlation inst difficulty, peak frequency
fig, ax = pl.subplots(2,1)
toplot = np.array([[stats.spearmanr(np.arange(4),\
	zscore_theta_peak_byInstCorr[obs_ind, inst_diff_order, 1, elec_n])[0]\
	for elec_n in np.arange(n_elec)] for obs_ind in np.arange(n_obs)])
im1, cn = mne.viz.plot_topomap(toplot.mean(axis=0),
			eps.info, cmap='RdBu_r', sensors=True, outlines='head',
			extrapolate ='local', head_pos = {'scale':[1.3, 1.7]},
			axes=ax[0], vmin=-.5, vmax=.5)
t, p = stats.ttest_1samp(toplot, popmean=0, axis=0)
im2, cn = mne.viz.plot_topomap(t, eps.info, cmap='RdBu_r', sensors=True,
	outlines='head', extrapolate ='local', head_pos = {'scale':[1.3, 1.7]},
	axes=ax[1], vmin=-2, vmax=2)
pl.colorbar(im1, ax=ax[0])
pl.colorbar(im2, ax=ax[1])
ax[0].set_title(\
	'overall spearman corr. between\ninst difficulty and peak frequency (by obs)')


fig, axs = pl.subplots(5, 7)
for obs_ind in np.arange(len(obs_all)):
	ax = axs.flatten()[obs_ind]
	toplot = np.array([stats.spearmanr(np.arange(4),\
		zscore_theta_peak_byInstCorr[obs_ind, inst_diff_order, 0, elec_n])[0]\
		for elec_n in np.arange(n_elec)])
	im, cn = mne.viz.plot_topomap(toplot,eps.info, cmap='RdBu_r',
		sensors=True, outlines='head', extrapolate ='local',
		head_pos = {'scale':[1.3, 1.7]}, axes=ax, vmin=-1, vmax=1)









behav_theta_freq = np.load('behav_theta_freq_5tp_padding_meanDiffPeakEst.npy', allow_pickle=True)
# behav_theta_freq = behav_theta_freq - behav_theta_freq.mean(axis=2)[:, :, np.newaxis]
# std_temp = behav_theta_freq.std(axis=2)[:, :, np.newaxis]
# std_temp[np.isnan(std_temp) | (std_temp==0)] = 1
# behav_theta_freq /= std_temp

ch_inds = np.array([eps.info['ch_names'].index(ch_name) for ch_name in ['AFz', 'AF3', 'AF4', 'Fz', 'F1', 'F2', 'FCz', 'FC1', 'FC2']])
ch_idx = ch_inds[3]

eeg_data_toCorrel = avg_theta_peak_byInstCorr[:,:,1,:].copy()
# eeg_data_toCorrel -= np.nanmean(eeg_data_toCorrel, axis=1)[:, np.newaxis,:]
# std_temp = eeg_data_toCorrel.std(axis=1)[:, np.newaxis, :]
# std_temp[np.isnan(std_temp) | (std_temp==0)] = 1
# eeg_data_toCorrel /= std_temp
# eeg_data_toCorrel = avg_theta_peak_byInstCorr[:,:,1,:]

fig, axs = pl.subplots(1, 4)
for ind, inst in enumerate(inst_diff_order):
	x, y = behav_theta_freq[0, :, inst], eeg_data_toCorrel[:, inst, ch_inds].mean(axis=-1)
	# x, y = theta_peak_zscore_ddm[mask_obs, 1], avg_theta_peak_byCorr[mask_obs, 1, ch_idx]
	axs[ind].plot(x, y, 'ko')
	a, b = stats.spearmanr(x, y)
	axs[ind].set_title('instruction: %s\nrho = %.3f, pval = %.3f' % (insttxts[inst], a, b))
	m, b = np.polyfit(x, y, 1)
	xs = np.array([x.min(), x.max()])
	axs[ind].plot(xs, m*xs + b, color=[.2, .5, .8])


fig, axs = pl.subplots(3, 4)
for ind, inst in enumerate(inst_diff_order):
	ax = axs[0, ind]
	a = np.array([stats.spearmanr(behav_theta_freq[0, :, inst],
		eeg_data_toCorrel[:, inst, elec_n]) for elec_n in np.arange(n_elec)])
	im, cn = mne.viz.plot_topomap(a[:, 0], eps.info, cmap='RdBu_r', sensors=True,
		outlines='head', extrapolate ='local', head_pos = {'scale':[1.3, 1.7]},
		axes=ax, vmin=-.5, vmax=.5, mask = a[:, 1]<.05)
	pl.colorbar(im, ax = ax)
	
	ax = axs[1, ind]
	im, cn = mne.viz.plot_topomap(-np.log10(a[:, 1]),
			eps.info, cmap='hot', sensors=True, outlines='head', extrapolate ='local', head_pos = {'scale':[1.3, 1.7]}, axes=ax, vmin=1.3, vmax=2.5)
	pl.colorbar(im, ax = ax)

	ax = axs[2, ind]; max_elec = np.argmax(-np.log10(a[:, 1])).squeeze()
	x, y = behav_theta_freq[0, :, inst], eeg_data_toCorrel[:, inst, max_elec]
	ax.plot(x, y, 'ko')
	a, b = stats.spearmanr(x, y)
	ax.set_title('channel %s\nrho = %.3f, pval = %.3f' % (np.str(np.array(eps.info['ch_names'])[max_elec]), a, b))
	m, b = np.polyfit(x, y, 1)
	xs = np.array([x.min(), x.max()])
	ax.plot(xs, m*xs + b, color=[.2, .5, .8])




























######   DOES NEURAL THETA BEHAVE?    #######

# mask_obs = n_trials_per_obs > 200
mask_obs = corr_all > .5

# mask_obs = np.ones(len(obs_all), dtype=np.bool)

elecdict = dict(zip(eps.info['ch_names'],np.arange(64))) 


ch_name = 'FCz'
ch_idx = [eps.info['ch_names'].index(ch_name)]
ch_idx = [6, 38, 34, 1, 2] # works
fig, axs = pl.subplots(1, 4)
for i in np.arange(4):
	x, y = theta_peak_zscore_ddm[mask_obs, 1], np.median(avg_theta_peak_byInstCorr[mask_obs, i, 1, :][..., ch_idx], axis=-1)
	# x, y = theta_peak_zscore_ddm[mask_obs, 1], avg_theta_peak_byCorr[mask_obs, 1, ch_idx]
	axs[i].plot(x, y, 'ko')
	a, b = stats.spearmanr(x, y)
	axs[i].set_title('instruction: %s\nrho = %.3f, pval = %.3f' % (insttxts[i], a, b))
	m, b = np.polyfit(x, y, 1)
	xs = np.array([x.min(), x.max()])
	axs[i].plot(xs, m*xs + b, color=[.2, .5, .8])

# plot correlation for just one electrode for all data by correctness
ch_name = 'FCz'
ch_idx = [eps.info['ch_names'].index(ch_name)]

ch_idx = [6, 38, 34, 1, 2] # works
# ch_idx = [1, 6, 32, 38, 27, 60, 34, 61]
# ch_idx = [1, 32, 33, 34, 60, 61]
fig, ax = pl.subplots(1, 1)
x, y = theta_peak_zscore_ddm[mask_obs, 1], np.median(np.median(avg_theta_peak_byInstCorr[mask_obs, :, 1, :], axis=1)[:, ch_idx], axis=-1)
# x, y = theta_peak_zscore_ddm[mask_obs, 1], np.median(np.median(avg_theta_peak_byInst[mask_obs, :, :], axis=1)[:, ch_idx], axis=-1)
# x, y = theta_peak_zscore_ddm[mask_obs, 1], avg_theta_peak_all[mask_obs, :][:, ch_idx].mean(axis=-1)
# x, y = theta_peak_zscore_ddm[mask_obs, 1], avg_theta_peak_byCorr[mask_obs, 1, :][:, ch_idx].mean(axis=-1)
ax.plot(x, y, 'ko')
a, b = stats.spearmanr(x, y)
ax.set_title('instruction: %s - channel(s) %s\nrho = %.3f, pval = %.3f' % ('all instr.', np.str(np.array(eps.info['ch_names'])[ch_idx]), a, b))
m, b = np.polyfit(x, y, 1)
xs = np.array([x.min(), x.max()])
ax.plot(xs, m*xs + b, color=[.2, .5, .8])



fig, axs = pl.subplots(3, 4)
for i in np.arange(4):
	ax = axs[0, i]
	a = np.array([stats.spearmanr(theta_peak_zscore_ddm[mask_obs, 1], avg_theta_peak_byInst[mask_obs, i, elec_n]) for elec_n in np.arange(64)])
	# a = np.array([stats.spearmanr(theta_peak_zscore_ddm[mask_obs, 1], avg_theta_peak_byInstCorr[mask_obs, i, 1, elec_n]) for elec_n in np.arange(64)])
	im, cn = mne.viz.plot_topomap(a[:, 0], eps.info, cmap='RdBu_r', sensors=True, outlines='head', extrapolate ='local',
						head_pos = {'scale':[1.3, 1.7]}, axes=ax, vmin=-.5, vmax=.5, mask = a[:, 1]<.05)
	pl.colorbar(im, ax = ax)
	
	ax = axs[1, i]
	im, cn = mne.viz.plot_topomap(-np.log10(a[:, 1]),
			eps.info, cmap='hot', sensors=True, outlines='head', extrapolate ='local', head_pos = {'scale':[1.3, 1.7]}, axes=ax, vmin=1.3, vmax=2.5)
	pl.colorbar(im, ax = ax)

	ax = axs[2, i]; max_elec = np.argmax(-np.log10(a[:, 1])).squeeze()
	x, y = theta_peak_zscore_ddm[mask_obs, 1], avg_theta_peak_byInst[mask_obs, i, max_elec]
	ax.plot(x, y, 'ko')
	a, b = stats.spearmanr(x, y)
	ax.set_title('channel %s\nrho = %.3f, pval = %.3f' % (np.str(np.array(eps.info['ch_names'])[max_elec]), a, b))
	m, b = np.polyfit(x, y, 1)
	xs = np.array([x.min(), x.max()])
	ax.plot(xs, m*xs + b, color=[.2, .5, .8])




fig, axs = pl.subplots(2, 1)
ax = axs[0]
# by Inst - mean
# a = np.array([stats.spearmanr(theta_peak_zscore_ddm[mask_obs, 1], np.median(avg_theta_peak_byInstCorr[mask_obs,...][:, :, 1, elec_n], axis=1)) for elec_n in np.arange(64)])
# by Inst - median
# a = np.array([stats.spearmanr(theta_peak_zscore_ddm[mask_obs, 1], np.median(avg_theta_peak_byInst[mask_obs,...][:, :, elec_n], axis=-1)) for elec_n in np.arange(64)])

# all - Ø
a = np.array([stats.spearmanr(theta_peak_zscore_ddm[mask_obs, 1], avg_theta_peak_all[mask_obs,elec_n]) for elec_n in np.arange(64)])

# a = np.array([stats.spearmanr(theta_peak_zscore_ddm[mask_obs, 1], avg_theta_peak_byCorr[mask_obs, 1, elec_n]) for elec_n in np.arange(64)])

# a = np.array([stats.spearmanr(theta_peak_classic[mask_obs, 0], avg_theta_peak_byCorr[mask_obs, 1, elec_n]) for elec_n in np.arange(64)])
im, cn = mne.viz.plot_topomap(a[:, 0],
			eps.info, cmap='RdBu_r', sensors=True, outlines='head', extrapolate ='local',
			head_pos = {'scale':[1.3, 1.7]}, axes=ax, vmin=-.5, vmax=.5, mask = a[:, 1]<.05)
pl.colorbar(im, ax = ax)
ax = axs[1]
im, cn = mne.viz.plot_topomap(-np.log10(a[:, 1]),
		eps.info, cmap='hot', sensors=True, outlines='head', extrapolate ='local', head_pos = {'scale':[1.3, 1.7]}, axes=ax, vmin=1.3, vmax=2.5)
pl.colorbar(im, ax = ax)















