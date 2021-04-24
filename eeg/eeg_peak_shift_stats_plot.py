# Script to compute spectra and run FOOOF toolbox to get frequency and amplitude of theta
# author: Mehdi Senoussi


import mne
import numpy as np
from matplotlib import pyplot as pl
from scipy import stats
# from statsmodels.stats.anova import AnovaRM

import pandas as pd

from mne.channels import find_ch_connectivity
from scipy import stats
import pingouin as pg

# loads the channel file
data_path = './data/'
n_elec = 64

insttxts = np.array(['LL', 'LR', 'RL', 'RR'])
inst_diff_order = np.array([3, 0, 2, 1], dtype=np.int)

#### load epochs

obs_all = np.arange(1, 40)
## participants 5, 15, 9, 16 and 33 were excluded (see Methods)
obs_all = obs_all[np.array([obs_i not in [5, 9, 15, 16, 33] for obs_i in obs_all])]
n_obs = len(obs_all)

# load one example Epochs object to have topo etc.
eps = mne.read_epochs(fname = data_path + 'obs_1/eeg/obs1_EEGclean_pre-stim-epo.fif.gz',
	proj = False, verbose= 50, preload=True)
n_elec = len(eps.ch_names)

avg_theta_peak_byInstCorr = np.zeros(shape = [n_obs, 4, 2, n_elec])
avg_theta_amp_byInstCorr = np.zeros(shape = [n_obs, 4, 2, n_elec])

# theta_par_all = []; acc_all = [];inst_all = []
for obs_ind, obs_i in enumerate(obs_all):
	print('loading obs %i' % obs_i)
	obs_eegpath = data_path + 'obs_%i/eeg/' % obs_i
	z = np.load(obs_eegpath +\
			'obs_%i_fooof_params_theta.npz' % obs_i,
			allow_pickle=True)['arr_0'][..., np.newaxis][0]

	theta_par_obs = z['theta_params_clean'].astype(np.float).copy()
	acc_obs = z['respcorrect'].astype(np.int).copy()
	inst_obs = z['instr_type'].astype(np.int).copy()

	mask_corr = acc_obs==1
	for instr in np.arange(4):
		mask_instr = inst_obs == instr

		for elec_n in np.arange(n_elec):
			theta_there_mask = theta_par_obs[elec_n, :, 0] != -1
			theta_params_theta_there = theta_par_obs[elec_n, theta_there_mask, :]
			instr_theta_there = mask_instr[theta_there_mask]
			corr_theta_there_instr =\
				acc_obs[theta_there_mask][instr_theta_there].astype(np.int)
			avg_theta_peak_byInstCorr[obs_ind, instr, 0, elec_n] =\
				np.nanmean(theta_params_theta_there[instr_theta_there, 0][corr_theta_there_instr == 1])
			avg_theta_peak_byInstCorr[obs_ind, instr, 1, elec_n] =\
				np.nanmean(theta_params_theta_there[instr_theta_there, 0][corr_theta_there_instr == 0])

			amp_temp = theta_par_obs[elec_n, :, 1].copy()
			amp_temp[amp_temp==-1] = 0
			avg_theta_amp_byInstCorr[obs_ind, instr, 0, elec_n] =\
				np.nanmean(amp_temp[mask_instr & (acc_obs == 1)])
			avg_theta_amp_byInstCorr[obs_ind, instr, 1, elec_n] =\
				np.nanmean(amp_temp[mask_instr & (acc_obs == 0)])


amp_corr = avg_theta_amp_byInstCorr[:, :, 0, :]
amp_corr = (amp_corr - np.nanmean(amp_corr, axis=2)[:, :, np.newaxis])\
			/np.nanstd(amp_corr, axis=2)[:, :, np.newaxis]
amp_incorr = avg_theta_amp_byInstCorr[:, :, 1, :]
amp_incorr = (amp_incorr - np.nanmean(amp_incorr, axis=2)[:, :, np.newaxis])\
		/np.nanstd(amp_incorr, axis=2)[:, :, np.newaxis]
amp_diff = np.nanmean(amp_corr - amp_incorr, axis=1)

connectivity, ch_names = find_ch_connectivity(eps.info, ch_type='eeg')
ch_names = np.array(ch_names)

p_accept = .05
threshold = stats.distributions.t.ppf(1 - .05, n_obs - 1)

# compute clusters
T_obs, clusters, p_values, _ =\
	mne.stats.permutation_cluster_1samp_test(X=amp_diff, n_permutations=1000,
		threshold=threshold, tail=1, n_jobs=-1, buffer_size=None,
		connectivity=connectivity)
if threshold.__class__ == dict:
	mask = p_values<p_accept
else:
	good_cluster_inds = np.where(p_values < p_accept)[0]
	mask = np.array(clusters)[good_cluster_inds].squeeze()


ch_inds = np.array([eps.info['ch_names'].index(ch_name)\
	for ch_name in ch_names[mask]])


fig, axs = pl.subplots(1, 3)
t, p = stats.ttest_1samp(amp_diff, popmean=0, axis=0)
im1, cn = mne.viz.plot_topomap(t, eps.info, cmap='RdBu_r',
	sensors=False, outlines='head', extrapolate ='local',
	head_pos = {'scale':[1.3, 1.7]}, axes=axs[0], vmin=-3, vmax=3,
	contours=0, mask=mask)
axs[0].set_title('Topo theta amplitude diff', fontsize=10)
fig.colorbar(im1, ax = axs[0])

zscore_theta_peak_byInstCorr = avg_theta_peak_byInstCorr.copy()\
	-np.nanmean(avg_theta_peak_byInstCorr, axis=1)[:, np.newaxis, :, :]
std_temp = np.nanstd(avg_theta_peak_byInstCorr, axis=1)[:, np.newaxis, :, :]
std_temp[np.isnan(std_temp) | (std_temp==0)] = 1
zscore_theta_peak_byInstCorr /= std_temp

ch_inds = np.array([eps.info['ch_names'].index(ch_name)\
	for ch_name in np.array(ch_names)[mask]])

cols = np.array([[25, 196, 64],[252, 140, 50]])/255
colincorr = np.array([110, 110, 110])/255
for corr_ind, corr_i in enumerate([0, 1]):
	xs = np.arange(4)+(corr_i-1)/8
	toplot = np.nanmean(zscore_theta_peak_byInstCorr[:, inst_diff_order, corr_i, :][:, :, ch_inds],
		axis=-1)
	a = axs[1].errorbar(x=xs, y=np.nanmean(toplot, axis=0), yerr = np.nanstd(toplot, axis=0)/n_obs**.5,
		fmt=['o', 'v'][corr_ind], lw = 2.5, mec = ['k', colincorr][corr_ind], mew = 2.5, ms=10, mfc=cols[corr_ind],
		color=['k', colincorr][corr_ind])
axs[1].set_xticks(np.arange(4))
axs[1].set_xticklabels(insttxts[inst_diff_order])
axs[1].grid(color = [.7, .7, .7], linewidth=.5)
axs[1].hlines(y=0, xmin=-.5, xmax=4.5)
axs[1].set_xlim(-.5, 3.5)

# RM-ANOVA
bla = np.nanmean(zscore_theta_peak_byInstCorr[:, inst_diff_order, 0, :][:,:, ch_inds], axis=-1)
subjs = np.repeat(obs_all, 4)
hand = np.tile(np.array([1, 2, 1, 2]), n_obs)
hemi = np.tile(np.array([1, 2, 2, 1]), n_obs)
df = pd.DataFrame({'obs':subjs, 'hand':hand, 'hemi':hemi,
			'theta_peak':bla.flatten()})
# run anova
aov = pg.rm_anova(data=df, dv='theta_peak', subject='obs', within=['hand', 'hemi'],
	detailed=True, effsize='n2')
fvals = aov['F'].values
pvals = aov['p-unc'].values
etas = aov['n2'].values
print('Theta Peak:\n\tF=%s\n\tp=%s\n\tn^2=%s'% (np.str(fvals), np.str(pvals), np.str(etas)))


# compute diff correct - incorrect
diff_peak = np.nanmean(zscore_theta_peak_byInstCorr[:, inst_diff_order, 0, :][:,:, ch_inds]\
	-zscore_theta_peak_byInstCorr[:, inst_diff_order, 1, :][:, :, ch_inds], axis=-1)
# RM-ANOVA
df = pd.DataFrame({'obs':subjs, 'hand':hand, 'hemi':hemi,
			'theta_peak_diff':diff_peak.flatten()})
aov = pg.rm_anova(data=df, dv='theta_peak_diff', subject='obs', within=['hand', 'hemi'],
	detailed=True, effsize='n2')
fvals = aov['F'].values
pvals = aov['p-unc'].values
etas = aov['n2'].values
print('Theta Peak diff (corr-incorr):\n\tF=%s\n\tp=%s\n\tn^2=%s'% (np.str(fvals), np.str(pvals), np.str(etas)))


###### SLOPE OF THETA SHIFT
corr_all_inst = np.load(data_path + 'corr_byInst_all_forEEG.npy')

overall_acc_per_inst = corr_all_inst.mean(axis=0)[inst_diff_order]*100

lin_rel_peak_inst = np.zeros(shape = [n_obs, 2])
for obs_ind in np.arange(n_obs):
	theta_peaks = np.nanmean(avg_theta_peak_byInstCorr[:, inst_diff_order, 0, :][obs_ind, :, ch_inds], axis=0)
	lin_rel_peak_inst[obs_ind, :] = np.polyfit(overall_acc_per_inst, theta_peaks, 1)

# inter-individual differences in accuracy linked to theta peak shift?
corr_all = np.load(data_path + 'corr_all_forEEG.npy')

xs = np.array([-.6, .6])
x = lin_rel_peak_inst[:, 0]
y = corr_all*100

rho, pval, outlier = pg.correlation.skipped(x,y,method='spearman')

correl_eeg_peak_acc = np.polyfit(x[~outlier], y[~outlier], 1)
axs[2].plot(x, y, 'o', ms=10, mec=[1,1,1], mfc=[.7, .7, .7], mew=1.5)
axs[2].plot(xs, correl_eeg_peak_acc[0]*xs + correl_eeg_peak_acc[1], 'k--')
axs[2].set_title('rho=%.2f, p=%.4f' % (rho, pval), fontsize=10)
axs[2].set_xlim([-.04, .04]); axs[2].set_ylim([55.5, 84]) 
axs[2].set_xlabel('Slope of theta increase with task "easiness"')
axs[2].set_ylabel('Overall accuracy (%)')

fig.set_size_inches(10, 4)
pl.tight_layout()





