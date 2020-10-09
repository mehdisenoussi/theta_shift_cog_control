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


n_elec = 64

eps_all = []
psds_welch_mean_all = np.zeros(shape = [len(obs_all), 40, 64])
for obs_ind, obs_i in enumerate(obs_all):
	print('obs %i'%obs_i)
	eps = mne.read_epochs(fname = data_path + 'obs_%i/eeg/obs%i_allclean_pre-stim_epo.fif.gz' % (obs_i, obs_i), proj = False, verbose= 50, preload=True)
	eps_all.append(eps)


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
			np.array([fg_res_n.peak_params[(fg_res_n.peak_params[:,0]>=4)\
				& (fg_res_n.peak_params[:,0]<=7)] for fg_res_n in fg.get_results()])
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
	
	