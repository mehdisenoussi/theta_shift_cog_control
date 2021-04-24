# Script to compute spectra and run FOOOF toolbox to get frequency and amplitude of theta
# author: Mehdi Senoussi

import mne, os
import numpy as np
from scipy import signal as sig
from fooof import FOOOFGroup

# loads the channel file
data_path = './data/'

#### load epochs
obs_all = np.arange(1, 40)
## participants 5, 9, 15, 16 and 33 were excluded (see Methods)
obs_all = obs_all[np.array([obs_i not in [5, 9, 15, 16, 33] for obs_i in obs_all])]

n_elec = 64

for obs_ind, obs_i in enumerate(obs_all):
	print('obs %i'%obs_i)
	obs_eegpath = data_path + 'obs_%i/eeg/' % obs_i
	eps = mne.read_epochs(fname = data_path + 'obs_%i/eeg/obs%i_EEGclean_pre-stim-epo.fif.gz' % (obs_i, obs_i),
		proj = False, verbose= 50, preload=True)

	freqs_welch, amp_welch =  sig.welch(x = eps.get_data(), axis = -1,
		fs = 200, window='hann', average = 'mean', nperseg=200,
		nfft=400, return_onesided=True)
	theta_params_clean = np.zeros(shape = [64, len(eps), 3]) - 1

	for elec_n in np.arange(n_elec):
		if np.logical_not(elec_n % 10): print('\telec : %s' % eps.info['ch_names'][elec_n])
		fg = FOOOFGroup(verbose = False, max_n_peaks = 4, peak_width_limits = [.5, 2])
		fg.fit(freqs_welch, amp_welch[:, elec_n, :], [2, 20], n_jobs=-1)

		theta_params =\
			np.array([fg_res_n.peak_params[(fg_res_n.peak_params[:,0]>3.9)\
				& (fg_res_n.peak_params[:,0]<8)] for fg_res_n in fg.get_results()])
		for trial_n in np.arange(len(theta_params)):
			if len(theta_params[trial_n]) == 1:
				theta_params_clean[elec_n, trial_n, :] = theta_params[trial_n].squeeze()
			elif len(theta_params[trial_n]) > 1:
				max_amp_ind = np.argmax(theta_params[trial_n][:, 1]).squeeze()
				theta_params_clean[elec_n, trial_n, :] = theta_params[trial_n][max_amp_ind, :].squeeze()

	np.savez(obs_eegpath +\
		'obs_%i_fooof_params_theta.npz' % obs_i,
		{'theta_params_clean':theta_params_clean, 'instr_type':eps.metadata['instr_type'].values,
		'respcorrect':eps.metadata['respcorrect'].values, 'resptime':eps.metadata['resptime'].values})



