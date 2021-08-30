"""
@authors: Mehdi Senoussi, Pieter Verbeke, Tom Verguts
"""

from theta_sync_compet_model_utils import Model_sim

import os
from mne import parallel as par
import numpy as np
import time as tm
base_path = './simulations/theta_amp/'
if not os.path.exists(base_path): os.mkdir(base_path)

drift = 0
thresh = 4
tiltrate = .020
Cgs_var_sd = 1
kick_value = .5
MFC_compet_thresh = .1
theta_amplitude = 0.75
sigma_compet = .075
sw2 = .5
inh_compet = .1
alpha_compet = .13
nReps = 50

sim_path = base_path +\
	'/LFC_compet_sw2_%.2f_kick_%.2f_thresh_%.2f_cgSd_%.2f_sigmaCompet_%.3f_inhCompet_%.2f_alphCompet_%.3f_tilt_%.4f_thetaAmp_%.2f/'\
	% (sw2, kick_value, MFC_compet_thresh, Cgs_var_sd, sigma_compet, inh_compet, alpha_compet, tiltrate, theta_amplitude)
if not os.path.exists(sim_path): os.mkdir(sim_path)

parallel, my_cvstime, _ = par.parallel_func(Model_sim, n_jobs = -1, verbose = 40)

a = tm.localtime()
print('thresh %.1f, drift %.1f - time start: %i:%i' % (thresh, drift, a.tm_hour, a.tm_min))

t = tm.time()
parallel(my_cvstime(Threshold = thresh, drift = drift, Cgs_var_sd = Cgs_var_sd,
	theta_freq = theta, theta_amplitude = theta_amplitude, gamma_freq = 30,
	sim_path = sim_path, sw2 = sw2, kick_value = kick_value,
	MFC_compet_thresh = MFC_compet_thresh, nReps = nReps,
	sigma_compet = sigma_compet, inh_compet=inh_compet,
	alpha_compet = alpha_compet, tiltrate = tiltrate,
	print_prog = False, n_trials = None, save_eeg = False,
	save_behav = True, return_eeg = False, return_behav = False) for theta in np.arange(4, 7.1, .5))
print('\ttime taken: %.2fmin' % ((tm.time() - t) / 60.))


