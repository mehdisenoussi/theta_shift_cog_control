# this script plots different details about the model (e.g. neural triplet)
# that will be used to describe the model in the supplementary materials of the manuscript

from model.theta_sync_compet_model_utils import *

import os
import time as tm
from matplotlib import pyplot as pl


def plot_shadederr(subp, data, error_measure = 'avg_sem', percs = [], color = 'blue',
	x = None, xlim = None, ylim = None, label = None, linestyle = '-', alpha = 1, linewidth = 1):
	error_val_up = np.zeros(data.shape[-1]); error_val_low = error_val_up
	if error_measure != None:
		if error_measure[:3]=='avg':
			curve_val = data.mean(axis=0)
		elif error_measure[:3]=='med':
			curve_val = np.median(data, axis=0)

		if error_measure[3:]=='sem':
			error_val_low = curve_val - data.std(axis=0) #/ np.sqrt(data.shape[0])
			error_val_up = error_val_low + 2*curve_val
		elif error_measure[-4:]=='perc':
			if len(percs)==0: percs = [25,75]
			error_val_low = np.percentile(data, percs[0], axis=0)
			error_val_up = np.percentile(data, percs[1], axis=0)

	subp.plot(x, curve_val, color = color, label = label, alpha = alpha, linestyle = linestyle, linewidth = linewidth)
	if error_measure != None:
		subp.fill_between(x, error_val_up, error_val_low, color = color, alpha = .2)
	pl.grid(); pl.ylim(ylim); pl.xlim(xlim)
	if label != None: pl.legend()


insttxts = np.array(['LL', 'RR', 'LR', 'RL'])

# timing of the experiment
srate = 500                                               	#  sampling rate per second
Preinstr_time = int(.2 * srate)                             #  pre-instruction time (1s)
Instr_time = int(.2 * srate)                              	#  instruction presentation (200 ms)
Prep_time = (np.arange(1.7, 2.2, .05) * srate).astype(int)  #  ISD ranging from 1700 to 2200 ms
Stim_time = int(.05 * srate)                              	#  Stimulus presentation of 50 ms
Resp_time = .7 * srate                                     	#  max response time of 1s
FB_time = int(.1 * srate)                                 	#  Feedback presentation of 500 ms
Response_deadline = .7 * srate + 1                        	#  Response deadline

# max trial time
TotT = (Preinstr_time + Instr_time + max(Prep_time) + Stim_time + Resp_time).astype(int)+n_tsteps

# model simulation parameters
sim_path = './'
tiltrate = .05
drift = 0
thresh = 4
Cgs_var_sd = 5
kick_value = .3
MFC_compet_thresh = .1
sigma_compet = .1
sw2 = .4
inh_compet = .1
alpha_compet = .1
theta = 7

nReps = 4

n_trials = 50

behav_data_design, Phase, MFC, Rate, Integr, burst_by_trial, LFC_sync, which_unit_winning, LFC_compet_by_time =\
	Model_sim(Threshold = thresh, drift = drift, Cgs_var_sd = Cgs_var_sd,
		theta_freq = theta, tiltrate = tiltrate, sim_path = sim_path,
		sw2 = sw2, nReps = nReps, kick_value = kick_value,
		MFC_compet_thresh = MFC_compet_thresh, alpha_compet = alpha_compet,
		sigma_compet = sigma_compet, inh_compet=inh_compet, n_trials = n_trials,
		save_behav = False, save_eeg = False, return_eeg = True,
		return_behav = True, print_prog = True)

# behav_data_design: Trials, Design, resp, accuracy, RT, Instruct_lock, Stim_lock, Response_lock
# Design: Instr, Stim, Preparation, Repetition
Design = behav_data_design[:, 1:5].astype(np.int)[:n_trials, :]
RT = behav_data_design[:, 7][:n_trials]
accuracy = behav_data_design[:, 6][:n_trials]

MFC = MFC[..., :n_trials]

which_unit_winning2 = which_unit_winning[:n_trials, :].T.copy().astype(np.float)

# make the winning unit NaN when it's out of compet window
compet_wind_mask = (1 / (1 + np.exp(-10 * (MFC[0,:,:]-1)))) > MFC_compet_thresh
# we put everything before 50ms to nan so that we don't have problems with beginning of the trial
compet_wind_mask[:125, :] = False
which_unit_winning2[np.logical_not(compet_wind_mask)] = np.nan



###### Plot phase neurons and rate neuron of the neural triplet ######

x = np.arange(Phase.shape[-2])/srate - .2

fig, axs = pl.subplots(4, 1)
for ind, i in enumerate(np.arange(4)):
	## task relevant units
	axs[ind].plot(x, Phase[LFC_sync[Design[i,0], 0].astype(np.int), 0, :, i])
	axs[ind].plot(x, Integr[:, :, i].T)
	axs[ind].plot(x, MFC[0, :, i].T, 'r--')

	## task irrelevant units
	# task_irr_units_mask = np.ones(8, dtype=np.bool)
	# task_irr_units_mask[LFC_sync[Design[i,0], :].astype(np.int)] = False
	# axs[i].plot(x, Phase[task_irr_units_mask, 0, :, i].T, color=[0, 0, .5])

	# the burst reaching processing units
	axs[ind].plot(x, burst_by_trial[i, :, 0], 'k', linewidth=3)
	axs[ind].axvline(x=0, color='k')
	axs[ind].axvline(x=.200, color='k')
	axs[ind].axvline(x=(Prep_time[Design[i, 2]] + Instr_time)/srate, color='k')
	axs[ind].axvline(x=(Prep_time[Design[i, 2]] + Instr_time + Resp_time)/srate, color='k')
	axs[ind].axvline(x=RT[i]/1000 + (Prep_time[Design[i, 2]] + Instr_time)/srate, linestyle='--',
		color=['r', 'g'][int(accuracy[i])])
	axs[ind].set_ylabel(insttxts[Design[i,0]])





x = np.arange(Phase.shape[-2])/srate - .2

fig, axs = pl.subplots(4, 1)
for ind, i in enumerate(np.arange(4)):
	rule_relevant_Snode = LFC_sync[Design[i,0], 0].astype(np.int)
	axs[ind].plot(x, Rate[rule_relevant_Snode, :, i]*10, color='k')
	axs[ind].plot(x, MFC[0, :, i].T, 'r--')

	axs[ind].plot(x, Phase[rule_relevant_Snode, 0, :, i], color='r')
	axs[ind].plot(x, Phase[rule_relevant_Snode, 1, :, i], color='b')
	axs[ind].plot(x, Integr[:, :, i].T)
	axs[ind].axvline(x=(Prep_time[Design[i, 2]] + Instr_time)/srate, color='k')
	axs[ind].axvline(x=(Prep_time[Design[i, 2]] + Instr_time + Resp_time)/srate, color='k')
	axs[ind].axvline(x=RT[i]/1000 + (Prep_time[Design[i, 2]] + Instr_time)/srate, linestyle='--',
		color=['r', 'g'][int(accuracy[i])])
	axs[ind].set_ylabel(insttxts[Design[i,0]])





###############################################################################
#########################	 	LFC COMPETITION 		#######################
###############################################################################
x = np.arange(Phase.shape[-2])/srate
start_t = 0; end_t = 1.9
start_tind = np.argwhere(x >= start_t).squeeze()[0]
end_tind = np.argwhere(x <= end_t).squeeze()[-1]
n_tsteps_prep = end_tind-start_tind
timep = np.linspace(start_t, end_t, n_tsteps_prep)

# get competition window data, otherwise all instructions are at 0 so it's meaningless
compet_wind_mask_selec = compet_wind_mask[start_tind:end_tind, :]
# this is just in case different trials have randomly different window sizes
compet_wind_mask_selec = np.floor(compet_wind_mask_selec.astype(np.float).mean(axis=1)).astype(np.bool)

n_tsteps_usef = compet_wind_mask_selec.sum(axis=0)
timep_usef = timep[compet_wind_mask_selec]

n_instr = 4
which_unit_winning_selec = which_unit_winning2[start_tind:end_tind, :]
which_unit_winning_selec = which_unit_winning_selec[compet_wind_mask_selec, :]


count_all = np.zeros(shape = [n_trials, n_instr, n_instr])-1
correct_bytime_all = np.zeros(shape = [n_trials, n_tsteps_usef])-1

for ins_n in np.arange(n_instr):
	ins_mask = Design[:, 0] == ins_n
	winning_all = which_unit_winning_selec[:, ins_mask].T

	# how many time points did each instruction win when "ins_n" was instructed?
	count_all[ins_mask, ins_n, :] = np.array([np.bincount(win_n[win_n!=-5].astype(np.int), minlength=4) for win_n in winning_all])
	# was the most activated rule node the instructed one? (for each time point, each trial)
	correct_bytime_all[ins_mask, :] = np.array([win_n == ins_n for win_n in winning_all]).astype(np.float)

trial_per_inst = np.bincount(Design[:,0], minlength=4)

n_groups = 5
n_per_group_per_inst = np.floor(trial_per_inst/n_groups).astype(np.int)
correct_bytime_all_grouped = np.zeros(shape=[n_groups, n_instr, n_tsteps_usef])
count_all_grouped = np.zeros(shape=[n_groups, n_instr, n_instr])
for ins_n in np.arange(n_instr):
	ins_mask = Design[:, 0]==ins_n
	corr_ins = correct_bytime_all[ins_mask, ...]
	count_ins = count_all[ins_mask,...]
	for gr in np.arange(n_groups):
		correct_bytime_all_grouped[gr, ins_n, :] = corr_ins[(gr*n_per_group_per_inst[ins_n]):((gr+1)*n_per_group_per_inst[ins_n]), :].mean(axis=0)
		count_all_grouped[gr, ins_n, :] = count_ins[(gr*n_per_group_per_inst[ins_n]):((gr+1)*n_per_group_per_inst[ins_n]), ins_n, :].mean(axis=0)

cols = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']


fig, axs = pl.subplots(1, 2)
for ins_n in np.arange(n_instr):
	count_ins = count_all_grouped[:, ins_n, :] * 2
	error_low = np.nanpercentile(count_ins, 5, axis=0)
	error_up = np.nanpercentile(count_ins, 95, axis=0)

	axs[0].plot(np.arange(1, 5)+(ins_n/6)-.25, count_ins.mean(axis = 0),
		'o', label = insttxts[ins_n])
	for ind in np.arange(4):
		x = (np.arange(1, 5)+(ins_n/6)-.25)[ind]
		axs[0].plot([x,x], [error_low[ind], error_up[ind]], 'k-')
	axs[0].set_title('Proportion of winning unit by instruct. by unit', fontsize=10)
	
	plot_shadederr(axs[1], correct_bytime_all_grouped[:, ins_n, :],
		error_measure = 'avg_perc', percs = [5, 95], x = timep_usef, color = cols[ins_n])
	axs[1].set_title('Proportion of winning unit by instr. by time point', fontsize=10)
axs[0].legend()



#########
# check the effect of amplitude and frequency on competition window
srate = 500
r2_MFC = 1
acc_slope = 10

thetas = np.arange(4, 7.1, .5)
n_timep = 1762 # 3.524s at 500Hz
times = np.linspace(0, n_timep/srate, n_timep)
amps = np.arange(1, 4.1, .5)
thresh = .1
n_points_above_thresh = np.zeros(shape = [len(amps), len(thetas)])
for amp_ind in np.arange(len(amps)):
	for theta_ind, theta_freq in enumerate(thetas):
		# damping parameter MFC
		damp_MFC = .005*theta_freq
		r2s = np.zeros(n_timep)
		MFC_temp = np.zeros([2, n_timep]) # 2 units
		MFC_temp[0, 0] = 1
		#coupling theta waves
		Ct = (theta_freq / srate) * 2 * np.pi

		# get the amplitude of theta oscill for that run
		amp = amps[amp_ind]
		for timep in np.arange(n_timep-1):
			MFC_temp[:, timep+1] = phase_updating(Neurons = MFC_temp[:, timep],
												Radius=amp, Damp=damp_MFC,
												Coupling=Ct, multiple=False)
		Be = 1 / (1 + np.exp(-acc_slope * (MFC_temp[0, :]-1)))

		Be_threshed = (Be[500:]>thresh).astype(np.int)
		diff = Be_threshed[1:]-Be_threshed[:-1]
		start_counting = np.argwhere(diff==1).squeeze()[0] - 1
		up_inds = np.argwhere(diff[start_counting:]==1).squeeze()
		down_inds = np.argwhere(diff[start_counting:]==-1).squeeze()
		n_points_above_thresh[amp_ind, theta_ind] =\
			np.max(down_inds - up_inds[:len(down_inds)])

fig, axs = pl.subplots(1, 2)
for amp_ind in np.arange(len(amps)):
	axs[0].plot(thetas, 1000*(n_points_above_thresh[amp_ind, :]/srate),
		'o-', label='%.1f'%amps[amp_ind])
axs[0].set_xlabel('MFC frequency in Hz'); axs[0].set_ylabel('Time interval above threshold (ms)')
axs[0].legend()
axs[0].set_title('Each line represents\none theta amplitude')

for theta_ind in np.arange(len(thetas)):
	axs[1].plot(amps, 1000*(n_points_above_thresh[:, theta_ind]/srate),
		'o-', label='%.1f'%thetas[theta_ind])
axs[1].set_xlabel('Theta Amplitude'); axs[1].set_ylabel('Time interval above threshold (ms)')
axs[1].legend()
axs[1].set_title('Each line represents\none theta frequency')



##########
# burst function based on MFC theta amplitude
#radius MFC
srate = 500
r2_MFC = 1
acc_slope = 10

fig, axs = pl.subplots(3, 1)

thetas = np.arange(2, 12, .5)
# thetas = np.arange(3, 8)
n_timep = 1762 # 3.524s at 500Hz
times = np.linspace(0, n_timep/srate, n_timep)
threshs = np.arange(.1, 1.1, .2)
n_points_above_thresh = np.zeros(shape = [len(threshs), len(thetas)])
for thresh_ind in np.arange(len(threshs)):
	for theta_ind, theta_freq in enumerate(thetas):
		# theta_freq = 8
		#damping parameter MFC
		damp_MFC = .03*theta_freq
		r2s = np.zeros(n_timep)
		MFC_temp = np.zeros([2, n_timep]) # 2 units
		MFC_temp[0, 0] = 1
		#coupling theta waves
		Ct = (theta_freq / srate) * 2 * np.pi

		for timep in np.arange(n_timep-1):
			MFC_temp[:, timep+1] = phase_updating(Neurons = MFC_temp[:, timep],
												Radius=r2_MFC, Damp=damp_MFC,
												Coupling=Ct, multiple=False)

		Be = 1 / (1 + np.exp(-acc_slope * (MFC_temp[0, :]-1)))

		thresh = threshs[thresh_ind]
		Be_threshed = (Be[500:]>thresh).astype(np.int)
		diff = Be_threshed[1:]-Be_threshed[:-1]
		start_counting = np.argwhere(diff==1).squeeze()[0] - 1
		up_inds = np.argwhere(diff[start_counting:]==1).squeeze()
		down_inds = np.argwhere(diff[start_counting:]==-1).squeeze()
		n_points_above_thresh[thresh_ind, theta_ind] =\
			np.max(down_inds - up_inds[:len(down_inds)])

pl.figure()
for thresh_ind in np.arange(len(threshs)):
	pl.plot(thetas, 1000*(n_points_above_thresh[thresh_ind, :]/srate),
		'o-', label='%.1f'%threshs[thresh_ind])
pl.xlabel('MFC frequency in Hz'); pl.ylabel('Time interval above threshold (ms)')
pl.legend()


fig, axs = pl.subplots(2, 1)
axs[0].plot(times, MFC_temp[0, :], label=theta_freq)
axs[0].fill_between(times, np.zeros(len(times)), (Be>.25), facecolor=[1,0,0,.2],
	edgecolor = [1,1,1,0])
axs[1].plot(times, Be, label=theta_freq)
axs[1].fill_between(times, np.zeros(len(times)), (Be>.25), facecolor=[1,0,0,.2],
	edgecolor = [1,1,1,0])
axs[0].set_title('Effect of theta frequency on competition window\n(for MFC threshold of .25)')
axs[0].set_ylabel('MFC exc. unit activity', fontsize=8)
axs[1].set_ylabel('logistic(MFC activity)', fontsize=8)
axs[1].set_xlabel('trial time course (s)', fontsize=8)
axs[0].set_xlim(.8, 1.8); axs[1].set_xlim(.8, 1.8)
# axs[2].plot(times, r2s, label=theta_freq)
# pl.legend([])

pl.plot(Be>thresh)
pl.title('time ratio above 50pc of chance to send a burst: %.2f' % (sum(Be>.5)/len(Be)))


pl.figure()
for thresh in threshs:
	pl.plot(thetas, (30/thetas/30)/(thresh/((thetas/20)+1.9)) + 10, 'o-')
	pl.plot(thetas, (30/thetas/30)/(thresh/((thetas/20)+1.9)) + 10, 'o-')


