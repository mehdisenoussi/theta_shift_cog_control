import os, glob
import numpy as np
from matplotlib import pyplot as pl

def plot_shadederr(subp, curve_val, error_val, color = 'blue', x = None, xlim = None, ylim = None, label = None, linestyle = '-', alpha = 1, linewidth = 1):
	subp.plot(x, curve_val, color = color, label = label, alpha = alpha, linestyle = linestyle, linewidth = linewidth)
	if np.any(error_val):
		subp.fill_between(x, curve_val + error_val, curve_val - error_val, color = color, alpha = .2)
	pl.grid(); pl.ylim(ylim); pl.xlim(xlim)
	if label != None: pl.legend()

base_path = '/Volumes/mehdimac/ghent/mystinfo/gitcleandata/'

mrk2trig_codes = dict({b'S  1':1, b'S 10':5, b'S 20':10, b'S 21':10,
	b'S 22':10, b'S 23':10, b'S 30':15, b'S 31':15, b'S 32':15,
	b'S 33':15, b'S 34':15, b'S 35':15, b'S 36':15, b'S 37':15,
	b'S 38':15, b'S 39':15, b'S 40':15, b'S 50':20, b'S 51':20,
	b'S 52':20, b'S 53':20,	b'S 60':25, b'S 61':25, b'S100':30})

########################################################################################################################
### get eyetracking triggers
obs_all = np.arange(1, 40)
# obs 5 and 15 less than 5 blocks, obs 9 left-handed 
obs_all = obs_all[np.array([obs_i not in [5, 9, 15] for obs_i in obs_all])]


for obs_i in obs_all:
	eyet_path = base_path + 'obs_%i/eyet/' % obs_i
	print('obs_%i' % obs_i)
	first_block = True
	for block in np.arange(1, 9):
		
		eyet_recon_path = eyet_path + 'recon_eeg_trig/'
		if not os.path.exists(eyet_recon_path):
			os.mkdir(eyet_recon_path)
		fp = eyet_path + 'obs%i_block%i_eyet.txt' % (obs_i, block)
		if os.path.exists(fp):
			first_block = False
			print('\tblock_%i: YES' % block)
			# open eyetracking file
			f = open(fp, 'r')
			raw = f.readlines()
			f.close()
			# how many columns does it have
			n_cols = len(raw[-1].split('\t'))

			# make array to store data and loop over each line to check whether
			# it is of interest (e.g. skipping header)
			processed = np.zeros(shape = [len(raw), n_cols], dtype='|S22')
			dataline_counter = 0
			for i in np.arange(len(raw)):
				# string to list
				line = raw[i].replace('\n','').replace('\r','').split('\t')
				
				# check if the line starts with '##' (denoting header)
				if '##' in line[0]:
					# skip processing
					continue
				elif '##' not in line[0]:
					processed[dataline_counter, ] = np.array(line)
					dataline_counter += 1

			processed = processed[:dataline_counter, :]

			# get the triggers and transform them because they get encoded by the SMI
			trigs_smi = processed[1:, -3].astype(np.float)
			trigs = np.zeros(len(trigs_smi))
			trig_codes = np.array([[3200, 0], [3208, 1], [3240, 5], [3280, 10], [3320, 15], [3104, 20], [3144, 25]])
			for trigi in trig_codes:
				trigs[trigs_smi == trigi[0]] = trigi[1]


			raw_trig_timep = processed[1:, 0].astype(np.float)
			t_zero = raw_trig_timep[0]

			first_trialstart_t = raw_trig_timep[np.where(trigs==5)[0][0]] - t_zero
			first_instON_t = raw_trig_timep[np.where(trigs == 10)[0][0]] - t_zero
			first_instOFF_t = raw_trig_timep[np.where(trigs == 15)[0][0]] - t_zero

			# transform trigger timings to seconds and to start at 0 with first trigger
			trig_timep = (raw_trig_timep - t_zero) / 10**6


			########################################################################
			### get EEG triggers to match eyetracking data
			fp = base_path + 'obs_%i/eeg/raw/obs%i_block%i.vmrk' % (obs_i, obs_i, block)

			f = open(fp, 'r')
			raw_vmrk = f.readlines()
			f.close()

			n_cols = len(raw_vmrk[40].replace('\n', '').replace('\r', '').split(','))

			processed_mrk = np.zeros(shape = [len(raw_vmrk), n_cols], dtype='|S22')
			dataline_counter = 0
			for i in np.arange(len(raw_vmrk)):
				line = raw_vmrk[i].replace('\n', '').replace('\r', '').split(',')
				
				# check if the line starts with 'Mk' (denoting trigger/marker)
				if (len(line[0]) != 0) & (line[0][:2] == 'Mk') & ('New Segment' not in line[0]):
					processed_mrk[dataline_counter, ] = np.array(line)
					dataline_counter += 1
			processed_mrk = processed_mrk[:dataline_counter, :]

			raw_mrks = processed_mrk[:, 1]
			mrks_as_trigs = np.zeros(len(raw_mrks))

			for codei in mrk2trig_codes.keys():
				mrks_as_trigs[raw_mrks == codei] = mrk2trig_codes[codei]
			
			# get the raw sampled time points of the eeg recording
			mrk_timep = processed_mrk[:, 2].astype(np.int)/500.



			# In block 5 of obs 4: a few EEG triggers are in the file but not in log or eyet file.
			# (the stim script was restarted but EEG recording was not, creating a mismatch)
			# We added 'fake' trials in log to match EEG and log (for EEG analysis scripts)
			# and we take them out here.
			if (obs_i == 4) & (block == 5):
				mrk_timep = mrk_timep[15:] - mrk_timep[16]
				mrks_as_trigs = mrks_as_trigs[15:]

			# In block 1 of obs 37: an EEG trigger is missing due to crash of EEG software, we add it here.
			if (obs_i == 37) & (block == 1):
				mrk_timep = np.concatenate([[0], mrk_timep])
				mrks_as_trigs = np.concatenate([[5], mrks_as_trigs])
			
			# for block 1 of obs 39, 3 additional EEG triggers are at the end of the EEG file 
			# because the block was stopped manually. We take out these 3 triggers.
			if (obs_i == 39) & (block == 1):
				mrk_timep = mrk_timep[:-3]
				mrks_as_trigs = mrks_as_trigs[:-3]


			

			# if the first EEG trigger is a 0 it means that it's the beginning of the block
			# therefore we can remove it and realign time points to it to make it align with the eyet data
			if mrks_as_trigs[0] == 1:
				mrk_timep = mrk_timep[1:]
				mrks_as_trigs = mrks_as_trigs[1:]
			elif mrks_as_trigs[0] != 5:
				print('\t\t\t\tHOUSTON WE HAVE A PROBLEM!')
			# realign EEG triggers time points to the beginning of the first trial
			mrk_timep -= mrk_timep[0]

			# if last EEG trigger is a 30 (end of block) remove it from EEG triggers
			# time points and trigger list (mrk_as_trigs) to fit with the eyet data
			if mrks_as_trigs[-1] == 30:
				mrk_timep = mrk_timep[:-1]
				mrks_as_trigs = mrks_as_trigs[:-1]


			#################################################
			# Get average lag between EEG and eyet triggers
			#################################################

			# only event triggers (remove zeros)
			trigs_noz = trigs[trigs > 0].astype(np.int)

			# where are triggers repeated
			reps = np.hstack([False, trigs_noz[1:] == trigs_noz[:-1]])
			# where there was no repetition (n-1 != n)
			no_reps = np.logical_not(reps)
			# get the non-repetitions indices
			no_reps_inds = np.where(no_reps)[0]
			# only keep non-repeated triggers
			trigs_noz_norep = trigs_noz[no_reps]
			# get the time stamps of the non-repeated triggers
			trig_timep_noz = trig_timep[trigs>0]

			a = trig_timep_noz[no_reps_inds]

			# remove the additional EyeT triggers due to block manually stopped
			if (obs_i == 39) & (block == 1):
				a = a[:-13]

			b = mrk_timep
			len_trigs = len(a)
			lags = []
			for i in np.arange(50):
				temp = a[:(len_trigs-i)] - b[i:len_trigs]
				temp = temp[(temp < .1) & (temp > 0)]
				lags.append(temp.mean())

			avg_lag = np.nanmean(lags)



			########################################################################################################################
			### use EEG triggers to insert in eyet gaze position data

			# transform EEG triggers time points (that are relative to first trial start (trigger 5)) into EyeT triggers:
			# - shift to the first time point of eyet recording (t_zero)
			# - multiply by 10**6 to get seconds
			
			# WE DONT DO THAT ANYMORE, WE TAKE CARE OF IT BEFORE - and only take the first
			# trial start trigger (5) and the block end trigger (30?)
			transf_eeg_timep = (((mrk_timep + avg_lag) * 10**6) + t_zero)
			# for each EEG trigger find the closest raw eyet time point
			corresp_eyet_timep_inds = np.array([np.argwhere(t >= raw_trig_timep)[-1] + 1 for t in transf_eeg_timep]).squeeze()

			new_raw_trig_chan = np.zeros(len(raw_trig_timep), dtype = np.int32)
			new_raw_trig_chan[corresp_eyet_timep_inds] = mrks_as_trigs.astype(np.int)

			# recreate EyeT data with matched EEG triggers.
			recon_eyet_trigs = np.vstack([raw_trig_timep, new_raw_trig_chan,
				processed[1:, 21].astype(np.float32), processed[1:, 22].astype(np.float32)])

			# np.savetxt(fname = eyet_recon_path + 'obs%i_block%i_recon_eyepos.txt' % (obs_i, block),
			# 	X = recon_eyet_trigs.T, fmt = '%i\t%i\t%.2f\t%.2f')

			np.savez(eyet_recon_path + 'obs%i_block%i_recon_eyepos.npz' % (obs_i, block),
				{'recon_eyet_trigs':recon_eyet_trigs.T, 'eeg_mrks':mrks_as_trigs,
				'trigs_noz_norep':trigs_noz_norep, 'eyet_trig_timep_noz':trig_timep_noz,
				'eyet_t_zero':t_zero})

		else:
			print('\tblock_%i: - - Not there - -' % block)





########################################################################################################################
### make epochs and concatenate files
# 500ms before instruction onset
epoch_tstart = -.5
# 800ms after the stim presentation for the longest ISD (2200ms)
# for the shortest ISDs it means that it's 1300ms after stim onset
epoch_tend = 3.2

# sampling rate of the eyetracker
srate = 250.
# number of samples needed to have "epoch_tstart" seconds pre-trial onset (i.e. instruction onset)
# and "epoch_tend" seconds post-trial onset
pre_samples = np.ceil(np.abs(epoch_tstart) / (1./srate)).astype(np.int)
post_samples = np.ceil(np.abs(epoch_tend) / (1./srate)).astype(np.int)

for obs_i in obs_all:
	print('obs_%i' % obs_i)
	# location of reconstructed eyet data from EEG triggers
	eyet_recon_path = base_path + 'obs_%i/eyet/recon_eeg_trig/' % obs_i
	eyetrecon_files = glob.glob(eyet_recon_path + '*.npz')

	# location of behavioral log files
	log_data_path = base_path + 'obs_%i/behav/' % obs_i

	first_block = True
	for file_n in eyetrecon_files:
		# get the block number from the eyet file name
		block = int(file_n.split('_')[-3][-1])
		
		# load the eyet trigger reconstructed file
		z = np.load(file_n, allow_pickle=True)['arr_0'][..., np.newaxis][0]
		# get the variable of interest
		recon_eyet_trigs = z['recon_eyet_trigs']
		t_zero = z['eyet_t_zero']

		# get the time in seconds to epoch in seconds
		eyet_timep_inSec = (recon_eyet_trigs[:, 0] - t_zero) / 10**6
		# find all the instruction onset triggers
		trial_start_trigs_inds = np.argwhere(recon_eyet_trigs[:, 1] == 10).squeeze()
		# find the time point in seconds at which each the instruction trigger appeared
		timep_trial_start = eyet_timep_inSec[trial_start_trigs_inds].squeeze()

		# create lists to store gaze position, time points in seconds, raw time points and triggers 
		eye_pos_all = []
		eyet_timep_all = []
		eyet_rawtimep_all = []
		trig_all = []
		# for each trigger store the length in number of samples (in order to equalize their length (should be ±1 sample))
		epochs_len = np.zeros(len(trial_start_trigs_inds), dtype = np.int)
		for trial_ind, t in enumerate(timep_trial_start):
			# create the sample mask to select only time points which were in a trial epoch based on 
			# the epoch length and the eyeT sampling rate
			epoch_mask = np.zeros(len(eyet_timep_inSec), dtype = np.bool)
			start_ind = trial_start_trigs_inds[trial_ind] - pre_samples
			end_ind = trial_start_trigs_inds[trial_ind] + post_samples
			epoch_mask[start_ind:end_ind] = True


			eye_pos_all.append(recon_eyet_trigs[epoch_mask, 2:])
			eyet_rawtimep_all.append(recon_eyet_trigs[epoch_mask, 0])
			eyet_timep_all.append(eyet_timep_inSec[epoch_mask] - t)
			trig_all.append(recon_eyet_trigs[epoch_mask, 1].astype(np.int32))
			epochs_len[trial_ind] = epoch_mask.sum()

		min_epoch_len = epochs_len.min()

		for trial_ind in np.arange(len(eye_pos_all)):
			eye_pos_all[trial_ind] = eye_pos_all[trial_ind][:min_epoch_len]
			eyet_timep_all[trial_ind] = eyet_timep_all[trial_ind][:min_epoch_len]
			trig_all[trial_ind] = trig_all[trial_ind][:min_epoch_len]

		timep_eyet_trial = np.array(eyet_timep_all).mean(axis = 0)
		# data dimensions: [epochs, samples, data[time_from_inst_onset, triggers, eyeposX, eyeposY]]  
		eyet_epoch_data = np.concatenate([np.array(trig_all)[..., np.newaxis], np.array(eye_pos_all)], 2)





		##############################################################################################################
		# Load log file to check trial correspondency
		# if obs_i in [23, 28, 33, 39]:
		if obs_i in [11, 21, 23, 28, 33, 39]:
			if os.path.exists(log_data_path + 'eeg_fix/'):
				log_data_path += 'eeg_fix/'
		log_fname = glob.glob(log_data_path + '*block%i*.txt' % block)
		log_data = np.loadtxt(log_fname[0], dtype = np.str)

		print('block: eyet-%i / log-%i' % (block, int(log_data[1, 1])))
		# print(('%i timep' % len(timep_eyet_trial)))
		neyet_trials = eyet_epoch_data.shape[0]
		nlog_trials = log_data.shape[0]-1
		goodtxt = 'GOED!'
		if neyet_trials != nlog_trials: goodtxt = '- - NIET GOED!! - -'
		print('\tn_trials: eyet-%i / log-%i\t\t%s' % (neyet_trials, nlog_trials, goodtxt))

		##############################################################################################################

		# get responded trials indices from log file
		responded_mask = log_data[1:, -3].astype(np.float) != -1

		# concatenate blocks and create array for block-"numbering" each trial
		block_info = block * np.ones(neyet_trials, dtype = np.int)
		if first_block:
			first_block = False
			eyet_epochs_allblocks = eyet_epoch_data
			block_n_all = block_info
			responded_mask_all = responded_mask
		else:
			eyet_epochs_allblocks = np.concatenate([eyet_epochs_allblocks, eyet_epoch_data], axis = 0)
			block_n_all = np.concatenate([block_n_all, block_info], axis = 0)
			responded_mask_all = np.concatenate([responded_mask_all, responded_mask], axis = 0)


	# # save concatenated eyet data across blocks
	# np.savez(base_path + 'obs_%i/eyet/eyet_epochs_allblocks.npz' % obs_i,
	# 	{'eyet_epochs_allblocks':eyet_epochs_allblocks, 'block_n_all':block_n_all})

	# save only trials in which observer responded (as "..clean.npz") to have a ready-to-use dataset
	print('\t\tn_resp_trials: %i/%i (%.2f)' % (sum(responded_mask_all),
		len(responded_mask_all), np.mean(responded_mask_all)))
	eyet_epochs_allblocks_clean = eyet_epochs_allblocks[responded_mask_all, ...]
	block_n_all_clean = block_n_all[responded_mask_all]
	np.savez(base_path + 'obs_%i/eyet/eyet_epochs_allblocks_clean.npz' % obs_i,
		{'eyet_epochs_allblocks_clean':eyet_epochs_allblocks_clean,
		'block_n_all_clean':block_n_all_clean})	









