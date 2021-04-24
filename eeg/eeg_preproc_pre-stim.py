# Script written by Mehdi Senoussi
# Extract EEG data 1s before stimuli presentation

import mne, os, glob
import numpy as np 
import matplotlib.pyplot as pl
import pandas as pd

mne.set_log_level(50)

def transform_dict_bytes_to_utf8(dict_in_bytes, list_keys):
	dict_in_utf8 = dict()
	for key_n in list_keys:
		list_vals_key_n = np.array([k.decode('utf-8') for k in dict_in_bytes[key_n]])
		dict_in_utf8.update({key_n.decode('utf-8'):list_vals_key_n})
	return dict_in_utf8


data_path = './data/'

# loads the channel file
montage = mne.channels.read_montage(os.path.abspath(data_path) + '/chanlocs_66elecs_noCz.loc')
# gaze threshold for rejecting trials (in degrees) 
fix_thresh_indeg = 1.5

obs_all = np.arange(1, 40)
## participants 5, 15, 9, 16 and 33 were excluded (see Methods)
obs_all = obs_all[np.array([obs_i not in [5, 9, 15, 16, 33] for obs_i in obs_all])]

for obs_i in obs_all:
	obs_path = data_path + 'obs_%i/' % obs_i
	print('\nobs %i' % obs_i)

	##########################################################################################################
	#############################				LOADING DATA			######################################
	##########################################################################################################
	events_block = []
	# loads all the EEG data files
	raws = []
	for block_i in range(1, 9):
		file_n = obs_path + 'eeg/raw/obs%i_block%i.vhdr' % (obs_i, block_i)
		if os.path.exists(file_n):
			print('\tblock %i - file %s' % (block_i, file_n))

			block_temp = mne.io.read_raw_brainvision(file_n, montage = montage,
				eog = ('HEOG', 'VEOG'), preload=True, verbose=50)
			# additional erroneous triggers in block 1 of participant 39
			if (obs_i == 39) & (block_i == 1):
				block_temp.crop(tmin = 0, tmax = 570)
			events_block.append(mne.events_from_annotations(block_temp, verbose=50)[0])

			raws.append(block_temp)
	raw_all = mne.concatenate_raws(raws)


	##########################################################################################################
	####################			RE-REFERENCING AND GETTING Cz back				##########################
	##########################################################################################################

	# get Cz back by adding it as a empty channel (will be useful when rereferencing)
	raw_all2 = mne.add_reference_channels(raw_all, ref_channels = 'Cz', copy = True) 

	## some specific stuff to this dataset because we use a channel location file that doesn't specify the channel type
	types = list(np.repeat(['eeg'], 67))
	# change elements in that list that aren't 'EEG' channels
	types[-2] = 'eog'; types[-3] = 'eog'; types[-4] = 'ecg';
	# create a dictionnary of channel names and types
	chtypes_dict = dict(zip(raw_all2.ch_names, types))
	raw_all2.set_channel_types(chtypes_dict)

	# apply a average reference
	raw_all2 = raw_all2.set_eeg_reference('average', verbose=None)

	# apply a band-pass filter
	raw_all2filt = raw_all2.copy().filter(None, 48, n_jobs = -1, fir_design = 'firwin', verbose=50)

	# store events from the RAW dataset
	events_all, events_all_id = mne.events_from_annotations(raw_all2, verbose=50)

	###### dealing with recording issues
	# take care of some recording issues (e.g. a failed block that left some triggers in the EEG file but that are not in the behavioral files)
	if obs_i == 4:
		events_all = np.concatenate([events_all[:1878, :], events_all[1894:, :]], axis = 0)


	if obs_i == 11:
		# A trigger "10" is missing which corrupts all event processing (EEG recording crashed at first).
		# we add it to the event list (1s before the next event, i.e.
		# 500 time points because sampling rate is 500Hz)
		events_all = np.concatenate([events_all[:636, :],
			np.array([512263-500, 0, 10])[np.newaxis, :],
			events_all[636:, :]], axis = 0)

	if obs_i == 21:
		# 3 triggers are missing at the beginning of block 6 (EEG recording crashed at first).
		# (raw eeg trigger time ~ 1544000, and at index 2279 and 2280)
		# These trials were removed from log data in the eeg_fix directory
		# and do not exist in the log and eyet-"keep_trials" data.
		# We remove these triggers (i.e. ignore these trials)
		events_all = np.concatenate([events_all[:2278, :], events_all[2281:, :]], axis = 0)
	
	if obs_i == 23:
		# One additional trial at the end of block 6 (EEG recording stopped manually)
		# creates a mismatch between log and EEG. We remove these triggers (i.e. ignore this trial)
		events_all = events_all[:-5, :]

	if obs_i == 27:
		# Some triggers are missing at the beginning of block 3 (EEG recording crashed at first).
		# A ~50 trigger is left which creates a mismatch between the number of "10" and "50" triggers.
		# We remove this trigger (i.e. ignore this trial)
		events_all = np.concatenate([events_all[:946, :], events_all[947:, :]], axis = 0)

	if obs_i == 28:
		# One additional trial at the end of block 8 (EEG recording stopped manually)
		# creates a mismatch between log and EEG. We remove these triggers (i.e. ignore this trial)
		events_all = events_all[:-5, :]

	if obs_i == 37:
		# The FIRST trigger of block 1 (a 10) is missing (EEG recording crashed at first).
		# We add this trigger in the event data with a 1s delay (i.e. preceding the following trigger 22).
		events_all = np.concatenate([events_all[:1, :],
			np.array([events_all[1, 0]-500, 0, 10])[np.newaxis, :],
			events_all[1:, :]], axis = 0)

	if obs_i == 39:
		# The FIRST trigger of block 1 (a 10) is missing (EEG recording crashed at first).
		# We add this trigger in the event data with a 1s delay (i.e. preceding the following trigger 22).
		events_all = events_all[:-5, :]

	################################################

	n_evs = len(events_all)
	trial_num = np.zeros(n_evs, dtype = np.int)
	t_num = -1
	for ind, ev in enumerate(events_all):
		t_num += int(ev[-1] in [10])
		trial_num[ind] = t_num

	z = np.load(data_path + 'obs_%i/eyet/obs_%i_thresh%.1fdeg_data_for_eeg_2.npz' %\
		(obs_i, obs_i, fix_thresh_indeg), allow_pickle=True)['arr_0'][..., np.newaxis][0]
	eeg_trials_torej_respEyeT = z['eeg_trials_torej']
	log_data_clean_forEEG = z['logdata_clean_forEEG']
	col_of_interest = np.array([0, 1, 2, 3, 4, 5, 7, 9])

	pd_log_clean = pd.DataFrame(log_data_clean_forEEG[:, col_of_interest],
		columns=['observer', 'block', 'trial', 'instr_type', 'ISD',
		'stimcombi', 'resptime', 'respcorrect'], dtype=np.float)

	trials_to_rej = np.where(eeg_trials_torej_respEyeT)[0]
	inds_to_keep = np.ones(n_evs, dtype = np.bool)
	for trej in trials_to_rej:
		inds_to_keep[np.where(trial_num == trej)[0]] = False

	events_all_clean = events_all[inds_to_keep, :]

	# check that log and EEG data match using the list of trial instructions
	instruction_EEG = events_all_clean[(events_all_clean[:,-1]>19) & (events_all_clean[:,-1]<24), -1]-20
	print('\n\tEEG and log files match: %s' %\
		np.str(np.all(instruction_EEG == pd_log_clean.instr_type.values)))

	########################################################
	## epoching
	epochs = mne.Epochs(raw_all2filt, events_all_clean, event_id = {'++':50, '-+':51, '+-':52, '--':53},
		tmin = -1, tmax = 0, proj = False, baseline = None, reject = {}, decim=2,
		detrend = 1, metadata = pd_log_clean, verbose = 50)

	dropped_eeg_trial_file = obs_path + 'eeg/obs_%i_eeg_trialRej_manual.npy' % obs_i
	if os.path.exists(dropped_eeg_trial_file):
		dropped_eeg_trials = np.load(dropped_eeg_trial_file)
		epochs = epochs.drop(dropped_eeg_trials)
		epochs.load_data()
	else:
		epochs.load_data()
		epochs.plot(n_epochs = 8, n_channels = 66, scalings = dict(eeg=10e-5),
			picks='all', block=True)

		ev_of_interest = np.where((events_all_clean[:, -1] > 49) & (events_all_clean[:, -1] < 54))[0]
		dropped_eeg_trials = np.zeros(len(ev_of_interest), dtype=np.bool)
		for ind, i in enumerate(ev_of_interest):
			dropped_eeg_trials[ind] = len(epochs.drop_log[i]) != 0

		np.save(dropped_eeg_trial_file, dropped_eeg_trials)

	epochs = epochs.interpolate_bads(reset_bads=True)
	print('\tsaving clean EEG...\n\n')
	epochs = epochs.resample(200)
	epochs.pick_types(eeg = True)
	epochs.save(obs_path + 'eeg/obs%i_EEGclean_pre-stim-epo.fif.gz' % obs_i, overwrite = True)




