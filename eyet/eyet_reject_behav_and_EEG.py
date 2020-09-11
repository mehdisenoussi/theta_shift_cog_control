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

########################################################################################################################
obs_all = np.arange(1, 40)
# obs 5 and 15 have less than 5 blocks, obs 9 left-handed
obs_all = obs_all[np.array([obs_i not in [5, 9, 15] for obs_i in obs_all])]

timep_orig = np.linspace(-.5, 3.2, 925)
isds = np.arange(1.7, 2.2, .05)
isds_tmask = np.array([(timep_orig > -0.1) & (timep_orig < (i+.2)) for i in isds])

screen_size_pix = np.array([1920, 1080])
width, length = [53, 30] # in cm
d = 60
pixincm = width/screen_size_pix[0]
# the size of pixels in degrees of visual angle.
# This is needed to compute the size of eye movements in degrees of visual angle (for trial rejection)
pixindeg = 360./np.pi * np.tan(pixincm/(2*d))

# 500ms before instruction onset
epoch_tstart = -.5
# 800ms after the stim presentation for the longest ISD (2200ms)
# for the shortest ISDs it means that it's 1300ms after stim onset
epoch_tend = 3.2

# threshold for rejecting trial in degrees
fix_thresh_indeg = 1.5
fix_thresh_inpix = fix_thresh_indeg/pixindeg

threshold_points = np.tile(np.array([[screen_size_pix[0]/2., screen_size_pix[1]/2.]]), (4, 1)) +\
						np.array([[-fix_thresh_inpix, -fix_thresh_inpix], [fix_thresh_inpix, -fix_thresh_inpix],\
							[-fix_thresh_inpix, fix_thresh_inpix], [fix_thresh_inpix, fix_thresh_inpix]])

fields = np.array(['observer', 'block', 'trial', 'instr_type', 'ISD',
	'stimcombi', 'response', 'resptime', 'resptime2', 'respcorrect'], dtype='<U11')


rej_trial_byobs = {}
behav_eyetclean = []
save_data = True
for obs_i in obs_all:
	# to store all eyet data  
	eyet_centr_all = []

	# location of reconstructed eyet data from EEG triggers
	eyet_recon_path = base_path + 'obs_%i/eyet/recon_eeg_trig/' % obs_i
	eyetrecon_files = glob.glob(eyet_recon_path + '*.npz')

	# load the eyet trigger reconstructed file
	z = np.load(base_path + 'obs_%i/eyet/eyet_epochs_allblocks_clean.npz' % obs_i, allow_pickle=True)['arr_0'][..., np.newaxis][0]
	# get the variable of interest
	eyet_epochs_allblocks_clean = z['eyet_epochs_allblocks_clean']
	block_n_all_clean = z['block_n_all_clean']
	usable_blocks = np.unique(block_n_all_clean)

	# location of behavioral log files
	log_data_path = base_path + 'obs_%i/behav/' % obs_i
	if obs_i in [33, 39]:
		# for participants 33 and 39 there was a mismatch between log data
		# and EEG/EyeT triggers for a few trials (because some blocks were
		# stopped manually). We thus need to use the "fixed" log files
		log_data_path += 'eeg_fix/'

	for block_ind, block in enumerate(usable_blocks):
		log_fname = glob.glob(log_data_path + 'task_obs%s_block%i_date*.txt' % (obs_i, block))[0]
		log_data = np.loadtxt(log_fname, dtype = np.str)
		if not block_ind: logdata_all = log_data[1:, :]
		else: logdata_all = np.concatenate([logdata_all, log_data[1:, :]], axis = 0)
	responded_mask = logdata_all[:, -3].astype(np.float) != -1
	logdata_all_clean = logdata_all[responded_mask, :]

	# get useful stuff from log
	isds_all = logdata_all_clean[:, 4].astype(np.int)

	n_trials = block_n_all_clean.shape[0]
	rej_trial_eyet = np.zeros(n_trials, dtype = np.bool)
	trial_data_gaze_inpix_all_x = np.array([])
	trial_data_gaze_inpix_all_y = np.array([])
	for trial_n in np.arange(n_trials):
		trial_data = eyet_epochs_allblocks_clean[trial_n, isds_tmask[isds_all[trial_n]], 1:]
		# keep the gaze position in raw pix values to plot it on the experimental display
		trial_data_gaze_inpix_all_x = np.hstack([trial_data_gaze_inpix_all_x, trial_data[:, 0]])
		trial_data_gaze_inpix_all_y = np.hstack([trial_data_gaze_inpix_all_y, trial_data[:, 1]])

		# centered on gaze position before trial onset
		trial_data_centered = np.array([trial_data[:, 0] - trial_data[0, 0], trial_data[:, 1] - trial_data[0, 1]])
		
		eyet_centr_all.append(trial_data_centered)
		gaze_pos_deg = trial_data_centered * pixindeg
		rej_trial_eyet[trial_n] = np.any(gaze_pos_deg > fix_thresh_indeg)

	print('obs_%i:\t%i trials left' %\
		(obs_i, n_trials - rej_trial_eyet.sum()))

	rej_trial_byobs['obs_%i' % obs_i] = [rej_trial_eyet.sum(), n_trials, rej_trial_eyet.mean()]

	logdata_allclean_noEyeMov = logdata_all_clean[np.logical_not(rej_trial_eyet), :]

	log_data_path = base_path + 'obs_%i/behav/' % obs_i
	data_all_struct = {}
	for ind, field in enumerate(fields):
		data_all_struct[field] = logdata_allclean_noEyeMov[:, ind]

	# save clean behav data (only responded trials, only trials without
	# gazes outside 1.5° radius of fixation)
	np.save(base_path +\
		'obs_%i/behav/obs_%i_behav_data_eyet_thresh%.1fdeg_struct.npy' %\
		(obs_i, obs_i, fix_thresh_indeg), data_all_struct)

	# save clean, i.e. responded, behav data and which trials need to be
	# rejected because gaze was >1.5° from fixation
	np.savez(base_path + 'obs_%i/eyet/obs_%i_thresh%.1fdeg_data_for_eeg.npz' %\
		(obs_i, obs_i, fix_thresh_indeg),
		{'rej_trial_eyet':rej_trial_eyet, 'logdata_all_clean':logdata_all_clean})




########################################################################
######					MATCH EyeT DATA WITH EEG 				  ######
########################################################################

obs_all = np.arange(1, 40)
## excluded participants
# obs 5 and 15 have less than 5 blocks, obs 9 left-handed
# obs 16, 23 and 33 have less than 200 trials after rejection based on EyeT
# obs_all = obs_all[np.array([obs_i not in [5, 9, 15, 16, 23, 33] for obs_i in obs_all])]
obs_all = obs_all[np.array([obs_i not in [5, 9, 15] for obs_i in obs_all])]

obs_all=np.array([36])
n_trials_max = np.zeros(len(obs_all), dtype = np.int)
trial_present_all = []
responded_mask_log_all = []
responded_mask_logeyet = []
good_obs = []
trials_all_info_all = [] 

rej_trial_byobs = {}
behav_eyetclean = []
save_data = True


for obs_ind, obs_i in enumerate(obs_all):
	print('obs_%i' % obs_i)

	log_data_path = base_path + 'obs_%i/behav/' % obs_i
	eyet_recon_path = base_path + 'obs_%i/eyet/recon_eeg_trig/' % obs_i
	eeg_data_path = base_path + 'obs_%i/eeg/raw/' % obs_i

	############### EYET ###############
	eyetrecon_files = glob.glob(eyet_recon_path + '*.npz')
	# load the eyet trigger reconstructed file
	z = np.load(base_path + 'obs_%i/eyet/eyet_epochs_allblocks.npz' % obs_i,
		allow_pickle = True)['arr_0'][..., np.newaxis][0]
	block_n_all = z['block_n_all']
	usable_blocks = np.unique(block_n_all)
	#####################################


	############### BEHAV ###############
	log_files = glob.glob(log_data_path + '*.txt')
	for ind, log_fname in enumerate(log_files):
		log_data = np.loadtxt(log_fname, dtype = np.str)
		if not ind: log_all_data = log_data[1:, :]
		else: log_all_data = np.concatenate([log_all_data, log_data[1:, :]], axis = 0)

	log_block_all = log_all_data[:, 1].astype(np.int)
	log_trial_all = log_all_data[:, 2].astype(np.int)
	responded_mask_log_all.append(log_all_data[:, -3].astype(np.float) != -1)


	############### EyeT ###############
	for block_ind, block in enumerate(usable_blocks):
		# for participant 39 block 1 and 7 were stopped manually, creating a mismatch
		# between log and EEG/EyeT triggers (last trial not written in log but triggers
		# sent to EEG and EyeT). We fixed that in 'eeg_fix/' log files.
		if obs_i in [33, 39]:
			logeyet_fname = glob.glob(log_data_path + 'eeg_fix/task_obs%s_block%i_date*.txt' % (obs_i, block))[0]
		else: logeyet_fname = glob.glob(log_data_path + 'task_obs%s_block%i_date*.txt' % (obs_i, block))[0]
		logeyet_data = np.loadtxt(logeyet_fname, dtype = np.str)
		if not block_ind:
			logeyet_all_data = logeyet_data[1:, :]
		else:
			logeyet_all_data = np.concatenate([logeyet_all_data, logeyet_data[1:, :]], axis = 0)
	logeyet_block_all = logeyet_all_data[:, 1].astype(np.int)
	logeyet_trial_all = logeyet_all_data[:, 2].astype(np.int)
	responded_mask_logeyet.append(logeyet_all_data[:, -3].astype(np.float) != -1)


	############### EEG ###############
	### behav file for eeg ###
	if os.path.exists(log_data_path + 'eeg_fix/'):
		log_files = glob.glob(log_data_path + 'eeg_fix/*.txt')
		for ind, log_fname in enumerate(log_files):
			logeeg_data = np.loadtxt(log_fname, dtype = np.str)
			if not ind:
				logeeg_all_data = logeeg_data[1:, :]
			else:
				logeeg_all_data = np.concatenate([logeeg_all_data, logeeg_data[1:, :]], axis = 0)
	else:
		logeeg_all_data = log_all_data
	logeeg_block_all = logeeg_all_data[:, 1].astype(np.int)
	logeeg_trial_all = logeeg_all_data[:, 2].astype(np.int)
	
	# In block 5 of obs 4: a few EEG triggers are in the file but not in log or eyet file.
	# (the stim script was restarted but EEG recording was not, creating a mismatch)
	# We added 'fake' trials in log to match EEG and log (for EEG analysis scripts)
	# and we take them out here.
	logeeg_block_all = logeeg_block_all[logeeg_trial_all != -1]
	logeeg_trial_all = logeeg_trial_all[logeeg_trial_all != -1]


	#####################################
	# amount of trials in log data (the max of the max)
	n_trials_max[obs_ind] = log_block_all.shape[0]
	# array to store whether a trial in the raw log data was present in EEG and EyeT
	trial_present = np.zeros(shape = [n_trials_max[obs_ind], 2], dtype = np.bool)
	trials_all_info = np.zeros(shape=[n_trials_max[obs_ind], 4])-2
	for trial_n in np.arange(n_trials_max[obs_ind]):
		# all logs
		trials_all_info[trial_n, 2:4] = log_trial_all[trial_n], log_block_all[trial_n]

		# eyet
		inds = np.argwhere(log_block_all[trial_n] == logeyet_block_all).squeeze()
		if len(inds) != 0:
			trial_present[trial_n, 0] = np.any(log_trial_all[trial_n] == logeyet_trial_all[inds])

		# eeg
		inds = np.argwhere(log_block_all[trial_n] == logeeg_block_all).squeeze()
		if len(inds) != 0:
			trial_present[trial_n, 1] = np.any(log_trial_all[trial_n] == logeeg_trial_all[inds])

		trials_all_info[trial_n, :2] =  trial_present[trial_n, :]

	trial_present_all.append(trial_present)
	trials_all_info_all.append(trials_all_info)

	
	##################################################################################################
	##################			Computer gaze pos. and TRIAL REJECTION 			######################
	##################################################################################################

	# to store all eyet data  
	eyet_centr_all = []

	# location of reconstructed eyet data from EEG triggers
	eyet_recon_path = base_path + 'obs_%i/eyet/recon_eeg_trig/' % obs_i
	eyetrecon_files = glob.glob(eyet_recon_path + '*.npz')

	# load the eyet trigger reconstructed file
	z = np.load(base_path + 'obs_%i/eyet/eyet_epochs_allblocks_clean.npz' % obs_i,
		allow_pickle=True)['arr_0'][..., np.newaxis][0]
	# get the variable of interest
	eyet_epochs_allblocks_clean = z['eyet_epochs_allblocks_clean']
	block_n_all_clean = z['block_n_all_clean']
	usable_blocks = np.unique(block_n_all_clean)

	# location of behavioral log files
	log_data_path = base_path + 'obs_%i/behav/' % obs_i
	if obs_i in [33, 39]:
		# for participants 33 and 39 there was a mismatch between log data
		# and EEG/EyeT triggers for a few trials (because some blocks were
		# stopped manually). We thus need to use the "fixed" log files
		log_data_path += 'eeg_fix/'

	for block_ind, block in enumerate(usable_blocks):
		log_fname = glob.glob(log_data_path + 'task_obs%s_block%i_date*.txt' % (obs_i, block))[0]
		log_data = np.loadtxt(log_fname, dtype = np.str)
		if not block_ind: logdata_all = log_data[1:, :]
		else: logdata_all = np.concatenate([logdata_all, log_data[1:, :]], axis = 0)
	responded_mask = logdata_all[:, -3].astype(np.float) != -1
	logdata_all_clean = logdata_all[responded_mask, :]

	# get useful stuff from log
	isds_all = logdata_all_clean[:, 4].astype(np.int)

	n_trials = block_n_all_clean.shape[0]
	rej_trial_eyet = np.zeros(n_trials, dtype = np.bool)
	trial_data_gaze_inpix_all_x = np.array([])
	trial_data_gaze_inpix_all_y = np.array([])
	for trial_n in np.arange(n_trials):
		trial_data = eyet_epochs_allblocks_clean[trial_n, isds_tmask[isds_all[trial_n]], 1:]
		# keep the gaze position in raw pix values to plot it on the experimental display
		trial_data_gaze_inpix_all_x = np.hstack([trial_data_gaze_inpix_all_x, trial_data[:, 0]])
		trial_data_gaze_inpix_all_y = np.hstack([trial_data_gaze_inpix_all_y, trial_data[:, 1]])

		# centered on gaze position before trial onset
		trial_data_centered = np.array([trial_data[:, 0] - trial_data[0, 0], trial_data[:, 1] - trial_data[0, 1]])
		
		eyet_centr_all.append(trial_data_centered)
		gaze_pos_deg = trial_data_centered * pixindeg
		rej_trial_eyet[trial_n] = np.any(gaze_pos_deg > fix_thresh_indeg)



	# clean log data
	if obs_i not in [33]:
		log_trials_torej = responded_mask_log_all[obs_ind].copy()
		maskk = responded_mask_log_all[obs_ind] & trials_all_info_all[obs_ind][:, 0].astype(np.bool)
		log_trials_torej[maskk] = rej_trial_eyet
		logdata_allclean_noEyeMov = log_all_data[np.logical_not(log_trials_torej), :]
	else:
		log_trials_torej = responded_mask_logeyet[obs_ind].copy()
		maskk = responded_mask_logeyet[obs_ind]
		log_trials_torej[maskk] = rej_trial_eyet
		logdata_allclean_noEyeMov = logeyet_all_data[np.logical_not(log_trials_torej), :]
	


	data_all_struct = {}
	for ind, field in enumerate(fields):
		data_all_struct[field] = logdata_allclean_noEyeMov[:, ind]

	print('\t%i log trials left' %\
		(n_trials_max[obs_ind] - log_trials_torej.sum()))


	# clean eeg data
	if obs_i not in [33]:
		eeg_trials_pres = trials_all_info_all[obs_ind][:, 1].astype(np.bool)
		eeg_trials_torej = log_trials_torej[eeg_trials_pres]
		logdata_forEEG_allclean_noEyeMov = log_all_data[eeg_trials_pres, :][np.logical_not(eeg_trials_torej)]
	else:
		eeg_trials_torej = log_trials_torej
		logdata_forEEG_allclean_noEyeMov = logeyet_all_data[np.logical_not(eeg_trials_torej), :]

	######	
	# save clean behav data (only responded trials, only trials without
	# gazes outside 1.5° radius of fixation)
	np.save(base_path +\
		'obs_%i/behav/obs_%i_behav_data_eyet_thresh%.1fdeg_struct.npy' %\
		(obs_i, obs_i, fix_thresh_indeg), data_all_struct)

	# save clean, i.e. responded, behav data and which trials need to be
	# rejected because gaze was >1.5° from fixation
	np.savez(base_path + 'obs_%i/eyet/obs_%i_thresh%.1fdeg_data_for_eeg.npz' %\
		(obs_i, obs_i, fix_thresh_indeg),
		{'eeg_trials_torej':eeg_trials_torej, 'logdata_clean_forEEG':logdata_forEEG_allclean_noEyeMov})







	#####################################



##### which trials to use in EEG
keep_trial_all = []
keep_trial_eeg = []
for obs_ind, obs_i in enumerate(obs_all):
	# !!!!!!!!!!!!!!!!!!!!!!!!
	# CHECK WHETHER THE EEG TRIALS "TO KEEP" are the correct ones because this bit of
	# code here was using "obs_%i_data_for_classif.npz" NOT obs_%i_thresh1.5deg_data_for_classif.npz !!!!!!
	# !!!!!!!!!!!!!!!!!!!!!!!!
	# z = np.load(base_path + 'obs_%i/eyet/obs_%i_data_for_classif.npz' % (obs_i, obs_i))['arr_0'][..., np.newaxis][0]

	z = np.load(base_path + 'obs_%i/eyet/obs_%i_thresh%.1fdeg_data_for_eeg.npz' % (obs_i, obs_i, fix_thresh_indeg),
		allow_pickle=True)['arr_0'][..., np.newaxis][0]

	print('obs %i - n_trials_remaining: %i' % (obs_i, np.logical_not(z['rej_trial_eyet']).sum()))

	keep_trial_all.append(np.ones(n_trials_max[obs_ind], dtype = np.bool))
	keep_trial_all[obs_ind][np.logical_not(responded_mask_log_all[obs_ind])] = False
	# take the eyet trials which were responded and attribute them the
	# "rejected eyet trial" value (basically indicating whether there
	# was a saccade in that trial)
	keep_trial_all[obs_ind][trial_present_all[obs_ind][:, 0] & responded_mask_log_all[obs_ind]] = np.logical_not(z['rej_trial_eyet'])

	keep_trial_eeg.append(keep_trial_all[obs_ind][trial_present_all[obs_ind][:, 1]])

	np.save(base_path + 'obs_%i/eeg/obs_%i_trials_to_analyze_responded_and_1.5deg_eyet_thresh.npy' %\
		(obs_i, obs_i), keep_trial_eeg[obs_ind])






