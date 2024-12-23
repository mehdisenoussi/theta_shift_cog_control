from psychopy import visual, event, core
import os, time, glob
import numpy as np
import time
from psychopy.visual import GratingStim

###################
# Log stimulation #
###################

def init_log(obs,block,timeAndDate):
	"""init log files"""
	outfolder = './results/obs_%s/' % obs
	old_subject = os.path.isdir(outfolder)
	if not old_subject:
		os.makedirs(outfolder)

	# init task response file
	output_task_file = '%stask_obs%s_block%i_date%s.txt' % (outfolder, obs, block, timeAndDate)
	task_out = open(output_task_file, 'a')

	task_out.write('observer\tblock\ttrial\tinstr_type\tISI\tstimcombi\tresponse\tresptime\tresptime2\trespcorrect\n')
	
	return task_out, outfolder


#################################
# Create the paradigm structure #
#################################

def make_mystinfo_design(n_instructions = 4, n_ISIs = 10, n_stimCombi = 4, block = 1, hash_ISIs_by_block = False):

	if hash_ISIs_by_block:
		if block%2: ISI_inds = np.arange(0, n_ISIs, 2)
		else: ISI_inds = np.arange(1, n_ISIs, 2)
		n_ISIs = ISI_inds.shape[0]

	n_trials = n_instructions*n_ISIs*n_stimCombi
	trials_info = np.zeros(shape=[n_trials, 3])
	trial_n=0
	for instr_n in np.arange(n_instructions):
		for delay in ISI_inds:
			for stimCombi_n in np.arange(n_stimCombi):
				trials_info[trial_n, :] = [instr_n, delay, stimCombi_n]
				trial_n += 1

	# n_trials = trials_info.shape[0]
	shuff_index = np.arange(n_trials); np.random.shuffle(shuff_index)
	trials_info = trials_info[shuff_index, :]

	new_arr = trials_info
	verif_arr = new_arr[:-1,0] == new_arr[1:, 0]

	print('\tVerifying instructions order..')
	while sum(verif_arr):
		bad_inst_ind = np.where(verif_arr)[0][0]+1
		non_bad_inst_inds = new_arr[:, 0] != new_arr[bad_inst_ind, 0]
		ind_to_place_bad_inst = np.where(non_bad_inst_inds[1:].astype(np.int) + non_bad_inst_inds[:-1].astype(np.int) > 1)[0][0] + 1
		bad_inst_info = new_arr[bad_inst_ind,:]
		mask = np.ones(n_trials).astype(np.bool); mask[bad_inst_ind] = False; new_arr = new_arr[mask,:]
		new_arr = np.concatenate([new_arr[:ind_to_place_bad_inst, :], bad_inst_info[np.newaxis, :], new_arr[(ind_to_place_bad_inst):, :]], axis = 0)
		verif_arr = new_arr[:-1, 0] == new_arr[1:, 0]

	print('\tAll done! (sum(verif_arr=%i)' % sum(new_arr[:-1, 0] == new_arr[1:, 0]))

	trials_info = new_arr

	return trials_info



###########################################################
# Computes whether the observer gave the correct response #
###########################################################

def compute_resp_correctness_discr2(key, trial_info, key_asso):
	correct = False
	# case in which both stims are Vertical OR Horizontal
	if (trial_info[2] == 0 and key in key_asso['cwtilt_keys']) or (trial_info[2] == 3 and key in key_asso['ccwtilt_keys']):
		if trial_info[0] in [0, 1] and key in key_asso['left_keys']: correct = True;
		if trial_info[0] in [2, 3] and key in key_asso['right_keys']: correct = True;

	# case in which the stim are Left-Vertical, Right-Horizontal
	if trial_info[2] == 1:
		if trial_info[0] == 0 and key in key_asso['left_keys'] and key in key_asso['ccwtilt_keys']: correct = True;
		if trial_info[0] == 1 and key in key_asso['left_keys'] and key in key_asso['cwtilt_keys']: correct = True;
		if trial_info[0] == 2 and key in key_asso['right_keys'] and key in key_asso['ccwtilt_keys']: correct = True;
		if trial_info[0] == 3 and key in key_asso['right_keys'] and key in key_asso['cwtilt_keys']: correct = True;
		
	# case in which the stim are Left-Horizontal, Right-Vertical
	if trial_info[2] == 2:
		if trial_info[0] == 0 and key in key_asso['left_keys'] and key in key_asso['cwtilt_keys']: correct = True;
		if trial_info[0] == 1 and key in key_asso['left_keys'] and key in key_asso['ccwtilt_keys']: correct = True;
		if trial_info[0] == 2 and key in key_asso['right_keys'] and key in key_asso['cwtilt_keys']: correct = True;
		if trial_info[0] == 3 and key in key_asso['right_keys'] and key in key_asso['ccwtilt_keys']: correct = True;

	correct_side = (trial_info[0] in [0, 1] and key in key_asso['left_keys']) or (trial_info[0] in [2, 3] and key in key_asso['right_keys'])

	return correct, correct_side
		

##################%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Do the staircase procedure using the Levitt rule and returns
# variables that might have change 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def do_staircase(tiltLvl, tiltStep, corr_resps, correct_side, trial_n, tiltChanges,
	lastTiltChangeSign, reversals, minTiltLvl, maxTiltLvl, minTiltStep, lastRespondedTrial):

	tilt = tiltLvl;

	if not sum(reversals):
		# 1st reversal do simple one-up-one-down
		if corr_resps[trial_n]: tiltSign = -1
		else: tiltSign = 1

		tilt += tiltSign * tiltStep;
		if (tilt > minTiltLvl) & (tilt < maxTiltLvl):
			tiltLvl = tilt
			tiltChanges[trial_n] = tiltSign
			if tiltChanges[trial_n] == -lastTiltChangeSign:
				reversals[trial_n] = 1
				tiltStep = tiltStep / 2.

			lastTiltChangeSign = tiltSign


	elif correct_side:
		tiltSign = 0
		# not 1st reversal: one-up-two-down
		if not corr_resps[trial_n]:
			tiltSign = 1 # one-up -> increase tiltLvl
		elif corr_resps[trial_n] and corr_resps[lastRespondedTrial]:
			tiltSign = -1 # two-down -> decrease tiltLvl
		
		tilt += tiltSign * tiltStep

		# stores the new tilt if it doesn't exceed the tilt limits
		if (tilt > minTiltLvl) & (tilt < maxTiltLvl):
			print('\ntiltLvl changed %i\n' % tiltSign)
			tiltLvl = tilt
			tiltChanges[trial_n] = tiltSign

			# if there is a reversal in this feature's tiltChange
			if tiltSign == -lastTiltChangeSign:
				reversals[trial_n] = 1;
				# takes care of the Levitt rule
				if np.mod(len(np.argwhere(reversals != 0)), 2):
					# if the reversal number is odd divide tiltStep by 2
					newTiltStep = tiltStep / 2.;
					
					# if no minimum for tilt step
					#tiltStep = newTiltStep;
					
					# if minimum tilt step
					if newTiltStep >= minTiltStep:
						tiltStep = newTiltStep
					

			if tiltSign: lastTiltChangeSign = tiltSign

	return tiltLvl, tiltStep, tiltChanges, lastTiltChangeSign, reversals
	














