from psychopy import visual, event, core
import os, time, glob
import numpy as np
from psychopy.visual import GratingStim
import exp_utils_mystinfo as exputils
import serial

import random
import constants
import pygaze
from pygaze import libscreen
from pygaze import libtime
from pygaze import liblog
from pygaze import libinput
from pygaze import eyetracker
from psychopy import sound

# libtime.expstart()


####################################################
# Observer's info
####################################################
obs = 0
block = 1

last_pause_trial_n = 0

train_settings = False
staircase = False

timeAndDate = time.strftime('%d_%m_%y_%H_%M_%S',time.localtime())
task_out, outfolder = exputils.init_log(obs,block,timeAndDate)
####################################################


####################################################
# Trigger info
####################################################
send_eegtriggers = False
use_eyetrack = False

# Initialize serial port
if send_eegtriggers or use_eyetrack:
	s = serial.Serial('COM1')
	s.write('D000');

if send_eegtriggers:
	s.write('A000'); time.sleep(.04)
	s.write('A255'); time.sleep(.04)
	s.write('A000')

if use_eyetrack:
	s.write('B000'); time.sleep(.01)
	s.write('B255'); time.sleep(.01)
	s.write('B000')

	fix_thresh = 2000


####################################################


# Create the Psychopy window
# myWin = visual.Window(fullscr = False, screen = 0, monitor = 'testMonitor', allowGUI = True) #, mouseVisible=False)
# myWin.setMouseVisible(False)

# screen_args = dict(screen = 0, monitor = 'testMonitor', allowGUI = True, dispsize = [1400, 750],
# 					bgc = [128, 128, 128], units = 'deg', fullscr = False)
screen_args = dict(screen = 0, monitor = 'testMonitor', allowGUI = True, dispsize = [960, 540],
					bgc = [128, 128, 128], units = 'deg', fullscr = False)
disp = libscreen.Display(**screen_args)
myWin = pygaze.expdisplay

if use_eyetrack:
	tracker = eyetracker.EyeTracker(disp, trackertype='smi')

event.Mouse(visible=False)

####################################################
# Create Psychopy visual objects
####################################################
fixcross_size = .15
fixh = visual.Line(myWin, units = 'deg', start = (-fixcross_size, 0), end = (fixcross_size, 0), lineColor = (-1, -1, -1), lineWidth = 2)
fixv = visual.Line(myWin, units = 'deg', start = (0, -fixcross_size), end = (0, fixcross_size), lineColor = (-1, -1, -1), lineWidth = 2)

# load instructions
lett_size = .75; lett_pos = .75
stim_lett_textures = np.array([	visual.TextStim(myWin, units='deg', height = lett_size,
									pos=(0, lett_pos), text='L', alignHoriz = 'center', color='black'),
								visual.TextStim(myWin, units='deg', height = lett_size,
									pos=(0, lett_pos), text='R', alignHoriz = 'center', color='black')])

hand_lett_textures = np.array([	visual.TextStim(myWin, units='deg', height = lett_size,
									pos=(0, -lett_pos), text='L', alignHoriz = 'center', color='black'),
								visual.TextStim(myWin, units='deg', height = lett_size,
									pos=(0, -lett_pos), text='R', alignHoriz = 'center', color='black')])

# grating parameters and create grating objects
stimSize = 5; contrast = .15; oris = [0, 0]; dist_to_center = 7.5; spatFreq = 1;
myStim = []

myStim.append(GratingStim(myWin, tex='sin', mask='raisedCos', sf=spatFreq, size=[stimSize,stimSize],\
	units='deg', pos=(-dist_to_center, 0.0), interpolate=False, ori=oris[0], contrast=contrast))
myStim.append(GratingStim(myWin, tex='sin', mask='raisedCos', sf=spatFreq, size=[stimSize,stimSize],\
	units='deg', pos=(dist_to_center, 0.0), interpolate=False, ori=oris[1], contrast=contrast))

tiltLvl = 3
####################################################


####################################################
# Timing
####################################################
ref_rate = 1/60.

preinstdur = 1
instdur = .2
ISIdur =  np.arange(1.7, 2.21, .05)
stimdur = .05
ITIdur =  np.arange(1.75, 2.1, .25)
max_respTime = .7

if train_settings:
	instdur = .5
	ISIdur =  np.arange(2, 2.51, .05)
	stimdur = .2
	max_respTime = 1.5

timer = core.CountdownTimer()
####################################################

# Create trial structure (instructions, ISIs and stim combination (Hori-Vert, Hori-Hori, etc.))
trials_info = exputils.make_mystinfo_design(n_instructions = 4, n_ISIs = ISIdur.shape[0], n_stimCombi = 4, block = block, hash_ISIs_by_block = True)
trials_info = trials_info.astype(np.uint8)
n_trials = trials_info.shape[0]
# Create key bindings
key_asso = {'ccwtilt_keys':['s', 'k'], 'cwtilt_keys':['d', 'l'], 'left_keys':['s', 'd'], 'right_keys':['k', 'l']}
# key_asso = {'left_key':'d', 'right_key':'k'}
all_possible_keys = np.hstack([key_asso['left_keys'], key_asso['right_keys'], 'escape', 'esc', 'space'])


####################################################
# Show instructions for the experiment
####################################################
# clear the back buffer for drawing
myWin.clearBuffer()

slide_n = 0
while slide_n < 7:
	#if (slide_n + 1) in [1, 2, 5]: suffix = target_ori
	suffix = ''
	instr_protocol = visual.ImageStim(myWin, './material/protocol_instruction_slides_d/slide_%i' % (slide_n+1) + suffix + '.png')
	instr_protocol.draw(); myWin.flip()
	k = event.waitKeys(keyList = ['right', 'left', 'space'])

	if k[0] == 'left': slide_n -= 1
	elif k[0] == 'right' or k[0] == 'space': slide_n += 1

	if slide_n < 0: slide_n = 0

myWin.flip()
time.sleep(3)
####################################################



# create variables
timings = np.zeros(shape = (200, 6))
resp_times = np.zeros(shape = 200)
corr_resps = np.zeros(shape = 200, dtype = np.uint8)
ISI_trial_n = np.zeros(shape = 200)


if staircase:
	# reversals holds trial numbers when a reversal in the staircase occured
	reversals = np.zeros(200);
	tiltChanges = np.zeros(200);
	lastTiltChangeSign = 0;
	lastRespondedTrial= 0;
	minTiltLvl = .5; maxTiltLvl = 15; tiltStep = 3;	minTiltStep = .1;
	tiltHistory = []

	# n_trials = 80

if send_eegtriggers: s.write('A001'); time.sleep(.004); s.write('A000')
if use_eyetrack: s.write('B001'); time.sleep(.01); s.write('B000')


txt = ''; trial_n = 0
n_trial_pause = 20


# calibrate eye tracker
if use_eyetrack:
	tracker.calibrate()

	# start eye tracking
	tracker.start_recording()

# event.Mouse(visible=False)


while txt != 'Bye' and trial_n < n_trials:
	break_trial = False; responded = False
	while not break_trial:

		####### trial baseline (pre-instruction) #######
		fixh.lineColor = [-1, -1, -1]; fixv.lineColor = [-1, -1, -1]
		fixh.draw(); fixv.draw()
		timings[trial_n, 0] = myWin.flip()
		timer.reset(preinstdur - (ref_rate/2.))
		if send_eegtriggers: s.write('A010'); time.sleep(.004); s.write('A000')
		if use_eyetrack: s.write('B005'); time.sleep(.01); s.write('B000')
		myWin.clearBuffer()

		####### Instruction #######
		# choose which letters to display for hand to use and target stimulus
		# order of instructions (hand-stim): 0 = L-L, 1 = L-R, 2 = R-L, 3 = R-R

		# hand letter is L if instruction index is 0 or 1, R if instruction index is 2 or 3
		hand_tex = hand_lett_textures[int(trials_info[trial_n, 0] > 1)]
		# stim letter is L if instruction index is even (0 or 2), R if odd (1 or 3)
		stim_tex = stim_lett_textures[int(trials_info[trial_n, 0] % 2)]

		# draw instructions
		hand_tex.draw(); stim_tex.draw()
		fixh.draw(); fixv.draw()

		while timer.getTime() > 0: continue

		timings[trial_n,1] = myWin.flip()

		timer.reset(instdur - (ref_rate/2.))
		if send_eegtriggers: s.write('A0%i' % (20 + trials_info[trial_n, 0])); time.sleep(.004); s.write('A000')
		if use_eyetrack: s.write('B010'); time.sleep(.01); s.write('B000')
		myWin.clearBuffer()



		####### ISI #######
		fixh.draw(); fixv.draw()
		ISI_trial_n[trial_n] = ISIdur[trials_info[trial_n, 1]]
		while timer.getTime() > 0:
			if use_eyetrack:
				continue
				# eye_pos = np.array(tracker.sample())
				# alleyepos.append(eye_pos)
				# alldist.append(np.sqrt(eye_pos[0]**2 + eye_pos[1]**2))
				# #print('eye dist from center = %.2f\n' % np.sqrt(eye_pos[0]**2 + eye_pos[1]**2))
				# if np.sqrt(eye_pos[0]**2 + eye_pos[1]**2) > fix_thresh:
					# wtxt = 'You have broken the fixation.\nPlease fixate the central cross during the whole trial\n\nPress SPACE to continue..'
					# waitText = visual.TextStim(myWin, units = 'pix', height = 20,
						# pos = (0, 0), text = wtxt, alignHoriz = 'center', color = 'black')
					# waitText.draw(); myWin.flip()
					# kk = event.waitKeys(keyList = ['space'])

					# break

			else: continue

		timings[trial_n, 2] = myWin.flip()
		timer.reset(ISI_trial_n[trial_n] - (ref_rate/2.))
		if send_eegtriggers: s.write('A0%i'%(30 + trials_info[trial_n, 1])); time.sleep(.004); s.write('A000')
		if use_eyetrack: s.write('B015'); time.sleep(.01); s.write('B000')
		myWin.clearBuffer()



		####### Grating #######
		# 0 is vertical, 90 is horizontal
		if trials_info[trial_n, 2] == 0:
			myStim[0].ori, myStim[1].ori = [0 + tiltLvl, 0 + tiltLvl]
		elif trials_info[trial_n, 2] == 1:
			myStim[0].ori, myStim[1].ori = [0 - tiltLvl, 0 + tiltLvl]
		elif trials_info[trial_n, 2] == 2:
			myStim[0].ori, myStim[1].ori = [0 + tiltLvl, 0 - tiltLvl]
		elif trials_info[trial_n, 2] == 3:
			myStim[0].ori, myStim[1].ori = [0 - tiltLvl, 0 - tiltLvl]

		myStim[0].draw(); myStim[1].draw()
		fixh.draw(); fixv.draw()

		while timer.getTime() > 0: continue

		timings[trial_n, 3] = myWin.flip()
		temp_resp_t = time.time()
		timer.reset(stimdur - (ref_rate/2.));
		if send_eegtriggers: s.write('A0%i' % (50 + trials_info[trial_n, 2])); time.sleep(.004); s.write('A000')
		if use_eyetrack: s.write('B020'); time.sleep(.01); s.write('B000')
		myWin.clearBuffer()

		fixh.lineColor = [-1, -1, 1]; fixv.lineColor = [-1, -1, 1];
		fixh.draw(); fixv.draw()
		while timer.getTime() > 0: continue
		timings[trial_n, 4] = myWin.flip()



		####### Response processing #######
		k = event.waitKeys(maxWait = max_respTime - (time.time()-temp_resp_t), keyList = all_possible_keys, timeStamped = timings[trial_n, 3])
		resp_times2 = time.time() - temp_resp_t

		if k is None:
			# k = [['none', -1.]]
			resp_key, resp_times[trial_n] = 'none', -1.
			correct = 0

			wtxt = 'Too late! Do you need a break? press SPACE to continue..'
			waitText = visual.TextStim(myWin, units = 'pix', height = 20,
				pos = (0, 0), text = wtxt, alignHoriz = 'center', color = 'black')
			waitText.draw(); myWin.flip()
			kk = event.waitKeys(keyList = ['space'])

			#add the trial to the end of the list
			trials_info = np.concatenate([trials_info, trials_info[trial_n, :][np.newaxis]], axis = 0)

			n_trials += 1

		elif k[0][0] in ['escape', 'esc']: txt = 'Bye'; break;
		
		else:
			# if observer responded, check correctness and change fixation cross color
			resp_key = k[0][0]
			resp_times[trial_n] = k[0][1]
			[correct, correct_side] = exputils.compute_resp_correctness_discr2(key = resp_key,
				trial_info = trials_info[trial_n, :], key_asso = key_asso)
			
			if send_eegtriggers: s.write('A0%i' % (60 + correct)); time.sleep(.004); s.write('A000')
			if use_eyetrack: s.write('B025'); time.sleep(.01); s.write('B000')
			
			if correct: fixh.lineColor = [-1, 1, -1]; fixv.lineColor = [-1, 1, -1];
			else: fixh.lineColor = [1, -1, -1]; fixv.lineColor = [1, -1, -1];
			fixh.draw(); fixv.draw()

			myWin.flip()
			timer.reset(.5 - (ref_rate/2.))
			while timer.getTime() > 0: continue
			responded = True
			
		corr_resps[trial_n] = correct
		# write in log file
		task_out.write('%s\t%i\t%i\t%i\t%i\t%i\t%s\t%.5f\t%.5f\t%i\n' % (obs, block, trial_n,
			trials_info[trial_n, 0], trials_info[trial_n, 1], trials_info[trial_n, 2], resp_key, resp_times[trial_n], resp_times2, correct))

		# Pause every n_trial_pause trials
		if trial_n >= (last_pause_trial_n + n_trial_pause):
			last_pause_trial_n = trial_n
			rt_last_chunk = resp_times[(trial_n - n_trial_pause):trial_n]
			rt_last_hits = rt_last_chunk[rt_last_chunk > 0]
			wtxt = 'A little break!\nIn the last %i trials your reaction time was: %.3f ms\n\npress SPACE to continue..' % (n_trial_pause, np.median(rt_last_hits))
			waitText = visual.TextStim(myWin, units = 'pix', height = 20,
				pos = (0, 0), text = wtxt, alignHoriz = 'center', color = 'black')
			waitText.draw(); myWin.flip()
			kk = event.waitKeys(keyList = ['space'])

		break_trial = True

		# Staircase
		if staircase:
			if (trial_n > 1) and responded:
				[tiltLvl, tiltStep, tiltChanges, lastTiltChangeSign, reversals] = exputils.do_staircase(tiltLvl, tiltStep, corr_resps,\
					correct_side, trial_n, tiltChanges, lastTiltChangeSign, reversals, minTiltLvl, maxTiltLvl, minTiltStep, lastRespondedTrial)
				lastRespondedTrial = trial_n
				tiltHistory.append(tiltLvl)

	####### ITI #######
	fixh.lineColor = [1, 1, 1]; fixv.lineColor = [1, 1, 1];
	fixh.draw(); fixv.draw()
	timings[trial_n, 5] = myWin.flip()
	timer.reset(np.random.choice(ITIdur) - (ref_rate/2.))
	myWin.clearBuffer()
	while timer.getTime() > 0: continue
		# fixh.draw(); fixv.draw()
		# endText = visual.TextStim(myWin, units='pix', height = 10,
	        # pos=(-300, -300), text=str(tracker.sample()), alignHoriz = 'center', color='black')
		# endText.draw()
		# myWin.flip()

	trial_n += 1

# send end of block trigger
if send_eegtriggers: s.write('A100'); time.sleep(.004); s.write('A000')
if use_eyetrack:
	s.write('B030'); time.sleep(.004); s.write('B000')
	# stop eye tracking
	tracker.stop_recording()

# close the log file
task_out.close()

# save parameters
outfolder = './results/obs_%s/' % obs
output_param_file = '%sparameters_obs%s_block%i_date%s.npz' % (outfolder, obs, block, timeAndDate)
np.savez(output_param_file, {'trials_info':trials_info, 'timings':timings, 'resp_times':resp_times, 'stimSize':stimSize, 'contrast':contrast,\
		'oris':oris, 'dist_to_center':dist_to_center, 'spatFreq':spatFreq, 'ref_rate':ref_rate, 'preinstdur':preinstdur,\
		'instdur':instdur, 'ISIdur':ISIdur, 'stimdur':stimdur, 'ITIdur':ITIdur, 'max_respTime':max_respTime} )


if staircase:
	output_stairc_file = '%sstaircase_parameters_obs%s.npz' % (outfolder, obs)
	np.savez(output_stairc_file, {'tiltLvl':tiltLvl, 'tiltStep':tiltStep, 'tiltChanges':tiltChanges,
		'reversals':reversals, 'tiltHistory':np.array(tiltHistory)} )

# np.save('alleyepos.npy',alleyepos)
# np.save('alldist.npy',alldist)

exitsound = sound.Sound(value='A', secs=.2)
exitsound.play(loops=1)

# if k[0][0] == 'escape': txt = 'ESCAPED ! bye !'
# else: 
txt = 'FINISHED ! Thanks !'
endText = visual.TextStim(myWin, units='pix', height = 30,
            pos=(0, 0), text=txt, alignHoriz = 'center', color='black')
endText.draw(); myWin.flip()
time.sleep(2)

print('\n\n\tAccuracy on block %i was: %.1f\n\n' % (block, corr_resps[:n_trials].mean()))

if use_eyetrack:
	tracker.close()
# disp.close()
myWin.close()
core.quit()


