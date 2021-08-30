import numpy as np
from scipy import signal as sig
from scipy import ndimage

##################
#    Functions   #
##################


def resp_correct_func(design_trialN = None, resp_trial_n=None):
	if design_trialN[0]==0:
		if (design_trialN[1]==0 or design_trialN[1]==2 ) and resp_trial_n==0:
			acc_trialN=1
		elif (design_trialN[1]==1 or design_trialN[1]==3 ) and resp_trial_n==1:
			acc_trialN=1
		else:
			acc_trialN=0

	if design_trialN[0]==1:
		if (design_trialN[1]==0 or design_trialN[1]==3 ) and resp_trial_n==2:
			acc_trialN=1
		elif (design_trialN[1]==1 or design_trialN[1]==2 ) and resp_trial_n==3:
			acc_trialN=1
		else:
			acc_trialN=0

	if design_trialN[0]==2:
		if (design_trialN[1]==0 or design_trialN[1]==2 ) and resp_trial_n==2:
			acc_trialN=1
		elif (design_trialN[1]==1 or design_trialN[1]==3 ) and resp_trial_n==3:
			acc_trialN=1
		else:
			acc_trialN=0

	if design_trialN[0]==3:
		if (design_trialN[1]==0 or design_trialN[1]==3 ) and resp_trial_n==0:
			acc_trialN=1
		elif (design_trialN[1]==1 or design_trialN[1]==2 ) and resp_trial_n==1:
			acc_trialN=1
		else:
			acc_trialN=0

	return acc_trialN


ins_all = np.array([[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 0, 1], [0, 1, 1, 0]])
n_tsteps = 100
sigma_compet = .1

def make_lfc_compet(ins = None, n_tsteps = 100, sigma = sigma_compet, alpha = .1,
	n_instr = 4, m_w = .5, sw1 = 0, sw2 = .5, inh_lvl = .1, min_val = 0, seed_n=None):

	#						 L-B  L-T  R-B  R-T
	inp_weights = np.array([[m_w, m_w, 0.0, 0.0],	# RR
						  	[0.0, 0.0, m_w, m_w],	# LL
						  	[m_w, sw1, sw1, m_w],	# RL
						  	[sw1, m_w, m_w, sw1]]) 	# LR
	#						 L-B  L-T  R-B  R-T
	ins_weights = np.array([[1.0, sw2, 0.0, 0.0],	# L-B
						  	[sw2, 1.0, 0.0, 0.0],	# L-T
						  	[0.0, 0.0, 1.0, sw2],	# R-B
						  	[0.0, 0.0, sw2, 1.0]]) 	# R-T
	# inh_lvl = .1
	inh_weights = -inh_lvl * np.ones(shape = [4, 4])
	inh_weights[np.identity(4, np.bool)] = 0


	np.random.seed(seed_n)
	rand_noise = np.random.randn(n_instr, n_tsteps) * sigma

	if ins is None: ins = np.array([1, 0, 0, 0])
	if inp_weights is None: inp_weights = np.random.rand(4, 4)
	if inh_weights is None: inh_weights = np.random.rand(4, 4)
	if ins_weights is None: ins_weights = np.random.rand(4, 4)

	lfc_acts = np.zeros(shape=[n_instr, n_tsteps], dtype=np.float32)
	lfc_acts[:, 0] = np.random.rand(4)/50.

	# out of input units (after potential collateral effects, e.g. L-Top activates L-Bottom)
	ins_out = np.dot(ins_weights, ins)
	# what gets in output units (instruction units)
	inOutputUnits = np.dot(inp_weights, ins_out)

	for t in np.arange(1, n_tsteps):
		lfc_acts[:, t] = lfc_acts[:, t-1]\
			+ alpha * (inOutputUnits + np.dot(inh_weights, lfc_acts[:, t-1]))\
			+ rand_noise[:, t]
		lfc_acts[lfc_acts[:, t] < min_val, t] = min_val

	return lfc_acts


def phase_updating(Neurons=[], Radius=1, Damp=0.3, Coupling=0.3, multiple=True):
	"""
	Returns updated value of the inhibitory (I) and excitatory (E) phase neurons
		@Param Neurons, list containing the current activation of the phase code units
		@Param Radius,  bound of maximum amplitude
		@Param Damp, strength of the attraction towards the Radius, e.g. OLM cells reduce  pyramidal cell activation
		@Param Coupling, frequency*(2*pi)/sampling rate
		@Param multiple, True or false statement whether the Neurons array includes more than one node or not
		
		Formula (2) and (3) from Verguts (2017) 
	"""
	# updating the phase neurons from the processing module
	if multiple:
		Phase = np.zeros ((len(Neurons[:,0]),2)) # Creating variable to hold the updated phase values ( A zero for each E and I phase neuron of each phase code unit)
		r2 = np.sum(Neurons * Neurons, axis = 1) # calculating the amplitude depending on the activation of the E and I phase neurons
		# updating the E phase neurons, the higher the value of the I neurons (Neurons[:, 1]), the lower the value of the E neurons
		Phase[:,0] = Neurons[:,0] -Coupling * Neurons[:,1] - Damp *((r2>Radius).astype(int)) * Neurons[:,0] 
		# updating the I phase neurons, the higher the value of the E neurons (Neuron[:, 0]), the higher the value of the I neurons
		Phase[:,1] = Neurons[:,1] +Coupling * Neurons[:,0] - Damp * ((r2>Radius).astype(int)) * Neurons[:,1]
	# updating the phase neurons of the MFC
	else:
		Phase = np.zeros((2)) 
		r2 = np.sum(Neurons[0] * Neurons[1], axis = 0) 
		Phase[0] = Neurons[0] -Coupling*Neurons[1] - Damp * ((r2>Radius).astype(int)) * Neurons[0]
		Phase[1] = Neurons[1] +Coupling*Neurons[0] - Damp * ((r2>Radius).astype(int)) * Neurons[1]

	return Phase


# Model function
def Model_sim(Threshold=4, drift=0, Cgs_var_sd = .5, theta_freq = 5,
	damp_thetaFreq_coef = .005, sim_path = './', MFC_compet_thresh = None,
	kick_value = None, sw2 = None, nReps = 10, sigma_compet = None,
	inh_compet = .1, alpha_compet = .1, tiltrate = .1, n_trials = None,
	save_eeg = True, save_behav = True, return_eeg = False, return_behav = False,
	print_prog = False, gamma_freq = 30, theta_amplitude = 1):
	"""
	Model simulation that writes away two files
		@Param Threshold, the response threshold
		@Param drift, drift (in Hz) of Neurons in Sensory and Action modules
		
		@File csv-file, should ressemble the behavioral data file you get after testing a participant
		@File npy-file, contains simulated EEG data
	"""


	n_tsteps = np.int(-5*theta_freq + 85)
	if n_tsteps<15: n_tsteps = 15
	LFC_compet_act_scale = 1
	bursting = False
	t_start_compet = 0

	# timing of the experiment
	srate = 500                                               # sampling rate per second
	Preinstr_time = int(.2 * srate)                           # pre-instruction time (1s)
	Instr_time = int(.2 * srate)                              #  instruction presentation (200 ms)
	Prep_time = (np.arange(1.7,2.2,.05) * srate).astype(int)  # ISD ranging from 1700 to 2200 ms, multiplying it by the sampling rate to get how many samples we have for each ISD
	Stim_time = int(.05 * srate)                              # Stimulus presentation of 50 ms
	Resp_time = .7 * srate                                    # max response time of 1s
	FB_time = int(.1 * srate)                                 # Feedback presentation of 500 ms
	Response_deadline = .7 * srate + 1                        # Response deadline

	# max trial time
	TotT = (Preinstr_time + Instr_time + max(Prep_time) + Stim_time + Resp_time).astype(int)+n_tsteps

	# variables for randomization
	nInstr = 4                                        # number of instructions
	nTilts = 2                                        # number of tilt directions
	nSides = 2                                        # number of stimuli locations
	nStim = nTilts * nSides                           # number of stimuli in total
	nResp = 4                                         # number of responses
	UniqueTrials = nInstr * nStim * len(Prep_time)    # number of different unique trials
	Tr = UniqueTrials * nReps                         # Total amount of trials

	###########################
	#    Processing Module   #
	##########################
	nNodes = nStim + nResp                          # total model nodes = stimulus nodes + response nodes
	r2max = 1                                       # max amplitude
	damp = 0.3                                      # damping parameter, e.g. OLM cells that damp the gamma amplitude
	decay = 1.                                      # decay parameter
	noise = 0.05                                    # noise parameter

	Phase = np.zeros((nNodes,2,TotT,Tr))            # phase neurons, each node has two phase neurons, we update it each timestep, based on the sample rate of each trial
	Rate = np.zeros((nNodes, TotT, Tr))             # rate neurons, each node has one rate neuron

	# Weights initialization
	W = np.ones((nStim,nResp))*0.5
	W[(0,2),1] = 0.1
	W[(0,2),3] = 0.1
	W[(1,3),0] = 0.1
	W[(1,3),2] = 0.1

	#########################
	#    Integrator Module  #
	#########################
	Integr = np.zeros(shape = [nResp, TotT, Tr]);              # inhibitory weights inducing competition
	inh = np.ones((nResp,nResp))*-0.01
	for i in range(nResp):
		inh[i,i] = 0

	cumul = 1
	# setting collapsing bound function from [Palestro, Weichart, Sederberg, & Turner (2018)]
	t_thresh = np.linspace(0, 1, 500)
	a_thresh = Threshold; k_thresh = 2; a_prime_thresh = .0; lamb_thresh = .35 # see eq. 1 of Palestro et al., 2018
	Threshold_byTime = a_thresh - (1 - np.exp(-(t_thresh/lamb_thresh)**k_thresh)) * ((a_thresh/2.) - a_prime_thresh)

	#######################
	#    Control Module   #
	######################
	r2_MFC = theta_amplitude                        #radius MFC
	Ct = (theta_freq / srate) * 2 * np.pi           #coupling theta waves
	damp_MFC = damp_thetaFreq_coef*theta_freq       #damping parameter MFC (was at .003 but it yielded extremely high theta amplitude for high theta frequencies (e.g. 8hz))
	acc_slope = 10                                  # MFC slope parameter, is set to -5 in equation (7) of Verguts (2017)
													#(steepness of burst threshold)

	MFC = np.zeros((2,TotT,Tr))                     # MFC phase units, two phase neurons
	Be=0                                            # bernoulli process initialized at 0 (rate code MFC)

	LFC = np.zeros((nInstr,Tr))                     # LFC stores information for each instruction for each trial
	LFC_sync = np.zeros((nInstr,4))
	LFC_sync[0,:]=[0,1,4,5]                         # LL  sync left stimulus nodes with left hand nodes
	LFC_sync[1,:]=[2,3,6,7]                         # RR  sync right stimulus nodes with right hand nodes
	LFC_sync[2,:]=[0,1,6,7]                         # LR  sync left stimulus nodes with right hand nodes
	LFC_sync[3,:]=[2,3,4,5]                         # RL  sync right stimulus nodes with left hand nodes

	# competition
	active_LFC_units_activity_by_time = np.zeros(shape = [Tr, TotT])
	burst_by_trial = np.zeros(shape = [Tr, TotT, 2])
	
	LFC_compet_by_time = np.zeros(shape=[Tr, 4, TotT], dtype=np.float32)
	which_unit_winning = np.zeros(shape=[Tr, TotT], dtype=np.int8) - 5


	############################
	#    Stimuli activations   #
	############################

	Stim_activation=np.zeros((nStim,nResp))                   # Stimulus activation matrix
	Stim_activation[0,:]=np.array([1,0,1,0])*tiltrate         # Activate 2 stimuli with left tilt (LL)
	Stim_activation[1,:]=np.array([0,1,0,1])*tiltrate         # Activate 2 stimuli with right tilt(RR)
	Stim_activation[2,:]=np.array([1,0,0,1])*tiltrate         # Activate left stimulus with left tilt and right with right tilt
	Stim_activation[3,:]=np.array([0,1,1,0])*tiltrate         # Activate left stimulus with right tilt and right stimulus with left tilt
			
	# Randomization for instructions, tilt of stimuli and ISD's
	# let's say: 1 = LL       left stim, left resp   | two times left tilt
	#            2 = RR       right stim, right resp | two time right tilt
	#            3 = LR       left stim, right resp  | left tilt (left) and right tilt (right)
	#            4 = RL       right stim, left resp  | right tilt (left) and left tilt (right)
	
	
	##################################
	#    Create a factorial design  #
	#################################
	Instr = np.repeat(range(nInstr), nStim * len(Prep_time)) # Repeat the instructions (nInstr: 0-4) for the ISD's of each stimulus and put it into an array
	Stim = np.tile(range(nStim), nInstr * len(Prep_time)) 	 # Repeat the stimuli for each instruction, total amount of stimuli

	Preparation = np.floor(np.array(range(UniqueTrials))/(nStim))%len(Prep_time) # Preparation Period, 11 levels 
	Design = np.column_stack([Instr, Stim, Preparation]) # Create an array that has a stack of lists, each list contains instruction, stimulus and a preparation period
	Design = np.column_stack([np.tile(Design,(nReps,1)), np.repeat(np.arange(nReps), UniqueTrials)]) # Repeat the design nReps
	np.random.shuffle(Design) # shuffle the design making it have a random order

	Design = Design.astype(int)
	
	#####################################################
	#    Oscillations start point of the phase neurons  #
	#####################################################
	start = np.random.random((nNodes,2))          # Draw random starting points for the two phase neurons of each node
	start_MFC = np.random.random((2))             # Acc phase neurons starting point
	# assign starting values
	Phase[:,:,0,0] = start
	MFC[:,0,0] = start_MFC

	#################################
	#            Records           #
	################################

	Hit = np.zeros((TotT,Tr))                     # Hit record, check for the sampling points of each trial
	RT = np.zeros((Tr))                           # RT record, 
	accuracy = np.zeros((Tr))                     # Accuracy record
	Instruct_lock = np.zeros((Tr))                # Instruction onset record
	Stim_lock = np.zeros((Tr))                    # Stimulus onset record
	Response_lock = np.zeros((Tr))                # Response onset record 
	resp = np.ones((Tr)) * -1                     # Response record
	preparatory_period = np.zeros((Tr))  
	sync = np.zeros((nStim, nResp, Tr))           # Sync record between the stimuli and the responses on each trial               


	############################################
	#            Preinstruction  Period        #
	###########################################

	time = 0
	if n_trials != None: Tr = n_trials
	for trial in range(Tr): # for every trial in total amount of trials

		if print_prog & np.logical_not(trial%10): print('trial %i' % trial)
		bursting = False

		# creating processing units' frequency noise
		# Coupling parameter for gamma oscillations in sensory and action nodes
		freq1 = ndimage.gaussian_filter(np.random.normal(size=TotT, loc=1, scale=Cgs_var_sd),
			sigma=200)*gamma_freq
		Cg_1 = (freq1/srate) * 2 * np.pi

		# Coupling gamma waves with frequency difference of "drift" Hz, for the action nodes
		freq2 = ndimage.gaussian_filter(np.random.normal(size=TotT, loc=1, scale=Cgs_var_sd),
			sigma=200)*(gamma_freq+drift)
		Cg_2 = (freq2/srate) * 2 * np.pi



		# FIRST STEP: copying over the phase values of the previous trial to the current trial
		if trial > 0:                                       ### index 0 of the phase neurons are already assigned random starting values, starting points are end points of previous trials
			Phase[:,:,0,trial] = Phase[:,:,time,trial-1]    ### Taking both phase neurons of all the nodes of the current trial and setting it equal to the phase neurons of the previous triaL
			MFC[:,0,trial] = MFC[:,time,trial-1]            ### Taking both phase neurons of the MFC of the current trial and setting it equal to the phase neurons of the previous trial 
				

		# SECOND STEP: updating phase code units each sample point in the preinstruction period
		## Pre-instruction time = no stimulation and no bursts
		for time in range(Preinstr_time): # looping across the sample points of the pre-instruction time
			## Cg_1 and Cg_2 are the oscillating gamma frequencies, stimulus and response nodes have different gamma frequencies!
			## Ct the oscillating frequency of the MFC phase neurons
			## r2max is the radius (amplitude) of a pair of inhibitory and excitatory neurons
			## damp is the damping value acting on the excitatory (e.g. OLM cells)
			Phase[0:nStim, : , time + 1, trial] =\
				phase_updating(Neurons=Phase[0:nStim, :, time, trial],
				Radius=r2max, Damp=damp, Coupling=Cg_1[time],
				multiple=True)            	### updating the stimulus nodes
			Phase[nStim:nNodes, : , time + 1, trial] =\
				phase_updating(Neurons=Phase[nStim:nNodes, : , time,trial],
				Radius=r2max, Damp=damp, Coupling=Cg_2[time], multiple=True)  			### updating the response nodes
			MFC[:, time+1, trial] = phase_updating(Neurons = MFC[:, time, trial],
													Radius=r2_MFC, Damp=damp_MFC,
													Coupling=Ct, multiple=False)                                ### updating the MFC node

		# THIRD STEP: Showing the instructions --> Phase reset
		t = time                    ### setting the current sample point of the preinstruction time on the current trial
		Instruct_lock[trial] = t    ### setting the instruction onset of the current trial
		
		## phase reset of the MFC phase neurons due to the instruction
		MFC[: , t, trial] = np.ones((2)) * r2_MFC
		


		##########################################
		#            Preparation Period          #
		##########################################
		
		## Instruction presentation and preparation period
		## start syncing but no stimulation yet
		preparatory_period[trial] = Prep_time[Design[trial,2]] ### Set the preparatory period (ISD), use the Preparation variable (which is randomized and 11 levels) to select a Prep_time
				
		for time in range(t , int(t + Instr_time + int(preparatory_period[trial]))): ### looping of the sample points of the ISD + instruction time period

			# SECOND STEP: updating phase code units each sample point in the preparatory period
			Phase[0:nStim, :, time + 1, trial] = phase_updating(\
				Neurons=Phase[0:nStim, :, time, trial],
				Radius=r2max, Damp=damp, Coupling=Cg_1[time], multiple=True)         ### updating the stimulus nodes
			Phase[nStim:nNodes, :, time + 1, trial]=phase_updating(\
				Neurons=Phase[nStim:nNodes, :, time, trial],
				Radius=r2max, Damp=damp, Coupling=Cg_2[time], multiple=True) ### updating the response nodes
			MFC[:, time+1, trial] = phase_updating(Neurons=MFC[:, time, trial],
								Radius=r2_MFC, Damp=damp_MFC, Coupling=Ct, multiple=False)                             ### updating the MFC node
		
			# THIRD STEP: Rate code MFC neuron activation is calculated by a bernoulli process, start syncing
			Be = 1 / (1 + np.exp(-acc_slope * (MFC[0,time,trial]-1))) ### Equation (7) in Verguts (2017)
			prob = np.random.random()

			# competition
			if Be > MFC_compet_thresh:
				if not bursting:
					t_start_compet = time
					LFC_compet_by_time[trial, :, time:(time+n_tsteps)] =\
						make_lfc_compet(ins = ins_all[Design[trial, 0]],
							n_tsteps = 	n_tsteps, alpha=alpha_compet,
							sw2 = sw2, sigma = sigma_compet,
							inh_lvl = inh_compet)/LFC_compet_act_scale
					bursting = True
			elif bursting:
				bursting = False
				t_start_compet = time
				LFC_compet_by_time[trial, :, time:(time+n_tsteps)] =\
					np.zeros(shape=[4, n_tsteps])
			
			# FOURTH STEP:
			if prob < Be:
				Hit[time, trial] = 1

				# competition
				which_unit_winning[trial, time] = np.argmax(LFC_compet_by_time[trial, :, time]).squeeze()
				LFC[which_unit_winning[trial, time], trial] =\
					LFC_compet_by_time[trial, which_unit_winning[trial, time], time]
				kick_t = kick_value
				# winning rule node amplifies the kick from the MFC
				kick_t *= LFC[which_unit_winning[trial, time], trial]

				burst_by_trial[trial, time, 0] = kick_t 

				# take the 4 nodes associated with the current instruction & kick them
				for nodes in LFC_sync[which_unit_winning[trial, time], :]: 
					Phase[int(nodes), 0, time + 1, trial] =\
						decay * Phase[int(nodes), 0, time + 1, trial] + kick_t

		t=time
		Stim_lock[trial]=t
		
		####################################################################################
		#            Response Period: syncing bursts and rate code stimulation             #
		####################################################################################
		### while the response of this trial is still equal to -1 (no answer has been given)
		while resp[trial] == -1 and time < t + Response_deadline: 
			time += 1

			# FIRST STEP: updating phase code units of processing module
			Phase[0:nStim,:,time+1,trial]=phase_updating(\
				Neurons=Phase[0:nStim,:,time,trial],
				Radius=r2max, Damp=damp, Coupling=Cg_1[time], multiple=True)
			Phase[nStim:nNodes,:,time+1,trial]=phase_updating(\
				Neurons=Phase[nStim:nNodes,:,time,trial],
				Radius=r2max, Damp=damp, Coupling=Cg_2[time], multiple=True)
			MFC[:,time+1,trial]=phase_updating(Neurons=MFC[:,time,trial],
				Radius=r2_MFC, Damp=damp_MFC, Coupling=Ct, multiple=False)
			
			# SECOND STEP: bernoulli process in MFC rate
			Be = 1/(1+np.exp(-acc_slope*(MFC[0,time,trial]-1)))
			prob = np.random.random()

			# competition
			if Be > MFC_compet_thresh:
				if not bursting:
					t_start_compet = time
					# just to make sure it does not try to compute
					# more time points for the competition than
					# there are time points left in the trial
					n_times_compet = np.min([ (TotT - time), n_tsteps])
					LFC_compet_by_time[trial, :, time:(time+n_tsteps)] =\
						make_lfc_compet(ins = ins_all[Design[trial, 0]],
							n_tsteps = 	n_tsteps, alpha=alpha_compet,
							sw2 = sw2, sigma = sigma_compet,
							inh_lvl = inh_compet)/LFC_compet_act_scale
					bursting = True
			elif bursting:
				bursting = False
				t_start_compet = time
				LFC_compet_by_time[trial, :, time:(time+n_tsteps)] =\
						np.zeros(shape=[4, n_tsteps])

			# THIRD STEP: Burst
			if prob < Be:
				Hit[time, trial] = 1

				which_unit_winning[trial, time] = np.argmax(LFC_compet_by_time[trial, :, time]).squeeze()
				LFC[which_unit_winning[trial, time], trial] =\
					LFC_compet_by_time[trial, which_unit_winning[trial, time], time]
				kick_t = kick_value
				# winning rule node amplifies the kick from the MFC
				kick_t *= LFC[which_unit_winning[trial, time], trial]
				burst_by_trial[trial, time, 0] = kick_t

				# take the 4 nodes associated with the current instruction & kick them
				for nodes in LFC_sync[which_unit_winning[trial, time], :]: 
					Phase[int(nodes), 0, time + 1, trial] =\
						decay * Phase[int(nodes), 0, time + 1, trial] + kick_t

			# FOURTH STEP: updating rate code units
			Rate[0:nStim, time, trial] = Stim_activation[Design[trial,1],:]*(1/(1+np.exp(-5*Phase[0:nStim,0,time,trial]-0.6))) ### Updating ratecode units for the stimulus nodes
			Rate[nStim:nNodes, time, trial] = np.matmul(Rate[0:nStim, time, trial],W)*(1/(1+np.exp(-5*Phase[nStim:nNodes,0,time,trial]-0.6))) ### Updating ratecode units for the response nodes
			Integr[:, time+1, trial] = np.maximum(0, Integr[:, time, trial]+cumul*Rate[nStim:nNodes, time, trial]+np.matmul(inh,Integr[:, time, trial]))+noise*np.random.random((nResp))
			
			
			for i in range(nResp):
				# collapsing bounds
				if Integr[i, time+1, trial]>Threshold_byTime[time-t]:
					resp[trial]=i
					Integr[:, time+1, trial] = np.zeros((nResp))

		LFC_compet_by_time[trial, :, time:] = 0

		RT[trial]=(time-t)*(1000/srate)
		t=time
		Response_lock[trial]=t
		accuracy[trial] = resp_correct_func(Design[trial,:], resp[trial])
		


	Trials=np.arange(Tr)
	Design=np.column_stack((Trials, Design[:n_trials], resp[:n_trials], accuracy[:n_trials], RT[:n_trials],
		Instruct_lock[:n_trials], Stim_lock[:n_trials], Response_lock[:n_trials]))
	Column_list='trial,instr,stim,isd,reps,response,accuracy,rt,instr_onset,stim_onset,resp_onset'
	filename_behavioral='Behavioral_Data_simulation_thetaFreq%.2fHz_thresh%.1f_drift%.1f_thetaAmp%.2f' % (theta_freq, Threshold, drift, theta_amplitude)
	if save_behav:
		np.savetxt(sim_path + filename_behavioral+'_1.csv', Design, header=Column_list, delimiter=',',fmt='%.2f')
	
	if save_eeg:
		Phase_ds = sig.resample(Phase, int(TotT/2.), axis = 2)
		MFC_ds = sig.resample(MFC, int(TotT/2.), axis = 1)
		Rate_ds = sig.resample(Rate, int(TotT/2.), axis = 1)
		Integr_ds = sig.resample(Integr, int(TotT/2.), axis = 1)
		burst_by_trial_ds = sig.resample(burst_by_trial, int(TotT/2.), axis = 1)
		active_LFC_units_activity_by_time_ds = sig.resample(active_LFC_units_activity_by_time, int(TotT/2.), axis = 1)

		EEG_data = {'Phase':Phase_ds[:,0,:,:], 'MFC':MFC_ds[0,:,:], 'Rate':Rate_ds, 'Integr':Integr_ds, 'burst_by_trial':burst_by_trial_ds,
						'active_LFC_units_activity_by_time':active_LFC_units_activity_by_time_ds}
		filename_EEG='EEG_Data_simulation_thetaFreq%.2fHz_thresh%.1f_drift%.1f_256Hz' % (theta_freq, Threshold, drift)
		np.savez(sim_path + filename_EEG + '.npz', EEG_data)

	if return_behav & return_eeg:
		return Design, Phase, MFC, Rate, Integr, burst_by_trial, LFC_sync, which_unit_winning, LFC_compet_by_time
	elif return_behav:
		return Design
	elif return_eeg:
		return Phase, MFC, Rate, Integr, burst_by_trial, LFC_sync, which_unit_winning, LFC_compet_by_time

