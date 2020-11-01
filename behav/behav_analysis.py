import numpy as np
from matplotlib import pyplot as pl
import pandas as pd
import glob, os
from scipy import stats
from scipy import signal as sig
import pingouin as pg 


# simple EZ-diffusion model based on Wagenmakers et al, 2007
def ezdiff(rt, correct, s = 1.0):
	if sum(correct==1)==0:
		correct = np.zeros(len(correct))
		correct[0]=1
	logit = lambda p:np.log(p/(1-p))

	assert len(rt)>0
	assert len(rt)==len(correct)
	
	assert np.max(correct)<=1
	assert np.min(correct)>=0
	
	pc=np.mean(correct)
  
	assert pc > 0
	
	# subtract or add 1/2 an error to prevent division by zero
	if pc==1.0:
		pc=1 - 1/(2*len(correct))
	if pc==0.5:
		pc=0.5 + 1/(2*len(correct))
	MRT=np.mean(rt[correct==1])
	VRT=np.var(rt[correct==1])

	assert VRT > 0
	
	r=(logit(pc)*(((pc**2) * logit(pc)) - pc*logit(pc) + pc - 0.5))/VRT
	v=np.sign(pc-0.5)*s*(r)**0.25
	a=(s**2 * logit(pc))/v
	y=(-1*v*a)/(s**2)
	MDT=(a/(2*v))*((1-np.exp(y))/(1+np.exp(y)))
	t=MRT-MDT
	
	return([a,v,t])


def pad_fft(data, n_t_pad, freqmaskpad):
	n_obs, n_instr = data.shape[0], data.shape[2]
	data_pad = np.zeros(shape=[n_obs, n_t_pad, n_instr])
	for obs_ind in np.arange(n_obs):
		for inst in np.arange(n_instr):
			avg_obs_data = data[obs_ind, :, inst].mean()
			data_pad[obs_ind, :, inst] = np.hstack([np.repeat(avg_obs_data, 5),
				data[obs_ind, :, inst], np.repeat(avg_obs_data, 5)])

	# return amplitude of padded signal
	return np.abs(np.fft.fft(data_pad, axis=1))[:, freqmaskpad, :]

# path to data
data_path = './data/'

insttxts = np.array(['LL', 'RL', 'LR', 'RR'])

fix_thresh_indeg = 1.5

# there were 11 ISDs
n_t = 11
step = 0.05
# instructions order by accuracy (higher to lower)
inst_diff_order = np.array([3, 0, 2, 1], dtype=np.int)

obs_all = np.arange(1, 40)
## participants 5, 15, 9, 16 and 33 were excluded (see Methods)
obs_all = obs_all[np.array([obs_i not in [5, 9, 15, 16, 33] for obs_i in obs_all])]

oad=[]
n_obs = len(obs_all)

n_trials_per_obs = np.zeros(len(obs_all))
for oind, obs_i in enumerate(obs_all):
	# Get the files
	obs_log_path = data_path + 'obs_%s/behav/' % obs_i
	# fields: ['observer', 'block', 'trial', 'instr_type', 'ISD', 'stimcombi', 'response', 'resptime', 'resptime2', 'respcorrect']
	# Stim Combi:
		# represents how the gratings were tilted (cw = clockwise, ccw = counter-clockwise):
			# 0 = cw (left stim) - cw (right stim)  /  1 = ccw - cw  /  2 = cw - ccw  /  3 = ccw - ccw
	filen = obs_log_path + 'obs_%i_behav_data_eyet_thresh%.1fdeg_struct_2.npy' % (obs_i, fix_thresh_indeg)
	data_all_struct = np.load(filen, encoding = 'ASCII', allow_pickle=True)[..., np.newaxis][0]

	# create a dictionary to store the data for that observer
	oad.append({})
	oad[oind]['rts'] = data_all_struct['resptime'].astype(np.float)
	oad[oind]['resps'] = data_all_struct['response'].astype(np.str)
	oad[oind]['instructs'] = data_all_struct['instr_type'].astype(np.int)
	oad[oind]['ISDs'] = data_all_struct['ISD'].astype(np.int)
	oad[oind]['stim_combis'] = data_all_struct['stimcombi'].astype(np.int)
	oad[oind]['corr_resp'] = data_all_struct['respcorrect'].astype(np.int)

	n_trials_per_obs[oind] = len(oad[oind]['rts'])

pdoad = pd.DataFrame(oad)  # pandas dataframe for all observers data

# compute DDM parameters, accuracy and RT across whole experiment for each observer
a = np.zeros(n_obs)
v = np.zeros(n_obs)
t = np.zeros(n_obs)
corr_all = np.zeros(n_obs)
rts_med_all = np.zeros(n_obs)
for obs_i in np.arange(n_obs):
	a[obs_i], v[obs_i], t[obs_i] = ezdiff(oad[obs_i]['rts'], oad[obs_i]['corr_resp'], s = 1.0)
	corr_all[obs_i] = oad[obs_i]['corr_resp'].mean()
	rts_med_all[obs_i] = np.median(oad[obs_i]['rts'])

np.save(data_path + 'corr_all_forEEG.npy', corr_all)

# compute accuracy, RT and DDM params per instruction (across all ISDs)
n_trials_inst = np.zeros([n_obs, 4])
a_inst = np.zeros([n_obs, 4])
v_inst = np.zeros([n_obs, 4])
t_inst = np.zeros([n_obs, 4])
corr_all_inst = np.zeros([n_obs, 4])
rts_med_all_inst = np.zeros([n_obs, 4])
for obs_i in np.arange(n_obs):
	for inst in np.arange(4):
		inst_mask = oad[obs_i]['instructs']==inst
		a_inst[obs_i, inst], v_inst[obs_i, inst], t_inst[obs_i, inst] =\
			ezdiff(oad[obs_i]['rts'][inst_mask], oad[obs_i]['corr_resp'][inst_mask], s = 1)
		corr_all_inst[obs_i, inst] = oad[obs_i]['corr_resp'][inst_mask].mean()
		rts_med_all_inst[obs_i, inst] = np.median(oad[obs_i]['rts'][inst_mask])
		n_trials_inst[obs_i, inst]= inst_mask.sum()

np.save(data_path + 'corr_byInst_all_forEEG.npy', corr_all_inst)

# per ISD per instruction
n_trials_ISD_inst = np.zeros([n_obs, n_t, 4])
corr_all_isd_inst = np.zeros([n_obs, n_t, 4])
rts_med_all_isd_inst = np.zeros([n_obs, n_t, 4])
for obs_i in np.arange(n_obs):
	for inst in np.arange(4):
		for isd in np.arange(n_t):
			mask = (oad[obs_i]['instructs']==inst) & (oad[obs_i]['ISDs']==isd)
			corr_all_isd_inst[obs_i, isd, inst] = oad[obs_i]['corr_resp'][mask].mean()
			rts_med_all_isd_inst[obs_i, isd, inst] = np.median(oad[obs_i]['rts'][mask])
			n_trials_ISD_inst[obs_i, isd, inst] = mask.sum()



# prepare stuff for the ANOVA
subjs = np.repeat(obs_all, 4)
hand = np.tile(np.array([1, 2, 1, 2]), n_obs)
hemi = np.tile(np.array([1, 2, 2, 1]), n_obs)

titles = ['Bound', 'Drift Rate', 'Non-dec. time', 'RT', 'Accuracy']
col = np.array([25, 196, 64])/255
fig, axs = pl.subplots(2, 3)
for data_ind, data in enumerate([a_inst, v_inst, t_inst, rts_med_all_inst, corr_all_inst]):
	ax = axs.flatten()[data_ind]
	toplot = data[:, inst_diff_order]

	ax.errorbar(x = np.arange(4), y=toplot.mean(axis=0),
		yerr=toplot.std(axis=0)/(toplot.shape[0]**.5), color='k', fmt = 'o',
	linestyle='', mfc = col, ms=8, mec='k', elinewidth=2.5, mew=2.5,
	capthick=0, ecolor = 'k', linewidth=2.5, zorder=2)

	# 2way RM-ANOVA
	df = pd.DataFrame({'obs':subjs, 'hand':hand, 'hemi':hemi, 'perf':toplot.flatten()})
	# run anova
	aov = pg.rm_anova(data=df, dv='perf', subject='obs', within=['hand', 'hemi'],
		detailed=True, effsize='n2')
	fs = aov['F'].values
	pvals = aov['p-unc']
	etas = aov['n2']

	# put F and p values as subplots titles
	ax.set_title('%s\nF=[%.2f, %.2f, %.2f]\npval=[%.4f, %.4f, %.4f]\neta2=[%.4f, %.4f, %.4f]' %\
		(titles[data_ind], fs[0], fs[1], fs[2], pvals[0],pvals[1],pvals[2],  etas[0],etas[1],etas[2]),
		fontsize=8)
	ax.set_xticks(np.arange(4)); ax.set_xticklabels(insttxts[inst_diff_order])
	ax.grid(color = [.7, .7, .7], linewidth=.5)
	ax.set_xlim(-.5, 3.5)
	print('%s:\n\tF=[%.6f, %.6f, %.6f]\n\tp=[%.6f, %.6f, %.6f]\n\teta2=[%.4f, %.4f, %.4f]\n\n' %\
		(titles[data_ind], fs[0], fs[1], fs[2], pvals[0],pvals[1],pvals[2],etas[0],etas[1],etas[2]))
fig.set_size_inches([8.5, 6])
pl.tight_layout()



################################################################################################################
#												ACCURACY-BY-ISD PER RULE
################################################################################################################

# detrend data
corr_all_isd_inst2 = sig.detrend(corr_all_isd_inst, axis=1)

step = .05
# we add 5 time points before and 5 after the Accuracy-by-ISD to
# obtain a 1 second signal
n_t_pad = n_t + 5 + 5
freqpad = np.fft.fftfreq(n_t_pad, step)
freqmaskpad = freqpad > 0
freqpad = freqpad[freqmaskpad]
n_freqpad = freqmaskpad.sum()
mask_theta = (freqpad>=4) & (freqpad<=7)
mask_theta_ind = np.argwhere(mask_theta).squeeze()

# pad + FFT
amp_padded = pad_fft(data=corr_all_isd_inst2, n_t_pad=n_t_pad, freqmaskpad=freqmaskpad)

# get peak
peak_theta_perInst = np.argmax(amp_padded[:, mask_theta, :], axis=1)

# make peak in hertz and order it by instruction difficulty
peak_theta_perInst_inHz_ordDiff = freqpad[mask_theta][peak_theta_perInst[:,inst_diff_order]]

# plot
fig, ax = pl.subplots(1, 1)
toplot = (peak_theta_perInst_inHz_ordDiff - peak_theta_perInst_inHz_ordDiff.mean(axis=1)[:, np.newaxis])
stdd = peak_theta_perInst_inHz_ordDiff.std(axis=1)[:, np.newaxis]
stdd[(stdd==0)] = 1
toplot = toplot / stdd


ax.errorbar(x = np.arange(4), y = toplot.mean(axis=0),
	yerr = toplot.std(axis=0)/np.sqrt(n_obs), mfc=col, color = 'k', fmt = '',
	marker='o', lw = 2.5, ms=10, mec='k', mew=2.5, linestyle='')

ax.set_xticks(np.arange(4))
ax.set_xticklabels(insttxts[inst_diff_order])
ax.grid(color = [.7, .7, .7], linewidth=.5)
ax.hlines(y=0, xmin=-.5, xmax=4.5)
ax.set_xlim(-.5, 3.5)


## stats
# prepare data for the ANOVA
subjs = np.repeat(obs_all, 4)
hand = np.tile(np.array([1, 2, 1, 2]), n_obs)
hemi = np.tile(np.array([1, 2, 2, 1]), n_obs)
df = pd.DataFrame({'obs':subjs, 'hand':hand, 'hemi':hemi,
			'theta_peak':toplot.flatten()})

aov = pg.rm_anova(data=df, dv='theta_peak', subject='obs', within=['hand', 'hemi'],
	detailed=True, effsize='n2')
title='2*2 RM-ANOVA (target-loc, hand)\nF(1, 33)=%s\np=%s' %\
	(np.str(aov['F'].values), np.str(aov['p-unc'].values))
ax.set_title(title, fontsize=8)
print(title)
fig.set_size_inches([4, 4])



