# Behavioral analysis for theta shift study
# by Mehdi Senoussi (02/08/19)

# This script loads the data, computes accuracy and RT by instruction and ISD (instruction-stimulus delay)
# Then it computes and prints the analysis of the link between accuracy, titrated tilt and peak theta frequency.
# Finally it computes the Fourier transform of the Accuracy by ISD, create a null distribution of no oscillation
# in the ISD and plots the correlation between accuracy and frequency.

import pylab as pl
import pandas as pd
import glob
from scipy import stats
import numpy as np
from statsmodels.formula.api import ols
from mne import parallel as par
from scipy import signal as sig
from statsmodels.stats.anova import AnovaRM
import matplotlib as mpl
from matplotlib import cm

# simple EZ-diffusion model based on Wagenmakers et al, 2007
def ezdiff(rt, correct, s = 1.0):

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


def pad_fft_normDiff(data, n_t_pad, freqpad, freqmaskpad, mask_theta_ind):
	n_obs, n_instr = data.shape[0], data.shape[2]
	data_pad = np.zeros(shape=[n_obs, n_t_pad, n_instr])
	for obs_ind in np.arange(n_obs):
		for inst in np.arange(n_instr):
			avg_obs_data = data[obs_ind, :, inst].mean()
			data_pad[obs_ind, :, inst] = np.hstack([np.repeat(avg_obs_data, 4),
				data[obs_ind, :, inst], np.repeat(avg_obs_data, 5)])

	# amplitude of padded signal
	amp_data_pad_goodfreqs = np.abs(np.fft.fft(data_pad, axis=1))[:, freqmaskpad, :]

	# compute peak measure
	diffn_data_amp_pad = np.rollaxis(np.array([[[(temp[i] - np.mean([temp[i-1],temp[i+1]]))/np.mean([temp[i-1],temp[i+1]])\
		for i in mask_theta_ind]\
		for temp in amp_data_pad_goodfreqs[:, :, inst]]\
		for inst in np.arange(n_instr)]),1)

	return diffn_data_amp_pad


def do_anova(meas, n_subjs, thetas, n_rules=2):
	n_subjs = n_reps_bundled
	n_thetas = len(thetas)
	df2_arr = np.zeros(shape = [n_subjs*n_thetas*n_rules, 4])
	index = 0
	for subj_i in np.arange(n_subjs):
		for theta_ind, theta_i in enumerate(np.arange(n_thetas)):#enumerate([0, -1]):
			for rule_i in np.arange(n_rules):
				df2_arr[index, :] = [subj_i, rule_i, theta_ind, meas[theta_i, rule_i, subj_i]]
				index += 1
	df2 = pd.DataFrame(df2_arr, columns=['obs', 'ruleFact', 'thetaFact', 'perf'])
	aovrm = AnovaRM(data=df2, depvar='perf', subject='obs', within=['ruleFact', 'thetaFact'], aggregate_func=np.mean)
	res = aovrm.fit()
	fs = res.anova_table['F Value'].values
	pvals = res.anova_table['Pr > F'].values
	return fs, pvals


isd_val = np.arange(1.7, 2.21, .05)
n_t = len(isd_val)
freq = np.fft.fftfreq(n_t, .05)
n_freq = int((n_t - 1) / 2)
freqmask = freq > 0
n_t = 11

# names of drift-diffusion model parameters
ddm_var_names = ['bound', 'drift', 'non-dec-time']

# paths
simu_path = './model/simulations/'
data_path = simu_path + 'LFC_compet_sw2_0.50_kick_0.50_thresh_0.10_cgSd_1.00_sigmaCompet_0.075_inhCompet_0.10_alphCompet_0.130_tilt_0.0200_dampThetaCoef_0.0050_1/'

# number of repetitions of each condition
n_reps = 50

# difference between center gamma frequencies
drift = 0
# thresholds of the model (in integrator unit)
threshold = 4
# theta frequencies of MFC unit
freq_step = .5
thetas = np.arange(4, 7.1, freq_step)


######################## BUNDLES ##########################
# how many repetitions should we group (bundle) toghether?
reps_bundle = 3
# how many repetition go into a bundle
n_reps_bundled = np.int(np.ceil(n_reps/reps_bundle))

rep_bundle_inds = np.arange(0, n_reps, reps_bundle)
reps_inds = np.arange(n_reps)
######################## ####### ##########################

# instructions are encoded in that order indices: [0, 1, 2, 3]
insttxts = np.array(['LL', 'RR', 'LR', 'RL'])

# We group instructions by difficulty as same-side rules (RR, LL) are
# algorithmically equivalent for the model, and same thing for different-side rules (RL & LR).
n_instr = 2

a = np.zeros(shape = [len(thetas), n_reps_bundled])
v = np.zeros(shape = [len(thetas), n_reps_bundled])
t = np.zeros(shape = [len(thetas), n_reps_bundled])
corr_all = np.zeros(shape = [len(thetas), n_reps_bundled])
rts_med_all = np.zeros(shape = [len(thetas), n_reps_bundled])

a_isd = np.zeros(shape = [len(thetas), n_t, n_reps_bundled])
v_isd = np.zeros(shape = [len(thetas), n_t, n_reps_bundled])
t_isd = np.zeros(shape = [len(thetas), n_t, n_reps_bundled])

corr_all_isd = np.zeros(shape = [len(thetas), n_t, n_reps_bundled])
rts_med_all_isd = np.zeros(shape = [len(thetas), n_t, n_reps_bundled])

a_inst = np.zeros(shape = [len(thetas), n_instr, n_reps_bundled])
v_inst = np.zeros(shape = [len(thetas), n_instr, n_reps_bundled])
t_inst = np.zeros(shape = [len(thetas), n_instr, n_reps_bundled])

corr_all_inst = np.zeros(shape = [len(thetas), n_instr, n_reps_bundled])
rts_med_all_inst = np.zeros(shape = [len(thetas), n_instr, n_reps_bundled])

a_isd_inst = np.zeros(shape = [len(thetas), n_t, n_instr, n_reps_bundled])
v_isd_inst = np.zeros(shape = [len(thetas), n_t, n_instr, n_reps_bundled])
t_isd_inst = np.zeros(shape = [len(thetas), n_t, n_instr, n_reps_bundled])

corr_all_isd_inst = np.zeros(shape = [len(thetas), n_t, n_instr, n_reps_bundled])
rts_med_all_isd_inst = np.zeros(shape = [len(thetas), n_t, n_instr, n_reps_bundled])


print('simu: %s' % data_path.split('/')[-2])
for ind_theta, theta_freq in enumerate(thetas):
	print('theta: %.2fHz' % theta_freq)
	filen = data_path + 'Behavioral_Data_simulation_thetaFreq%.2fHz_thresh%.1f_drift%.1f.csv' % (theta_freq, threshold, drift)
	pd_data = pd.read_csv(filen)
	# discard non-responded trials
	pd_data = pd_data[pd_data['response'] != -1]

	if pd_data.reps.min() == 50:
		pd_data.reps = pd_data.reps.values - 50

	new_inst = pd_data.instr.values.copy()
	new_inst[new_inst == 1] = 0
	new_inst[new_inst == 2] = 1
	new_inst[new_inst == 3] = 1
	pd_data.instr = new_inst

	# make rt in seconds
	pd_data.rt /= 1000

	for rep_ind, rep_n in enumerate(rep_bundle_inds):
		# check we don't go over the number of reps
		last_rep_this_bundle = rep_n + reps_bundle
		if last_rep_this_bundle > n_reps:
			last_rep_this_bundle = n_reps

		this_bundle_inds = reps_inds[rep_n:last_rep_this_bundle]
		this_bundle_mask = np.array([data_rep in this_bundle_inds for data_rep in pd_data.reps])
		pd_data_rep_n = pd_data[this_bundle_mask]

		# overall
		a[ind_theta, rep_ind],\
		v[ind_theta, rep_ind],\
		t[ind_theta, rep_ind] =\
			ezdiff(pd_data_rep_n['rt'], pd_data_rep_n['accuracy'], s = 1.0)
		corr_all[ind_theta, rep_ind] = pd_data_rep_n['accuracy'].mean()
		rts_med_all[ind_theta, rep_ind] = np.median(pd_data_rep_n['rt'])

		# by ISD
		for isd in np.arange(n_t):
			isd_mask = pd_data_rep_n['isd']==isd
			a_isd[ind_theta, isd, rep_ind],\
			v_isd[ind_theta, isd, rep_ind],\
			t_isd[ind_theta, isd, rep_ind] = \
				ezdiff(pd_data_rep_n['rt'][isd_mask], pd_data_rep_n['accuracy'][isd_mask], s = 1.0)
			corr_all_isd[ind_theta, isd, rep_ind] =\
				pd_data_rep_n['accuracy'][isd_mask].mean()
			rts_med_all_isd[ind_theta, isd, rep_ind] =\
				np.median(pd_data_rep_n['rt'][isd_mask])

		# by Instructions
		for inst in np.arange(n_instr):
			inst_mask = pd_data_rep_n['instr']==inst
			a_inst[ind_theta, inst, rep_ind],\
			v_inst[ind_theta, inst, rep_ind],\
			t_inst[ind_theta, inst, rep_ind] = \
				ezdiff(pd_data_rep_n['rt'][inst_mask], pd_data_rep_n['accuracy'][inst_mask], s = 1.0)
			corr_all_inst[ind_theta, inst, rep_ind] =\
				pd_data_rep_n['accuracy'][inst_mask].mean()
			rts_med_all_inst[ind_theta, inst, rep_ind] =\
				np.median(pd_data_rep_n['rt'][inst_mask])


		# by ISD by Instructions
		for isd in np.arange(n_t):
			isd_mask = pd_data_rep_n['isd']==isd
			for inst in np.arange(n_instr):
				isd_inst_mask = isd_mask & (pd_data_rep_n['instr']==inst)
				corr_all_isd_inst[ind_theta, isd, inst, rep_ind] =\
					pd_data_rep_n['accuracy'][isd_inst_mask].mean()
				rts_med_all_isd_inst[ind_theta, isd, inst, rep_ind] =\
					np.median(pd_data_rep_n['rt'][isd_inst_mask])


################################################################################
#############################		 plot data 		############################
################################################################################

cols = ["#7ed35fff", "#ffb505ff"]

# group by instruction difficulty
insttxts = np.array(['Easy', 'Difficult'])
inst_diff_order = np.array([0, 1])

# Plotting effect of rule difficulty and theta frequency on
# DDM parameters (bound, drift rate, non-decision time), RT and accuracy
var_names = ['bound', 'drift', 'non-dec-time', 'RT', 'Acc']
fig, axes = pl.subplots(2, 3)
for ind, meas in enumerate([a_inst, v_inst, t_inst, rts_med_all_inst, corr_all_inst]):
	ax = axes.flatten()[ind]
	for inst_ind, inst in enumerate(inst_diff_order):
		ax.errorbar(x=thetas+((inst_ind-1)/12.), y=meas[:, inst, :].mean(axis=-1),
			yerr=meas[:, inst, :].std(axis=-1) ,#/ np.sqrt(n_reps_bundled),
			fmt='o-', color=cols[inst_ind], mec='black', zorder=1, mew=2, ms = 5)

	fs, pvals = do_anova(meas, n_reps_bundled, thetas, n_rules=2)
	ax.set_title('%s\nF=[%.1f, %.1f, %.1f]\npval=[%.3f, %.3f, %.3f]' %\
		(var_names[ind], fs[0], fs[1], fs[2], pvals[0],pvals[1],pvals[2]),
		fontsize=8)

axes[1, 2].remove()
axes[1, 1].set_xlabel('Model\'s Theta freq. (Hz)')
pl.tight_layout()
pl.suptitle('Effect of rule difficulty and theta frequency on DDM params, RT and accuracy', fontsize=8)
pl.subplots_adjust(left=.09, bottom=.1, right=.97, top=.85, wspace=.28, hspace=.32)



############################################################################
################## 		compute "optimal" theta freq 		################
############################################################################

optim_theta_freq = np.argmax(corr_all_inst[:, :, :].squeeze(), axis=0)*freq_step + thetas[0]

fig, ax = pl.subplots(1,1)
violins = ax.violinplot(optim_theta_freq.T, showmedians=True, vert=False)
ax.set_yticks([1,2])
ax.set_yticklabels(['Easy', 'Difficult'])
t,p = stats.ttest_rel(optim_theta_freq[0,:], optim_theta_freq[1,:])
ax.set_title('ACCURACY\nT-test: t=%.1f, p=%.8f' % (t, p))



########################################################################################
######### 		Plot model oscillation in accuracy-by-ISD 		################
########################################################################################
step = .05
# we add 4 time points before and 5 after the Accuracy-by-ISD to
# obtain a 1 second signal
n_t_pad = n_t + 4 + 5
freqpad = np.fft.fftfreq(n_t_pad, step)
freqmaskpad = freqpad > 0
freqpad = freqpad[freqmaskpad]
n_freqpad = freqmaskpad.sum()
mask_theta = (freqpad>=4) & (freqpad<=7)
mask_theta_ind = np.argwhere(mask_theta).squeeze()

# Low theta
corr_all_isd2 = np.rollaxis(corr_all_isd.squeeze().copy()[0, ...], 1)[:,:,np.newaxis]
# detrend
corr_all_isd2 = sig.detrend(corr_all_isd2, axis=1)
# pad + FFT + peak measure
diffn_amp_pad_lowF = pad_fft_normDiff(data=corr_all_isd2, n_t_pad=n_t_pad, freqpad=freqpad,
	freqmaskpad=freqmaskpad, mask_theta_ind=mask_theta_ind)

# High theta
corr_all_isd2 = np.rollaxis(corr_all_isd.squeeze().copy()[-1, ...], 1)[:,:,np.newaxis]
# detrend
corr_all_isd2 = sig.detrend(corr_all_isd2, axis=1)
# pad + FFT + peak measure
diffn_amp_pad_highF = pad_fft_normDiff(data=corr_all_isd2, n_t_pad=n_t_pad, freqpad=freqpad,
	freqmaskpad=freqmaskpad, mask_theta_ind=mask_theta_ind)



fig, axs = pl.subplots(2, 2)
col = 'k'

theta_low_acc = corr_all_isd[0, :, :].squeeze()*100
theta_high_acc = corr_all_isd[-1, :, :].squeeze()*100

axs[0,0].errorbar(x = isd_val, y=theta_low_acc.mean(axis=-1), yerr = theta_low_acc.std(axis=-1),
	color = col, fmt = 'o-')
axs[0,1].errorbar(x = isd_val, y=theta_high_acc.mean(axis=-1), yerr = theta_high_acc.std(axis=-1),
	color = col, fmt = 'o-')
axs[0,0].set_ylim(50, 100); axs[0,1].set_ylim(50, 100)
axs[0,0].set_yticks(np.arange(50, 91, 10)); axs[0,1].set_yticks(np.arange(50, 91, 10))

axs[1,0].hist(np.argmax(diffn_amp_pad_lowF, axis=2).flatten()+4,
	bins=np.arange(5)+3.5,density=True, width=.8)

axs[1,1].hist(np.argmax(diffn_amp_pad_highF, axis=2).flatten()+4,
	bins=np.arange(5)+3.5,density=True, width=.8)



#############################################
########## Across all MFC theta #############
#############################################

n_estim_theta_freq = 4
diff_amp_pad_all = np.zeros(shape = [len(thetas), n_reps_bundled, n_estim_theta_freq])
for theta_freq_ind in np.arange(len(thetas)):
	print('MFC theta: %i' % thetas[theta_freq_ind])
	corr_all_isd2 = np.rollaxis(corr_all_isd.squeeze().copy()[theta_freq_ind, ...], 1)[:,:,np.newaxis]
	# detrend
	corr_all_isd2 = sig.detrend(corr_all_isd2, axis=1)
	# pad + FFT + peak measure
	diffn_amp_pad = pad_fft_normDiff(data=corr_all_isd2, n_t_pad=n_t_pad, freqpad=freqpad,
		freqmaskpad=freqmaskpad, mask_theta_ind=mask_theta_ind)
	# store them all in diff_amp_pad_all array
	diff_amp_pad_all[theta_freq_ind,:,:] = diffn_amp_pad.squeeze()


gold_col = np.array([254, 226, 52, 50])/255
obs_all = np.arange(n_reps_bundled)
fig, ax = pl.subplots(1, 1)
ax.plot([3.5, 7.5], [3.5, 7.5], 'k--')
ax.violinplot(freqpad[mask_theta][np.argmax(diff_amp_pad_all, axis=-1).squeeze()].T,
	positions=thetas, showmedians=True, showextrema=False, widths=.4)
for theta_ind, mfc_theta in enumerate(thetas):
	data = diff_amp_pad_all[theta_ind, :, :]
	data_peak = np.argmax(data, axis=-1)
	toplot = freqpad[mask_theta][data_peak]
	x = mfc_theta

	ax.plot((np.arange(n_reps_bundled)-(n_reps_bundled/2))/(n_reps_bundled*6)+np.repeat(x, n_reps_bundled),
		toplot, 'o', mfc = gold_col, mec=[0,0,0,.1], mew= 1.2, zorder=1, ms=9)
ax.set_xlim(3.5,7.5); ax.set_ylim(3.5,7.5)
ax.set_xticks(thetas); ax.set_yticks(thetas)


data = diff_amp_pad_all
data_peak = freqpad[mask_theta][np.argmax(data, axis=-1)]
corr_x, corr_y = np.repeat(thetas, n_reps_bundled), data_peak.flatten()
rho, pval = stats.spearmanr(corr_x, corr_y)
sse = np.sqrt((corr_x - corr_y)**2).sum()
ax.set_title('r=%.2f, p=%.6f, SSE:%.1f' % (rho, pval, sse),
	fontdict={'fontsize':10, 'color':'k', 'weight': 'bold'})
m, b = np.polyfit(corr_x, corr_y, 1)
xs = np.array([corr_x.min(), corr_x.max()])
ax.plot(xs, m*xs + b, color='gold', linewidth=3)


