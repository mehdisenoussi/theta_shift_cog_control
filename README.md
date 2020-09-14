## Welcome to the "Theta shift" repository.

This is a set of Python scripts to perform the analyses for the "theta shift in cognitive control" experiment dataset. (Senoussi, Verbeke, Desender, De Loof, Talsma & Verguts - 2020 (Submitted))
These scripts were written and tested on Python version 3.7. (full list of Python packages used to come)
Here is a link to the preprint: [https://www.biorxiv.org/content/10.1101/2020.08.30.273706v1](https://www.biorxiv.org/content/10.1101/2020.08.30.273706v1)

The dataset is available through this Open Science Framework repository: https://osf.io/nwh87/?view_only=b11ee1f860804da582c816fe8acdecad

To run any of the scripts you should be in the "theta_shift_cog_control" directory and place the data from OSF in the "data" folder inside this directory.

# Model
The scripts to run the model, plot the supplementary figures showing how it works and analyze its behavioral results are in the folder "model".
To reproduce the behavioral results figures from the paper you need to run "model/theta_sync_compet_model.py" (which will use functions from "model/theta_sync_compet_model_utils.py"). Then run "model/model_behav_analysis.py".
The script "model/model_plot_supplFig.py" will run a small amount of simulations in order to plot the behavior of different units and nodes of the model, e.g. as represented in Supplementary Fig. 1 of the manuscript.


# Experiment
The experimental results must be analyzed using scripts from the "eyet", "behav" and "eeg" directories.
1) First run the "eyet/eyet_match_EEG_triggers_epoch_concat_data.py" script to match eyetracking data to EEG data.
2) Then run "eyet/eyet_reject_behav_and_EEG.py" to find all trials that need to be rejected due to eye movements.
3) Run "behav/behav_analysis.py" to reproduce the behavioral results of the study.
4) Then run "eeg/eeg_preproc_pre-stim.py" to preprocess the EEG data
5) Then run "eeg/eeg_estimate_theta_peak_preStim.py" to compute the peak and amplitude measures of theta oscillations on EEG data
6) Finally, run "eeg/eeg_peak_shift_stats_plot.py" to reproduce the final results on the EEG data

Let us know if you need any assistance to run these scripts using the contact information provided in our bioRxiv preprint (see above).
Thank you for your interest!