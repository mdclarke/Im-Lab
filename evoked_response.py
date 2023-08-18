#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 09:20:23 2023

@author: maggie
"""
import mne
import os
import os.path as op
import numpy as np
from mne.preprocessing import maxwell_filter, compute_average_dev_head_t
from mne.chpi import compute_chpi_amplitudes, compute_chpi_locs, compute_head_pos, write_head_pos

####### set params ########
path = '/home/maggie/data/'

#subject
s = 'p004'

# set to True if headpos has not already been calculated & saved
calc_headpos = False
##########################

# read in system specific cross-talk and fine calibration files
ct = op.join(path, 'ct_sparse.fif')
fc = op.join(path, 'sss_cal.dat')

# read in raw data
raw = mne.io.Raw(op.join(path, 'ENS', '%s' %s, 
                         '%s_MEG_ENS_navon_raw.fif' %s), preload=True)

# make a copy of raw and filter (you can change the bandpass here or choose 
# to not filter)
raw_filtered = raw.copy().load_data().filter(l_freq=None, h_freq=50.)

# plot raw data and visually inspect for bad channels
#raw_filtered.plot()

# set bad channels in data structure - get these from 
# runsheet OR visual inspection
raw_filtered.info['bads'] = ['MEG1213', 'MEG1132', 'MEG1742']

## filter out the cHPI frequencies - only use this if you don't bandpass filter.
## the lowpass filter will get rid of HPI high frequencies. I commented this 
## step out for now.
#mne.chpi.filter_chpi(raw)

# setup for movement compensation by extracting coil info
# (perform on unfiltered raw)
if calc_headpos == True:
    amps = compute_chpi_amplitudes(raw)
    locs = compute_chpi_locs(raw.info, amps)
    pos = compute_head_pos(raw.info, locs)
    # save pos file (computing takes awhile)
    write_head_pos(op.join(path, 'ENS', '%s' %s, '%s_pos.fif' %s), pos)
else:
    pos = mne.chpi.read_head_pos(op.join(path, 'ENS', '%s' %s, 
                                         '%s_pos.fif' %s))

# get initial head position and calculate average (across time) head position
orig_head_dev_t = mne.transforms.invert_transform(raw.info["dev_head_t"])
avg_head_dev_t = mne.transforms.invert_transform(compute_average_dev_head_t
                                                     (raw, pos))

# plot head positions over time (green=initial, red=average)
if not os.path.exists(op.join(path, 'ENS', '%s' %s, 'figures')):
    os.makedirs(op.join(path, 'ENS', '%s' %s, 'figures'))
write_head_pos(op.join(path, 'ENS', '%s' %s, 'figures', 
                       '%s_pos.fif' %s), pos)

fig = mne.viz.plot_head_positions(pos)
for ax, val, val_ori in zip(
    fig.axes[::2],
    avg_head_dev_t["trans"][:3, 3],
    orig_head_dev_t["trans"][:3, 3],
):
    ax.axhline(1000 * val, color="r")
    ax.axhline(1000 * val_ori, color="g")
fig.savefig(op.join(path, 'ENS', '%s' %s, 'figures', '%s_headpos' %s))

# use average position for movement compensation destination    
destination = (raw.info['dev_head_t']['trans'][:3, 3])

# perform tSSS - some params for this function will change when running
# pediatric subjects
tsss_mc = maxwell_filter(raw_filtered, calibration=fc, cross_talk=ct,
                         st_duration=10, head_pos=pos, 
                         destination=destination)
# find triggers
events = mne.find_events(raw, stim_channel='STI101', 
                       shortest_event=1/raw.info['sfreq'])

adjusted_events = events.copy()

# adjust triggers for visual delay from projector
t_adjust= -35e-3 #35 ms
t_adj = int(np.round(t_adjust *raw.info['sfreq']))
adjusted_events[:,0] += t_adj

event_dict = {
    "fixation cross": 1,
    "Visual1": 2,
    "Visual2": 3,
    "Visual3": 4,
    "Visual4": 5}

# plot triggers
fig = mne.viz.plot_events(
    events, event_id=event_dict, sfreq=raw.info["sfreq"], 
    first_samp=raw.first_samp)

#reject mags and grads > 9000 fT, 4000 fT/cm
reject = dict(mag=9000e-15, grad=4000e-13)

epochs = mne.Epochs(tsss_mc, adjusted_events, event_id=event_dict, 
                    tmin=-0.2, tmax=0.5, baseline=(-0.15, 0.05), 
                    reject=reject, preload=True)

#epochs.equalize_event_counts(epochs)  # if needed

# show epochs dropped
epochs.plot_drop_log()

conditions = ['Visual1', 'Visual2', 'Visual3', 'Visual4']

vis1 = epochs["Visual1"].average()
vis2 = epochs["Visual2"].average()
vis3 = epochs["Visual3"].average()
vis4 = epochs["Visual4"].average()

# to save individual evoked files
# for c in conditions:
#     evoked = epochs[c].average()
#     evoked.save(op.join(path, 'ENS', '%s_MEG_ENS_navon_%s_ave.fif' % (s, c)))
    
mne.viz.plot_compare_evokeds(dict(Visual1=vis1, Visual2=vis2, Visual3=vis3, 
                                  Visual4=vis4), legend="upper left", 
                             show_sensors="upper right")
    
