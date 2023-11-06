#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 15:00:41 2023

@author: maggie clarke

Step #1
Append task data files, perform tSSS & movement compensation
"""
import mne
import os
import os.path as op
from mne.preprocessing import maxwell_filter, compute_average_dev_head_t
from mne.chpi import (compute_chpi_amplitudes, compute_chpi_locs, 
                      compute_head_pos, write_head_pos)

####### set these  before running #############################################
path = '/home/maggie/data/'

# setup subject name ### change this for each subject
s = 'vml_meg_015'

# set to True if headpos has not already been calculated & saved
calc_headpos = True
###############################################################################

# read in system specific cross-talk and fine calibration files 
ct = op.join(path, 'ct_sparse.fif')
fc = op.join(path, 'sss_cal.dat')

# append task runs & save output
raw = mne.io.Raw(op.join(path, 'VML', '%s' %s,
                         '%s_1_raw.fif' %s), preload=True)
raw2 = mne.io.Raw(op.join(path, 'VML', '%s' %s,
                         '%s_2_raw.fif' %s), preload=True)
raw.append([raw, raw2])
raw.save(op.join(path, 'VML', '%s' %s, '%s_all_raw.fif' %s))
# delete old raw objects for memory
del raw, raw2

# read in raw data
raw = mne.io.Raw(op.join(path, 'VML', '%s' %s, 
                         '%s_all_raw.fif' %s), preload=True)

# make a copy of raw and apply a lowpass filter (55 Hz) for vizualization
raw_filtered = raw.copy().load_data().filter(l_freq=None, h_freq=55.)

# plot raw data and visually inspect for bad channels
raw_filtered.plot()
# delete filtered raw object
del raw_filtered

# set bad channels in data structure - get these from 
# runsheet OR visual inspection
raw.info['bads'] = ['MEG1213']

# setup for movement compensation by extracting coil info
if calc_headpos == True:
    amps = compute_chpi_amplitudes(raw)
    locs = compute_chpi_locs(raw.info, amps)
    pos = compute_head_pos(raw.info, locs)
    # save pos file (computing takes awhile)
    write_head_pos(op.join(path, 'VML', '%s' %s, '%s_all_pos.fif' %s), pos)
else:
    pos = mne.chpi.read_head_pos(op.join(path, 'VML', '%s' %s, 
                                         '%s_all_pos.fif' %s))

# get initial head position and calculate average position (across time)
orig_head_dev_t = mne.transforms.invert_transform(raw.info["dev_head_t"])
avg_head_dev_t = mne.transforms.invert_transform(compute_average_dev_head_t
                                                     (raw, pos))

# plot head positions over time (green=initial, red=average)
if not os.path.exists(op.join(path, 'VML', '%s' %s, 'figures')):
    os.makedirs(op.join(path, 'VML', '%s' %s, 'figures'))

fig = mne.viz.plot_head_positions(pos)
for ax, val, val_ori in zip(
    fig.axes[::2],
    avg_head_dev_t["trans"][:3, 3],
    orig_head_dev_t["trans"][:3, 3],
):
    ax.axhline(1000 * val, color="r")
    ax.axhline(1000 * val_ori, color="g")
fig.savefig(op.join(path, 'VML', '%s' %s, 'figures', '%s_att_all_headpos' %s))

# use average position for movement compensation destination    
destination = (raw.info['dev_head_t']['trans'][:3, 3])

# perform tSSS - we don't need to apply eSSS for event-related data
tsss_mc = maxwell_filter(raw, calibration=fc, cross_talk=ct,
                         st_duration=6, head_pos=pos, 
                         destination=destination, st_correlation=0.98)

# save the tSSS processed file for next steps
tsss_mc.save(op.join(path, 'VML', '%s' %s, '%s_all_tsss.fif' %s))
