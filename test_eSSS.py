#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 12:43:54 2023

@author: maggie clarke

Test eSSS for supression of vibrational artifacts
"""
import mne
import os.path as op
from mne.preprocessing import maxwell_filter

path = '/home/maggie/data/'

# read in system specific cross-talk and fine calibration files
ct = op.join(path, 'ct_sparse.fif')
fc = op.join(path, 'sss_cal.dat')

# read in raw data
raw = mne.io.Raw(op.join(path, 'sam', 'pilot_4_task_1_raw.fif'), preload=True)

# perform regular tSSS for comparison
tsss = maxwell_filter(raw, calibration=fc, cross_talk=ct, st_duration=10)

# here we will bandapss the empty-room data around the 29.5Hz vibration artifact
erm = mne.io.Raw(op.join(path, 'sam', 'empty_room.fif'),
                 preload=True).filter(l_freq=27, h_freq=31)

## compute erm projectors from empty-room to use for eSSS
erm_proj = mne.compute_proj_raw(erm, meg='combined')
# plot the combined projectors
mne.viz.plot_projs_topomap(erm_proj, colorbar=True,
                           info=erm.info)
# perform eSSS on raw data
esss = maxwell_filter(raw, calibration=fc, cross_talk=ct, st_duration=10, 
                      extended_proj=erm_proj)

# plot tSSS and eSSS spectra to ensure the artifact is suppressed
tsss.plot_psd(fmin=1,fmax=55)
esss.plot_psd(fmin=1,fmax=55)
