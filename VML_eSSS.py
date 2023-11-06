#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 13:22:08 2023

@author: maggie

"""
import mne
import os.path as op
from mne.preprocessing import maxwell_filter

path = '/home/maggie/data/'

subj = 'vml_meg_011'

# read in system specific cross-talk and fine calibration files
ct = op.join(path, 'ct_sparse.fif')
fc = op.join(path, 'sss_cal.dat')

# read in raw data
raw = mne.io.Raw(op.join(path, 'VML', '%s' %subj, 
                         '%s_rest_raw.fif' %subj), preload=True)

# perform regular tSSS for comparison
tsss = maxwell_filter(raw, calibration=fc, cross_talk=ct, st_duration=10)

# read in erm
erm = mne.io.Raw(op.join(path, 'VML', '%s' %subj, '%s_erm_raw.fif' %subj),
                 preload=True)

# create 3 copies, bandpass around each artifact
erm1 = erm.copy().filter(l_freq=26, h_freq=31)
erm2 = erm.copy().filter(l_freq=39, h_freq=43)
erm3 = erm.copy().filter(l_freq=49, h_freq=53)

# create a seperate projector for each artifact
erm_proj_1 = mne.compute_proj_raw(erm1, meg='combined')
erm_proj_2 = mne.compute_proj_raw(erm2, meg='combined')
erm_proj_3 = mne.compute_proj_raw(erm3, meg='combined')

# make a list of the projectors to pass to maxwell_filter for eSSS
projs = erm_proj_1 + erm_proj_2 + erm_proj_3

# plot the combined projectors
mne.viz.plot_projs_topomap(projs, colorbar=True,
                           info=erm.info)
# perform eSSS on raw data
esss = maxwell_filter(raw, calibration=fc, cross_talk=ct, st_duration=10, 
                      extended_proj=projs)

# plot tSSS and eSSS spectra to ensure the artifact is suppressed
erm.plot_psd(fmin=1, fmax=55)
tsss.plot_psd(fmin=1,fmax=55)
esss.plot_psd(fmin=1,fmax=55)
