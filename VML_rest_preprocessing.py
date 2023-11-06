#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 15:16:25 2023

@author: maggie clarke

Suppress vibrational artifacts in rest data with eSSS 
"""
import mne
import os.path as op
from mne.preprocessing import maxwell_filter

####### set these  before running #############################################
path = '/home/maggie/data/'

# setup subject name ### change this for each subject
s = 'vml_meg_015'
###############################################################################

# read in system specific cross-talk and fine calibration files
ct = op.join(path, 'ct_sparse.fif')
fc = op.join(path, 'sss_cal.dat')

# read in raw data
raw = mne.io.Raw(op.join(path, 'VML', '%s' %s, 
                         '%s_rest_raw.fif' %s), preload=True)

# read in empty-room file
erm = mne.io.Raw(op.join(path, 'VML', '%s' %s, '%s_erm_raw.fif' %s),
                 preload=True)

# plot raw and erm spectra to ensure all 3 artifacts exist 
# in the spectra (~29, 42, 50 Hz)
# if they do, continue with next steps
fig = raw.plot_psd(fmin=1, fmax=55)
fig.savefig(op.join(path, 'VML', '%s' %s, 'figures', '%s_raw_spectra' %s))
fig = erm.plot_psd(fmin=1, fmax=55)
fig.savefig(op.join(path, 'VML', '%s' %s, 'figures', '%s_erm_spectra' %s))

# perform regular tSSS for comparison
tsss = maxwell_filter(raw, calibration=fc, cross_talk=ct, st_duration=6)

# create 3 copies of erm and bandpass around each vibrational artifact
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
fig = mne.viz.plot_projs_topomap(projs, colorbar=True, info=erm.info)
fig.savefig(op.join(path, 'VML', '%s' %s, 'figures', '%s_projs' %s))

# perform eSSS on raw data
esss = maxwell_filter(raw, calibration=fc, cross_talk=ct, st_duration=6, 
                      extended_proj=projs)

# plot tSSS and eSSS spectra to ensure the artifact is suppressed
fig = tsss.plot_psd(fmin=1,fmax=55)
fig.savefig(op.join(path, 'VML', '%s' %s, 'figures', '%s_tsss_spectra' %s))
fig = esss.plot_psd(fmin=1,fmax=55)
fig.savefig(op.join(path, 'VML', '%s' %s, 'figures', '%s_eSSS_spectra' %s))

# save eSSS processed data
esss.save(op.join(path, 'VML', '%s' %s, '%s_rest_tsss.fif' %s))
