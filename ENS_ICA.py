#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 10:26:24 2023

@author: maggie

Use ICA to find blink and eye movement components & suppress from the data
"""

import mne
import os.path as op
from mne.preprocessing import ICA

####### set these  before running #############################################
path = '/home/maggie/data/ENS/'

# setup subject name ### change this for each subject
s = 'p007'
###############################################################################

# read in processed data
filename =  op.join(path, '%s' %s, '%s_att_all_tsss.fif' %s)
tsss = mne.io.Raw(filename, preload=True)

# remove low freqeuncy drifts before performing ICA by apply a highpass filter  
# these drifts can affect the accuracy of the ICA algorithm
filt_tsss = tsss.copy().filter(l_freq=1.0, h_freq=None)

# fit ICA - first 15 is probably good enough to captur blinks/heart as we'd
# expect these to be pretty strong
ica = ICA(n_components=15, max_iter="auto", random_state=97)
ica.fit(filt_tsss)
ica

# Now lets automatically detect bad components
# define filter params for EOG detection
l_freq=1
h_freq=10

ica.exclude = []
# find which ICs match the EOG pattern
eog_indices, eog_scores = ica.find_bads_eog(tsss, l_freq=l_freq,
                                            h_freq=h_freq)
ica.exclude = eog_indices

# barplot of ICA components - ICs that match EOG will be shown in RED
fig = ica.plot_scores(eog_scores)
fig.savefig(op.join(path, '%s' %s, 'figures', 
                    '%s_ICscores.png' %s))

# plot time series of all the independant components 
# chosen components will be shown greyed out
# you can see the chosen components resemble blinks & eye movements
ica.plot_sources(tsss, show_scrollbars=False)

# plot spatial patterns of ICs
# Again, the second and third component are blinks & eye movements
fig = ica.plot_components()
fig.savefig(op.join(path, '%s' %s, 'figures', '%s_ICs.png' %s))

# Here we plot the data before and after excluding the 2 bad components 
fig = ica.plot_overlay(tsss, exclude=eog_indices)
fig.savefig(op.join(path, '%s' %s, 'figures',
                    '%s_ICAoverlay.png' %s))

# apply ICA once satisfied
tsss_ica = ica.apply(tsss, exclude=eog_indices)

# save cleaned data
tsss_ica.save(op.join(path, '%s' %s,  '%s_att_tsss_ica.fif' %s))
