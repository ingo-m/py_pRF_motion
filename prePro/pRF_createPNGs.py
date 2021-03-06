# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 16:52:56 2016

@author: marian
many elements of this script are taken from
py_28_pRF_finding from Ingo Marquardt
"""
# %%
# *** Import modules
from __future__ import division  # so that 1/3=0.333 instead of 1/3=0
import os
import numpy as np
#import Image
from PIL import Image
import pickle

# %%
# *** Define parameters

# %%
# consider only motion conditions, set static to zero?
lgcStc = False

# Number of presented runs
varNumRuns = 4

# list of volumes per run
varNumTPinSingleRun = np.array([172, 172, 172, 172])

# list of stimuli used for run (varies with individual recording)
lstRunStimuli = np.array([1, 2, 3, 4])

# Determine the factors by which the image should be downsampled (since the
# original array is rather big; factor 2 means:downsample from 1200 to 600):
factorX = 8
factorY = 8

# load aperture positions
strPathApertPos = '/home/john/Desktop/20161220_stimuli/Masks/mskBar.npz'

# Output path for time course files:
strPathOut = '/home/john/Desktop/20161220_stimuli/PNGs/mskBar'
if not os.path.exists(strPathOut):
    os.makedirs(strPathOut)

# %%
# provide parameters for pRF time course creation

# Base name of pickle files that contain order of stim presentat. in each run
# file should contain 1D array, column contains present. order of aperture pos,
# here file is 2D where 2nd column contains present. order of motion directions
strPathPresOrd = '/home/john/Desktop/20161220_stimuli/Conditions/Conditions_run0'


# %%
# *** Load presentation order of conditions
print('------Load presentation order of apert pos and mot dir')
aryPresOrd = np.empty((0, 2))
for idx01 in range(0, varNumRuns):
    # reconstruct file name
    # ---> consider: some runs were shorter than others(replace next row)
    filename1 = (strPathPresOrd + str(lstRunStimuli[idx01]) + '.pickle')
    # filename1 = (strPathPresOrd + str(idx01+1) + '.pickle')
    # load array
    with open(filename1, 'rb') as handle:
        array1 = pickle.load(handle)
    tempCond = array1["Conditions"]
    # ---> consider that some runs were shorter than others (delete next row)
    tempCond = tempCond[:varNumTPinSingleRun[idx01], :]
    # add temp array to aryPresOrd
    aryPresOrd = np.concatenate((aryPresOrd, tempCond), axis=0)
aryPresOrd = aryPresOrd.astype(int)
del(tempCond)

# deduce the number of time points (conditions/volumes) from len aryPresOrd
varNumTP = len(aryPresOrd)

# set static conditions to zero
if lgcStc:
    aryPresOrd[aryPresOrd[:, 1] == 9, :] = 0

# %%
# *** Load aperture information
print('------Load aperture information')
# Load npz file content into list:
with np.load(strPathApertPos) as objMsks:
    lstMsks = objMsks.items()

for objTmp in lstMsks:
    strMsg = '---------Mask type: ' + objTmp[0]
    # The following print statement prints the name of the mask stored in the
    # npz array from which the mask shape is retrieved. Can be used to check
    # whether the correct mask has been retrieved.
    print(strMsg)
    aryApertPos = objTmp[1]

# convert to integer
aryApertPos = aryApertPos.astype(int)

# up- or downsample the image
aryApertPos = aryApertPos[0::factorX, 0::factorY, :]
tplPngSize = aryApertPos.shape[0:2]

# *** Combine information of aperture pos and presentation order
print('------Combine information of aperture pos and presentation order')
aryCond = aryApertPos[:, :, aryPresOrd[:, 0]]

# When presenting the stimuli, psychopy presents the stimuli from the array
# 'upside down', compared to a representation of the array where the first row
# of the first column is in the upper left. In order to get the PNGs to have
# the same orientations as the stimuli on the screen during the experiment, we
# need to flip the array. (See ~/py_pRF_motion/stimuli/Main/prfStim_Motion*.py
# for the experiment script in which this 'flip' occurs.)
aryCond = np.flipud(aryCond)

scaleValue = 255  # value to multipy mask value (1s) with for png format
for index in np.arange(aryCond.shape[2]):
    im = Image.fromarray(scaleValue * aryCond[:, :, index].astype(np.uint8))
    filename = 'Ima_' + str(index) + '.png'
    im.save(os.path.join(strPathOut, filename))
