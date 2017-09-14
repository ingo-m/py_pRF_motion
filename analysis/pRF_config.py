# -*- coding: utf-8 -*-
"""Define pRF finding parameters here."""

# Part of py_pRF_motion library
# Copyright (C) 2016  Marian Schneider, Ingo Marquardt
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


# Number of x-positions to model:
varNumX = 10
# Number of y-positions to model:
varNumY = 10
# Number of pRF sizes to model:
varNumPrfSizes = 10

# Extend of visual space from centre of the screen (i.e. from the fixation
# point) [degrees of visual angle]:
varExtXmin = -5.19
varExtXmax = 5.19
varExtYmin = -5.19
varExtYmax = 5.19

# Maximum and minimum pRF model size (standard deviation of 2D Gaussian)
# [degrees of visual angle]:
varPrfStdMin = 0.1
varPrfStdMax = 10.0

# Volume TR of input data [s]:
varTr = 2.832

# Voxel resolution of the fMRI data [mm]:
varVoxRes = 0.4

# Extent of temporal smoothing that has been applied to fMRI data
# [standard deviation of the Gaussian kernel, in seconds]:
# the same smotthing will be applied to pRF time course models
# [set 0 if not desired]
varSdSmthTmp = 0

# Number of processes to run in parallel:
varPar = 11

# L2 regularisation factor for ridge regression
varL2reg = 0.0

# Size of high-resolution visual space model in which the pRF models are
# created (x- and y-dimension). The x and y dimensions specified here need to
# be the same integer multiple of the number of x- and y-positions to model, as
# specified above. In other words, if the the resolution in x-direction of the
# visual space model is ten times that of varNumX, the resolution in
# y-direction also has to be ten times varNumY. The order is: first x, then y.
tplVslSpcHighSze = (200, 200)

# Parent path to functional data
strPathNiiFunc = '/home/john/Documents/20161221/func_regAcrssRuns_cube_up'
# list of nii files in parent directory (all nii files together need to have
# same number of volumes as there are PNGs):
lstNiiFls = ['func_07_up_aniso_smth.nii',
             'func_08_up_aniso_smth.nii',
             'func_09_up_aniso_smth.nii',
             'func_10_up_aniso_smth.nii',
             ]
# which run should be hold out for testing? [python index strating from 0,
# or None if no testing run should be set aside]
varTestRun = None

# Path of mask (to restrict pRF model finding):
strPathNiiMask = '/home/john/Documents/20161221/retinotopy/mask/tmp_gm_evc.nii.gz'

# Output basename:
strPathOut = '/home/john/Documents/20161221/retinotopy/pRF_results_motion/pRF_results'

# Which version to use for pRF finding. 'numpy' or 'cython' for pRF finding on
# CPU, 'gpu' for using GPU.
strVersion = 'gpu'

# Create pRF time course models?
lgcCrteMdl = True

# reduce presented motion direction from 8 to 4?
lgcAoM = False

# length of the runs that were done
vecRunLngth = [172] * len(lstNiiFls)

# Number of fMRI volumes and png files to load:
varNumVol = sum(vecRunLngth)

# cross validate?
lgcXval = False

# set which set of hrf functions should be used
lgcOldSchoolHrf = True

if lgcOldSchoolHrf:  # use legacy hrf function
    strBasis = '_oldSch'
    # use only canonical hrf function
    switchHrfSet = 1
else:  # use hrf basis
    # decide of how many functions the basis set should consist:
    # 1: canonical hrf function
    # 2: canonical hrf function and 1st tmp derivative
    # 3: canonical hrf function, 1st tmp and spatial derivative
    switchHrfSet = 3
    strBasis = '_bsSet' + str(switchHrfSet)

if lgcXval:
    varNumXval = len(lstNiiFls)  # set nr of xvalidations, equal to nr of runs

# For time course model creation, the following parameters have to
# be provided:

# visual stimuli that were used for this run (if everything is well 1,2,3, asf)
vecVslStim = [1, 2, 3, 4]

# Basename of the filenames that have the presentation orders saved
strPathPresOrd = '/home/john/Documents/20161221/retinotopy/design_matrix/Conditions_run0'

# Sample PNGs at this resolution (pixel*pixel):
tplPngSize = tplVslSpcHighSze

# Basename of the 'binary stimulus files'. The files need to be in png
# format and number in the order of their presentation during the
# experiment.
strPathPng = '/home/john/Documents/20161221/retinotopy/pRF_stimuli/frame_'

# If we use existing pRF time course models, the path to the respective
# file has to be provided (including file extension, i.e. '*.npy'):
strPathMdl = '/home/john/Documents/20161221/retinotopy/design_matrix/pRF_timecourses.npy'

# reduce presented motion direction from 8 to 4?
lgcAoM = False

if lgcAoM:
    # number of motion directions
    varNumMtDrctn = 5 * switchHrfSet
else:
    # number of motion directions
    varNumMtDrctn = 9 * switchHrfSet
