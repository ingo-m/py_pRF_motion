# -*- coding: utf-8 -*-
"""Find best fitting model time courses for population receptive fields.

Use `import pRF_config as cfg` for static pRF analysis.

Use `import pRF_config_motion as cfg` for pRF analysis with motion stimuli.
"""


# Part of py_pRF_mapping library
# Copyright (C) 2016  Ingo Marquardt
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.


# *****************************************************************************
# *** Import modules

import config as cfg

import time
import numpy as np
import h5py

from model_creation_main import model_creation
from preprocessing_main import pre_pro_models
from preprocessing_main import pre_pro_func

from find_prf_gpu_motion_main import find_prf_gpu

# *****************************************************************************


# *****************************************************************************
# *** Check time
print('---pRF analysis')
varTme01 = time.time()
# *****************************************************************************


# *****************************************************************************
# *** Preparations

# Convert preprocessing parameters (for temporal and spatial smoothing) from
# SI units (i.e. [s] and [mm]) into units of data array (volumes and voxels):
cfg.varSdSmthTmp = np.divide(cfg.varSdSmthTmp, cfg.varTr)
cfg.varSdSmthSpt = np.divide(cfg.varSdSmthSpt, cfg.varVoxRes)
# *****************************************************************************


# *****************************************************************************
# *** Create or load pRF time course models

aryPrfTc = model_creation()
# *****************************************************************************


# *****************************************************************************
# *** Preprocessing

# Preprocessing of pRF model time courses

# Number of features (e.g. motion directions):
varNumFtr = aryPrfTc.shape[0]

# Loop through features:
for idxFtr in range(varNumFtr):
    aryPrfTc[idxFtr, :] = pre_pro_models(aryPrfTc[idxFtr, :],
                                         varSdSmthTmp=cfg.varSdSmthTmp,
                                         varPar=cfg.varPar)

# Change order of axes in order to fit with GPU function:
aryPrfTc = np.moveaxis(aryPrfTc,
                       [0, 1, 2, 3, 4],
                       [3, 0, 1, 2, 4])

# Preprocessing of functional data:
aryLgcMsk, hdrMsk, aryAff, aryLgcVar, aryFunc, tplNiiShp = pre_pro_func(
    cfg.strPathNiiMask, cfg.lstPathNiiFunc, lgcLinTrnd=cfg.lgcLinTrnd,
    varSdSmthTmp=cfg.varSdSmthTmp, varSdSmthSpt=cfg.varSdSmthSpt,
    varPar=cfg.varPar)
# *****************************************************************************


# *****************************************************************************
# *** Store data in hdf5 format

print('---------Save pRF model time courses to disk (hdf5 format)')

# Path for design matrix hdf5 file:
strPthDsng = cfg.strDirHdf + 'aryPrfTc.hdf5'

# Create hdf5 file:
flePrfTc = h5py.File(strPthDsng, 'w')

# Save data:
dtsPrfTc = flePrfTc.create_dataset('aryPrfTc',
                                   data=aryPrfTc.asytpe(np.float32),
                                   dtype=np.float32)

flePrfTc.close()

del(aryPrfTc)

print('---------Save preprocessed functional data to disk (hdf5 format)')

# At this point, the functional data has the following shape:
# aryFunc[voxelCount, time].

# Path for design matrix hdf5 file:
strPthFnc = cfg.strDirHdf + 'aryFunc.hdf5'

# Create hdf5 file:
fleFnc = h5py.File(strPthFnc, 'w')

# Save data:
dtsPrfTc = fleFnc.create_dataset('aryFunc',
                                 data=aryFunc.asytpe(np.float32),
                                 dtype=np.float32)

fleFnc.close()

del(aryFunc)
# *****************************************************************************


# *****************************************************************************
# *** Find pRF models for voxel time courses

find_prf_gpu()
# *****************************************************************************


# *****************************************************************************
# *** Report time

varTme02 = time.time()
varTme03 = varTme02 - varTme01
print('---Elapsed time: ' + str(varTme03) + ' s')
print('---Done.')
# *****************************************************************************
