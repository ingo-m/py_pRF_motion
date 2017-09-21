# -*- coding: utf-8 -*-
"""Find best fitting model time courses for population receptive fields."""


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

import os
import time
import h5py
import numpy as np
import multiprocessing as mp

from model_creation_main import model_creation
from preprocessing_main import pre_pro_models
from preprocessing_main import pre_pro_func

from find_prf_gpu_put_func import put_func
from find_prf_gpu_put_design import put_design
from find_prf_gpu_solve import find_prf_gpu
from find_prf_gpu_get_res import get_res
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
# ***  Preprocessing of pRF model time courses

# Number of features (e.g. motion directions):
varNumFtr = aryPrfTc.shape[0]

# Loop through features:
for idxFtr in range(varNumFtr):
    aryPrfTc[idxFtr, :] = pre_pro_models(aryPrfTc[idxFtr, :],
                                         varSdSmthTmp=cfg.varSdSmthTmp,
                                         varPar=cfg.varPar)

# Change order of axes in order to fit with GPU function, from
# aryPrfTc[feature, x-position, y-position, SD, time] to
# aryPrfTc[x-position, y-position, SD, time, feature].
aryPrfTc = np.moveaxis(aryPrfTc,
                       [0, 1, 2, 3, 4],
                       [4, 0, 1, 2, 3])

# At this point, the pRF model time course have been z-scored. In order to
# avoid precision problems during GLM fitting, we scale them up.
aryPrfTc = np.multiply(aryPrfTc,
                       1000.0).astype(np.float32)

# Reshape pRF model time courses:
aryPrfTc = np.reshape(aryPrfTc,
                      ((aryPrfTc.shape[0]
                        * aryPrfTc.shape[1]
                        * aryPrfTc.shape[2]),
                       aryPrfTc.shape[3],
                       aryPrfTc.shape[4]))

# Now, aryPrfTc has the following dimensions:
# aryPrfTc[(x-pos * y-pos * SD), time, feature]

# Original total number of pRF time course models (before removing models with
# zero variance):
varNumMdlsTtl = aryPrfTc.shape[0]

# Change type to float 32:
aryPrfTc = aryPrfTc.astype(np.float32)

# The pRF model is fitted only if variance along time dimension is not very
# low. Get variance along time dimension:
vecVarPrfTc = np.var(aryPrfTc, axis=1)

# Low value with float32 precision for comparison:
varLow32 = np.array(([1.0])).astype(np.float32)[0]

# Boolean array for models with variance greater threshold for at least one
# motion direction:
vecLgcVar = np.max(
                   np.greater(vecVarPrfTc,
                              varLow32),
                   axis=1
                   )

del(vecVarPrfTc)

# Take models with variance less than zero out of the array:
aryPrfTc = aryPrfTc[vecLgcVar, :, :]

# Write pRF model time courses to disk (in hdf5 format). Shape of the data:
# aryPrfTc[(x-pos * y-pos * SD), time, feature]. Models with low variance are
# not included.

print('---------Writing pRF model time courses to disk')

# Create hdf5 file:
strDsgnHdf = cfg.strDirHdf + 'aryPrfTc.hdf5'
fleDsgn = h5py.File(strDsgnHdf, 'w')

# Create dataset within hdf5 file:
dtsDsgn = fleDsgn.create_dataset('aryPrfTc',
                                 data=aryPrfTc,
                                 dtype=np.float32)

# Close file:
fleDsgn.close()

del(aryPrfTc)
# *****************************************************************************


# *****************************************************************************
# ***  Preprocessing of functional data


# Preprocessing of functional data:
aryLgcMsk, hdrMsk, aryAff, aryLgcVar, aryFunc, tplNiiShp = pre_pro_func(
    cfg.strPathNiiMask, cfg.lstPathNiiFunc, lgcLinTrnd=cfg.lgcLinTrnd,
    varSdSmthTmp=cfg.varSdSmthTmp, varSdSmthSpt=cfg.varSdSmthSpt,
    varPar=cfg.varPar)

# At this point, the funtional time courses have been z-scored. In order to
# avoid precision problems during GLM fitting, we scale them up.
aryFunc = np.multiply(aryFunc,
                      1000.0).astype(np.float32)

# Number of voxels for which pRF finding will be performed:
varNumVoxInc = aryFunc.shape[0]

print('---------Number of voxels on which pRF finding will be performed: '
      + str(varNumVoxInc))

# Write preprocessed functional data to disk (in hdf5 format). Shape of the
# data: aryFunc[voxelCount, time].

print('---------Writing preprocessed functional data to disk')

# Create hdf5 file:
strFuncHdf = cfg.strDirHdf + 'aryFunc.hdf5'
fleFunc = h5py.File(strFuncHdf, 'w')

# Create dataset within hdf5 file:
dtsFunc = fleFunc.create_dataset('aryFunc',
                                 data=aryFunc,
                                 dtype=np.float32)

# Close file:
fleFunc.close()

del(aryFunc)
# *****************************************************************************


# *****************************************************************************
# *** Decide on number of chunks

# We cannot commit the entire functional data to GPU memory, we need to create
# chunks. Check how many chunks will be created:
varNumChnk = int(
                 np.ceil(
                         np.divide(
                                   float(varNumVoxInc),
                                   float(cfg.varVoxPerChnk)
                                   )
                         )
                 )

print(('---------Functional data will be split into '
       + str(varNumChnk)
       + ' batches'))

# Size of functional data in MB:
varSzeFunc = os.path.getsize(strFuncHdf)

varSzeFunc = np.divide(float(varSzeFunc),
                       1000000.0)

print(('---------Size of functional data: '
       + str(np.around(varSzeFunc))
       + ' MB'))

# Size of each batch:
varSzeChnk = np.divide(varSzeFunc, varNumChnk)

print(('---------Size of one chunk of functional data: '
       + str(np.around(varSzeChnk))
       + ' MB'))
# *****************************************************************************


# *****************************************************************************
# *** Find pRF models for voxel time courses

print('------Find pRF models for voxel time courses')

# Vector with the modeled x-positions of the pRFs:
vecMdlXpos = np.linspace(cfg.varExtXmin,
                         cfg.varExtXmax,
                         cfg.varNumX,
                         endpoint=True)

# Vector with the modeled y-positions of the pRFs:
vecMdlYpos = np.linspace(cfg.varExtYmin,
                         cfg.varExtYmax,
                         cfg.varNumY,
                         endpoint=True)

# Vector with the modeled standard deviations of the pRFs:
vecMdlSd = np.linspace(cfg.varPrfStdMin,
                       cfg.varPrfStdMax,
                       cfg.varNumPrfSizes,
                       endpoint=True)

# Create a queue for functional data:
queFunc = mp.Queue(maxsize=10)

# Create a queue for pRF model time courses:
queDsgn = mp.Queue(maxsize=100)

# Create a queue for results:
queRes = mp.Queue(maxsize=100)

# Start function that loads functional data:
prcPutFunc = mp.Process(target=put_func,
                        args=(strFuncHdf,
                              varNumChnk,
                              cfg.varVoxPerChnk,
                              queFunc)
                        )

# Start function that loads pRF time course models:
prcPutDsgn = mp.Process(target=put_design,
                        args=(strDsgnHdf,
                              varNumChnk,
                              queDsgn)
                        )

# Start function that performs pRF model finding:
prcSolve = mp.Process(target=find_prf_gpu,
                      args=()
                      )

# Start function that collects results:
prcGetRes = mp.Process(target=get_res,
                       args=()
                       )

# Daemon (kills processes when exiting):
prcPutFunc.Daemon = True
prcPutDsgn.Daemon = True
prcSolve.Daemon = True
prcGetRes.Daemon = True

# Start processes:
prcPutFunc.start()
prcPutDsgn.start()
prcSolve.start()
prcGetRes.start()

# Join processes:
prcSolve.join()
prcGetRes.join()
prcPutFunc.join()
prcPutDsgn.join()

# Collect results from queue:
lstPrfRes[idxPrc] = queOut.get(True)
# *****************************************************************************


# *****************************************************************************
# *** Report time

varTme02 = time.time()
varTme03 = varTme02 - varTme01
print('---Elapsed time: ' + str(varTme03) + ' s')
print('---Done.')
# *****************************************************************************
