# -*- coding: utf-8 -*-
"""pRF model creation."""

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

import numpy as np
import nibabel as nb
from model_creation_load_png import load_png
from model_creation_pixelwise import conv_dsgn_mat
from model_creation_timecourses import crt_prf_tcmdl
from model_creation_features import append_features

import config as cfg


def model_creation():
    """
    Create or load pRF model time courses.

    Parameters
    ----------
    Parameters for pRF model creation are imported from config.py file.

    Returns
    -------
    aryPrfTc : np.array
        4D numpy array with pRF time course models, with following dimensions:
        `aryPrfTc[x-position, y-position, SD, volume]`.
    """
    if cfg.lgcCrteMdl:  #noqa

        # *********************************************************************
        # *** Load stimulus information from PNG files

        print('------Load stimulus information from PNG files')

        aryPngData = load_png(cfg.varNumVol,
                              cfg.strPathPng,
                              cfg.tplVslSpcSze,
                              varStrtIdx=cfg.varStrtIdx,
                              varZfill=cfg.varZfill)
        # *********************************************************************

        # *********************************************************************
        # *** Append additional stimulus features (e.g. motion direction)

        print('------Append additional stimulus features')

        # Additional stimulus feature dimension is added to the pixel-wise
        # design matrix, now of shape aryPngDataFtr[feature, x-position,
        # y-position, time] at int8 precision.
        aryPngData = append_features(aryPngData,
                                     cfg.lstDsgn)
        # *********************************************************************

        # *********************************************************************
        # *** Convolve pixel-wise design matrix with HRF model

        print('------Convolve pixel-wise design matrix with HRF model')

        aryPixConv = conv_dsgn_mat(aryPngData,
                                   cfg.varTr,
                                   cfg.varPar)

        del(aryPngData)

        # Debugging feature:
        # np.save('/home/john/Desktop/aryPixConv.npy', aryPixConv)
        # *********************************************************************

        # *********************************************************************
        # *** Create pRF time courses models

        print('------Create pRF time course models')

        aryPrfTc = crt_prf_tcmdl(aryPixConv,
                                 cfg.strDirHdf,
                                 tplVslSpcSze=cfg.tplVslSpcSze,
                                 varNumX=cfg.varNumX,
                                 varNumY=cfg.varNumY,
                                 varExtXmin=cfg.varExtXmin,
                                 varExtXmax=cfg.varExtXmax,
                                 varExtYmin=cfg.varExtYmin,
                                 varExtYmax=cfg.varExtYmax,
                                 varPrfStdMin=cfg.varPrfStdMin,
                                 varPrfStdMax=cfg.varPrfStdMax,
                                 varNumPrfSizes=cfg.varNumPrfSizes,
                                 varPar=cfg.varPar)
        # *********************************************************************

        # *********************************************************************
        # *** Save pRF time course models

        print('------Save pRF time course models to disk')

        # Save the 4D array as '*.npy' file:
        np.save(cfg.strPathMdl,
                aryPrfTc)

        # Save 4D array as '*.nii' file (for debugging purposes):
        niiPrfTc = nb.Nifti1Image(np.max(aryPrfTc, axis=0), np.eye(4))
        nb.save(niiPrfTc, cfg.strPathMdl)
        # *********************************************************************

    else:

        # *********************************************************************
        # *** Load existing pRF time course models

        print('------Load pRF time course models from disk')

        # Load the file:
        aryPrfTc = np.load((cfg.strPathMdl + '.npy'))

        # Check whether pRF time course model matrix has the expected
        # dimensions:
        vecPrfTcShp = aryPrfTc.shape

        # Logical test for correct dimensions:
        lgcDim = ((vecPrfTcShp[1] == cfg.varNumX)
                  and
                  (vecPrfTcShp[2] == cfg.varNumY)
                  and
                  (vecPrfTcShp[3] == cfg.varNumPrfSizes)
                  and
                  (vecPrfTcShp[4] == cfg.varNumVol))

        # Only fit pRF models if dimensions of pRF time course models are
        # correct:
        if not(lgcDim):
            # Error message:
            strErrMsg = ('---Error: Dimensions of specified pRF time course ' +
                         'models do not agree with specified model parameters')
            raise ValueError(strErrMsg)
    # *************************************************************************

    return aryPrfTc
