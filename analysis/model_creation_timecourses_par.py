# -*- coding: utf-8 -*-
"""Parallelisation function for crt_prf_tcmdl."""

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
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from utilities import crt_gauss


def prf_par(aryMdlParamsChnk, tplVslSpcSze, varNumVol, aryPixConv, queOut):
    """
    Create pRF time course models.

    Parameters
    ----------
    aryMdlParamsChnk : np.array
        2D numpy array containing the parameters for the pRF models to be
        created. Dimensionality: `aryMdlParamsChnk[model-ID, parameter-value]`.
        For each model there are five values: (0) an index starting from zero,
        (1) an additional stimulus feature (e.g. motion direction), (2) the
        x-position, (3) the y-position, and (4) the standard deviation.
        Parameters 2, 3, and 4 are in units of visual space.
    tplVslSpcSze : tuple
        Pixel size of visual space model in which the pRF models are created
        (x- and y-dimension).
    varNumVol : int
        Number of time points (volumes).
    aryPixConv : np.array
        4D numpy array containing HRF-convolved pixel-wise design matrix, with
        shape `aryPixConv[feature, x-position, y-position, time]`.
    queOut : multiprocessing.queues.Queue
        Queue to put the results on.

    Returns
    -------
    lstOut : list
        List containing the following object:
        aryOut : np.array
            2D numpy array, where each row corresponds to one model time
            course, the first column corresponds to the index number of the
            model time course, and the remaining columns correspond to time
            points).

    Notes
    -----
    The list with results is not returned directly, but placed on a
    multiprocessing queue.
    """
    # Number of combinations of model parameters in the current chunk:
    varChnkSze = np.size(aryMdlParamsChnk, axis=0)

    # Number of features (e.g. motion directions):
    # varNumFtr = aryPixConv.shape[0]

    # Output array with pRF model time courses:
    aryOut = np.zeros([varChnkSze, varNumVol], dtype=np.float32)

    # Loop through combinations of model parameters:
    for idxMdl in range(varChnkSze):

        # Feature index of current model:
        varTmpFtr = int(aryMdlParamsChnk[idxMdl, 1])

        # Depending on the relation between the number of x- and y-positions
        # at which to create pRF models and the size of the super-sampled
        # visual space, the indicies need to be rounded:
        varTmpX = np.around(aryMdlParamsChnk[idxMdl, 2], 0)
        varTmpY = np.around(aryMdlParamsChnk[idxMdl, 3], 0)
        varTmpSd = np.around(aryMdlParamsChnk[idxMdl, 4], 0)

        # Create pRF model (2D):
        aryGauss = crt_gauss(tplVslSpcSze[0],
                             tplVslSpcSze[1],
                             varTmpX,
                             varTmpY,
                             varTmpSd).astype(np.float32)

        # Multiply super-sampled pixel-time courses with Gaussian pRF models:
        aryPrfTcTmp = np.multiply(aryPixConv[varTmpFtr, :, :, :],
                                  aryGauss[:, :, None])

        # Calculate sum across x- and y-dimensions - the 'area under the
        # Gaussian surface'. This is essentially an unscaled version of the pRF
        # time course model (i.e. not yet scaled for the size of the pRF).
        aryPrfTcTmp = np.sum(aryPrfTcTmp, axis=(0, 1), dtype=np.float32)

        # Normalise the pRF time course model to the size of the pRF. This
        # gives us the ratio of 'activation' of the pRF at each time point, or,
        # in other words, the pRF time course model. REMOVED - normalisation
        # has been moved to funcGauss(); pRF models are normalised when to have
        # an area under the curve of one when they are created.
        # aryPrfTcTmp = np.divide(aryPrfTcTmp,
        #                         np.sum(aryGauss, axis=(0, 1)))

        # Put model time courses into the function's output array:
        aryOut[idxMdl, :] = aryPrfTcTmp

    # Put column with the indicies of model-parameter-combinations into the
    # output array (in order to be able to put the pRF model time courses into
    # the correct order after the parallelised function):
    aryOut = np.hstack((np.array(aryMdlParamsChnk[:, 0],
                                 ndmin=2,
                                 dtype=np.float32).T,
                        aryOut)).astype(np.float32)

    # Put output to queue:
    queOut.put(aryOut)
