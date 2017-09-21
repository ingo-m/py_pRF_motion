# -*- coding: utf-8 -*-
"""Put functional data on queue."""

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

import h5py
import numpy as np


def put_func(strFuncHdf, varNumChnk, varVoxPerChnk, queFunc):
    """A."""
    # -------------------------------------------------------------------------
    # *** Preparations

    # Functional data on disk (in hdf5 format) is supposed to have following
    # shape: aryFunc[voxelCount, time].

    # Load hdf5 file:
    fleFunc = h5py.File(strFuncHdf, 'r')

    # Get reference to dataset in hdf5 file:
    dtsFunc = fleFunc['aryFunc']

    # Number voxels:
    # varNumVox = dtsFunc.shape[0]

    # Number of volumes (time points):
    varNumVol = dtsFunc.shape[1]
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # *** Put data on queue

    # Index of first voxel to be included in current chunk:
    varChnkStr = 0

    # Index of last voxel to be included in current chunk:
    varChnkEnd = varVoxPerChnk

    # Loop through chunks and put functional data on queue:
    for idxChnk in range(varNumChnk):

        # All but the last chunks are handled the same way:
        if idxChnk < (varNumChnk - 1):

            # Get data from hdf5 file, and transpose, so that the shape becomes
            # aryFuncTmp[time, voxel].
            aryFuncTmp = dtsFunc[varChnkStr:varChnkEnd, :].T

        else:

            # Get remaining voxel time courses:
            aryRmne = dtsFunc[varChnkStr:, :]

            # Number of remaining voxels:
            varRmne = aryRmne.shape[0]

            # For the last chunk, the data need to be zero-padded in order for
            # the chunk to have the same size as all previous chunks:
            aryFuncTmp = np.zeros(varVoxPerChnk, varNumVol, dtype=np.float32)

            aryFuncTmp[0:varRmne, :] = aryRmne

            aryFuncTmp = aryFuncTmp.T

        # Put time courses on queue:
        queFunc.put(aryFuncTmp, True)

        # Update indices:
        varChnkStr += varVoxPerChnk
        varChnkEnd += varVoxPerChnk
    # -------------------------------------------------------------------------
