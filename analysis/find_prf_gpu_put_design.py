# -*- coding: utf-8 -*-
"""Put pRF model time courses on queue."""


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
import numpy as np  #noqa


def put_design(strDsgnHdf, varNumChnk, queDsgn):
    """A."""
    # -------------------------------------------------------------------------
    # *** Preparations

    # pRF model time courses on disk (in hdf5 format) are supposed to have
    # following shape: aryPrfTc[(x-pos * y-pos * SD), time, feature].
    # Models with low variance are not included.

    # Load hdf5 file:
    fleDsng = h5py.File(strDsgnHdf, 'r')

    # Get reference to dataset in hdf5 file:
    dtsDsgn = fleDsng['aryPrfTc']

    # Number of model time courses:
    varNumMdls = dtsDsgn.shape[0]
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # *** Put data on queue

    # Loop through chunks:
    for idxChnk in range(varNumChnk):

        # Loop through models:
        for idxMdl in range(varNumMdls):

            # Get data from hdf5 file. Shape: aryDsgnTmp[time, feature].
            aryDsgnTmp = dtsDsgn[idxMdl, :, :]

            # Put time courses on queue:
            queDsgn.put(aryDsgnTmp, True)
    # -------------------------------------------------------------------------
