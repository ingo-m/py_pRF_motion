# -*- coding: utf-8 -*-
"""Append additional stimulus features  to design matrix."""

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
import pickle


def append_features(aryPngData, lstDsgn):
    """
    Append additional stimulus features to design matrix.

    Parameters
    ----------
    aryPngData : np.array
        3D Numpy array containing pixel-wise design matrix, at np.int8
        precision, with the following structure:
        aryPngData[x-position, y-position, time]
    lstDsgn : list
        List containing paths of pickles files with information about stimulus
        features.

    Returns
    -------
    aryPngDataFtr : np.array
        4D Numpy array containing pixel-wise design matrix, at np.int8
        precision, with the following structure:
        aryPngDataFtr[feature, x-position, y-position, time]

    Notes
    -----
    Append additional stimulus features (e.g. motion direction) to pixel-wise
    design matrix .
    """
    # Ensure data has int8 precision:
    aryPngData = aryPngData.astype(np.int8)

    # List for arrays with design matrix of additional stimulus features:
    lstFtr = []

    # Load design matrices with additional stimulus features (e.g. motion
    # direction within aperture) from pickle files:
    for strTmp in lstDsgn:
        with open(strTmp, 'r') as objPckl:
            # The second column of the respective array holds information on
            # motion directions:
            lstFtr.append(pickle.load(objPckl)['Conditions'][:, 1])

    # Concatenate motion direction arrays from all runs:
    aryFtr = np.hstack(lstFtr)

    # It is assumed that features are coded in integer format:
    aryFtr = aryFtr.astype(np.int32)

    # Get number of non-zero unique values in feature-design-matrix (non-zeros
    # because zero represents the absence of a sitmulus).
    varNumFtr = np.nonzero(np.unique(aryFtr))[0].shape[0]

    # Array for pixel-wise design matrices with feature dimension. Shape:
    # aryPngDataFtr[feature, x-position, y-position, time]
    aryPngDataFtr = np.zeros((varNumFtr,
                              aryPngData.shape[0],
                              aryPngData.shape[1],
                              aryPngData.shape[2]), dtype=np.int8)

    # Loop through parameters and create pixelwise timecourses for all
    # combinations of features. Range starts at `1`, because `0` codes for
    # absence of stimulus.
    for idxFtr in range(1, (varNumFtr + 1)):

        # Logical array for occurence of current feature ('True' if the
        # feature was present at that time point). Dimensions:
        # vecFtr[volume]
        vecFtr = np.equal(aryFtr, idxFtr).astype(np.int8)

        # Temporary array: 'one' if the current feature was
        # present at the respective x, y, and time position/point.
        aryTmp = np.multiply(aryPngData,
                             vecFtr[None, None, :],
                             dtype=np.int8)

        # Put information abotu current feature, x, and y position
        # into design matrix:
        aryPngDataFtr[(idxFtr - 1), :, :, :] = aryTmp

    return aryPngDataFtr
