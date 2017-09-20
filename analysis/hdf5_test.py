#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Test of h5py for creation of hdf5 files.
"""

import numpy as np
import h5py

# Create hdf5 file:
fleDsgn = h5py.File('/home/john/Documents/20161221/retinotopy/hdf5/design_matrix.hdf5', 'w')

# Create dataset within hdf5 file:
dtsDsgn = fleDsgn.create_dataset('design_matrix',
                                 (1000,
                                  1000,
                                  1000),
                                 dtype=np.float32)

# Counter:
idxCnt = 0

for idx01 in range(1000):

    # Place random data on dataset:
    dtsDsgn[idx01, :, :] = np.random.randn(1000, 1000).astype(np.float32)

    print(idxCnt)
    idxCnt += 1

# Close file:
fleDsgn.close()
