#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Test of h5py for creation of hdf5 files.
"""

import numpy as np
import h5py

# Create hdf5 file:
fleDsgn = h5py.File('/home/john/Desktop/tmp/my_hdf5.hdf5', 'w')

# Create dataset within hdf5 file:
dtsDsgn = fleDsgn.create_dataset('design_matrix',
                                 data=np.random.randn(1000, 1000,
                                                      ).astype(np.float32),
                                 dtype=np.float32)

## Counter:
#idxCnt = 0
#
#for idx01 in range(1000):
#
#    # Place random data on dataset:
#    dtsDsgn[idx01, :, :] = np.random.randn(1000, 1000).astype(np.float32)
#
#    print(idxCnt)
#    idxCnt += 1

# Close file:
fleDsgn.close()

# Load hdf5 file:
fleDsgn = h5py.File('/home/john/Desktop/tmp/my_hdf5.hdf5', 'r')

dts01 = fleDsgn['design_matrix']

dts01.shape

ary01 = dts01[:, :]
