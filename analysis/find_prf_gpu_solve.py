# -*- coding: utf-8 -*-
"""Main function for pRF finding."""

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

import os
import numpy as np
import threading
import tensorflow as tf


def find_prf_gpu(varNumMdls, varNumChnk, varVoxPerChnk, varNumVol, varNumBeta,
                 varL2reg, queDsgn, queFunc, queRes):
    """
    Find best pRF model for voxel time course.

    Parameters
    ----------

    ...

    varL2reg : float
        L2 regularisation factor for ridge regression.
    queOut : multiprocessing.queues.Queue
        Queue to put the results on.

    Returns
    -------

    ...

    Notes
    -----

    ...

    """
    # -------------------------------------------------------------------------
    # *** Queue-feeding-functions

    def funcPlcDsgn():
        """Place pRF model time courses on queue."""
        # Iteration counter:
        idxCnt = 0

        while True:

            # Get design matrix from queue:
            aryDsgnTmp = queDsgn.get(True)

            # Feed design matrix Tensorflow placeholder
            dicDsgnIn = {objPlcHldDsgn: aryDsgnTmp}

            # Push to the queue:
            objSess.run(objDsgnEnQ, feed_dict=dicDsgnIn)

            idxCnt += 1

            # Stop if coordinator says stop:
            if objCoord.should_stop():
                break

            # Stop if all data has been put on the queue:
            elif idxCnt == varNumIt:
                break

    def funcPlcFunc():
        """Place functional data on queue."""
        # Iteration counter:
        idxCnt = 0

        while True:

            # Get design matrix from queue:
            aryFuncTmp = queFunc.get(True)

            # Feed design matrix Tensorflow placeholder
            dicFuncIn = {objPlcHldFunc: aryFuncTmp}

            # Push to the queue:
            objSess.run(objFuncEnQ, feed_dict=dicFuncIn)

            idxCnt += 1

            # Stop if coordinator says stop:
            if objCoord.should_stop():
                break

            # Stop if all data has been put on the queue:
            elif idxCnt == varNumIt:
                break

    # -------------------------------------------------------------------------
    # *** Miscellaneous preparations

    # Total number of iterations (number of models * number of chunks of
    # functional data):
    varNumIt = varNumMdls * varNumChnk

    # Multiply L2 regularization factor with identity matrix:
    aryL2reg = np.multiply(np.eye(varNumBeta),
                           varL2reg).astype(np.float32)

    # Reduce logging verbosity:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # -------------------------------------------------------------------------
    # *** Prepare status indicator

    # We create a status indicator for the time consuming pRF model finding
    # algorithm. Number of steps of the status indicator:
    varStsStpSze = 20

    # Vector with pRF values at which to give status feedback:
    vecStatPrf = np.linspace(0,
                             varNumIt,
                             num=(varStsStpSze+1),
                             endpoint=True)
    vecStatPrf = np.ceil(vecStatPrf)
    vecStatPrf = vecStatPrf.astype(int)

    # Vector with corresponding percentage values at which to give status
    # feedback:
    vecStatPrc = np.linspace(0,
                             100,
                             num=(varStsStpSze+1),
                             endpoint=True)
    vecStatPrc = np.ceil(vecStatPrc)
    vecStatPrc = vecStatPrc.astype(int)

    # Counter for status indicator:
    varCntSts01 = 0
    varCntSts02 = 0

    # -------------------------------------------------------------------------
    # *** Loop through chunks

    print('------Run graph')

    # Define session:
    # objSess = tf.Session()
    with tf.Graph().as_default(), tf.Session() as objSess:

        # -----------------------------------------------------------------
        # *** Prepare queue for pRF time courses

        print('------Prepare queue for pRF time courses')

        # Queue capacity:
        varCapDsgnQ = 10

        # The queue:
        objDsgnQ = tf.FIFOQueue(capacity=varCapDsgnQ,
                                dtypes=[tf.float32],
                                shapes=[(varNumVol, varNumBeta)])

        # Method for getting queue size:
        objSzeDsgnQ = objDsgnQ.size()

        # Placeholder that is used to put design matrix on computational
        # graph:
        objPlcHldDsgn = tf.placeholder(tf.float32,
                                       shape=[varNumVol, varNumBeta])

        # The enqueue operation that puts data on the graph.
        objDsgnEnQ = objDsgnQ.enqueue([objPlcHldDsgn])

        # Number of threads that will be created:
        varNumThrd = 1

        # The queue runner (places the enqueue operation on the queue?).
        objRunDsgnQ = tf.train.QueueRunner(objDsgnQ, [objDsgnEnQ] * varNumThrd)
        tf.train.add_queue_runner(objRunDsgnQ)

        # The tensor object that is retrieved from the queue. Functions like
        # placeholders for the data in the queue when defining the graph.
        objDsgn = objDsgnQ.dequeue()

        # -----------------------------------------------------------------
        # *** Prepare queue for functional data

        print('------Prepare queue for functional data')

        # Queue capacity:
        varCapFuncQ = 10

        # The queue:
        objFuncQ = tf.FIFOQueue(capacity=varCapFuncQ,
                                dtypes=[tf.float32],
                                shapes=[(varNumVol, varVoxPerChnk)])

        # Method for getting queue size:
        # objSzeFuncQ = objFuncQ.size()

        # Placeholder that is used to put design matrix on computational
        # graph:
        objPlcHldFunc = tf.placeholder(tf.float32,
                                       shape=[varNumVol, varVoxPerChnk])

        # The enqueue operation that puts data on the graph.
        objFuncEnQ = objFuncQ.enqueue([objPlcHldFunc])

        # Number of threads that will be created:
        varNumThrd = 1

        # The queue runner (places the enqueue operation on the queue?).
        objRunFuncQ = tf.train.QueueRunner(objFuncQ, [objFuncEnQ] * varNumThrd)
        tf.train.add_queue_runner(objRunFuncQ)

        # The tensor object that is retrieved from the queue. Functions like
        # placeholders for the data in the queue when defining the graph.
        objFunc = objFuncQ.dequeue()

        # -----------------------------------------------------------------
        # *** Fill queue

        # Coordinator needs to be initialised:
        objCoord = tf.train.Coordinator()

        # Buffer size (number of samples to put on queue before starting
        # execution of graph):
        # varBuff = 10

        # Define & run extra thread with graph that places pRF time courses on
        # queue:
        objThrdDsgn = threading.Thread(target=funcPlcDsgn)
        objThrdDsgn.setDaemon(True)
        objThrdDsgn.start()

        # Define & run extra thread with graph that places functional data on
        # queue:
        objThrdFunc = threading.Thread(target=funcPlcFunc)
        objThrdFunc.setDaemon(True)
        objThrdFunc.start()

        # Stay in this while loop until the specified number of samples
        # (varBuffer) have been placed on the queue).
        # varTmpSzeQ = 0
        # while varTmpSzeQ < varBuff:
        #     varTmpSzeQ = min([objSess.run(objSzeDsgnQ),
        #                       objSess.run(objSzeFuncQ)])

        # -----------------------------------------------------------------
        # *** Prepare & run the graph

        # Regularisation factor matrix:
        with tf.device('/gpu:0'):
            objL2reg = tf.Variable(aryL2reg)

        # The computational graph. Operation that solves matrix (in the
        # least squares sense), and calculates residuals along time
        # dimension. There are two versions: (1) The number of measurements
        # (e.g. volumes) is greater than or equal to the number of
        # predictors (betas). (2) The number of measurements is less than
        # the number of predictors.

        # (1) Number of measurements greater/equal to number of predictors:
        if np.greater_equal(varNumVol, varNumBeta):
            objMatSlve = tf.reduce_sum(
                                       tf.squared_difference(
                                                             objFunc,
                                                             tf.matmul(
                                                                       objDsgn,
                                                                       tf.matmul(
                                                                                 tf.matmul(
                                                                                           tf.matrix_inverse(
                                                                                                             tf.add(
                                                                                                                    tf.matmul(
                                                                                                                               objDsgn,
                                                                                                                               objDsgn,
                                                                                                                               transpose_a=True,
                                                                                                                               transpose_b=False
                                                                                                                               ),
                                                                                                                    objL2reg
                                                                                                                    )
                                                                                                             ),
                                                                                           objDsgn,
                                                                                           transpose_a=False,
                                                                                           transpose_b=True
                                                                                           ),
                                                                                 objFunc
                                                                                 )
                                                                       ),
                                                             ),
                                       axis=0
                                       )

        # (2) Number of measurements less than number of predictors:
        else:
            objMatSlve = tf.reduce_sum(
                                       tf.squared_difference(
                                                             objFunc,
                                                             tf.matmul(
                                                                       objDsgn,
                                                                       tf.matmul(
                                                                                 tf.matmul(
                                                                                           objDsgn,
                                                                                           tf.matrix_inverse(
                                                                                                             tf.add(
                                                                                                                    tf.matmul(
                                                                                                                               objDsgn,
                                                                                                                               objDsgn,
                                                                                                                               transpose_a=False,
                                                                                                                               transpose_b=True
                                                                                                                               ),
                                                                                                                    objL2reg
                                                                                                                    )
                                                                                                             ),
                                                                                           transpose_a=True,
                                                                                           transpose_b=False
                                                                                           ),
                                                                                 objFunc
                                                                                 )
                                                                       ),
                                                             ),
                                       axis=0
                                       )

        # Variables need to be initialised:
        objSess.run(tf.global_variables_initializer())

        # Mark graph as read-only (would throw an error in case of memory
        # leak):
        objSess.graph.finalize()

        # Loop through chunks of functional data:
        for idxChnk in range(varNumChnk):

            # Loop through models:
            for idxMdl in range(varNumMdls):

                # varTme01 = time.time()

                # Run main computational graph and put results on queue:
                queRes.put(objSess.run(objMatSlve), True)

                # print(('---------Time for graph call: '
                #        + str(time.time() - varTme01)))

                # Status indicator:
                if varCntSts02 == vecStatPrf[varCntSts01]:
                    # Number of elements on queue:
                    varTmpSzeQ = objSess.run(objSzeDsgnQ)
                    # Prepare status message:
                    strStsMsg = ('---------Progress: '
                                 + str(vecStatPrc[varCntSts01])
                                 + ' % --- Number of elements on queue: '
                                 + str(varTmpSzeQ))
                    print(strStsMsg)
                    # Only increment counter if the last value has not been
                    # reached yet:
                    if varCntSts01 < varStsStpSze:
                        varCntSts01 = varCntSts01 + int(1)
                # Increment status indicator counter:
                varCntSts02 = varCntSts02 + 1

        # Stop threads.
        objCoord.request_stop()
        # objSess.close()
