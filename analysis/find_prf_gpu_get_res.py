# -*- coding: utf-8 -*-
"""Fetch pRF finding results form queue."""


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

def get_res(varNumChnk, varNumMdls, vecLgcVar, queRes):
    """"A."""
    varCnt = 0

    # Loop through chunks of functional data:
    for idxChnk in range(varNumChnk):

        # Loop through models:
        for idxMdl in range(varNumMdls):

            aryTmp = queRes.get(True)

            if varCnt < 10:
                print('aryTmp.shape')
                print(aryTmp.shape)
                print('aryTmp[0:5]')
                print(aryTmp[0:5])
                varCnt += 1

#    # Array for results of current chunk:
#    aryTmpRes = np.zeros((varNumMdls,
#                          lstFunc[idxChnk].shape[1]),
#                         dtype=np.float32)
#
#
#    # Vector for minimum squared residuals:
#    vecResSsMin = np.zeros((varNumVox), dtype=np.float32)
#
#    # Vector for indices of models with minimum residuals:
#    vecResSsMinIdx = np.zeros((varNumVox), dtype=np.int32)
#
#        # Get indices of models with minimum residuals (minimum along
#        # model-space) for current chunk:
#        vecResSsMinIdx[varChnkStr:varChnkEnd] = np.argmin(aryTmpRes, axis=0)
#        # Get minimum residuals of those models:
#        vecResSsMin[varChnkStr:varChnkEnd] = np.min(aryTmpRes, axis=0)
#
#
#
#    # We delete the original array holding the functional data to conserve
#    # memory. Therefore, we first need to calculate the mean (will be needed
#    # for calculation of R2).
#
#    # After finding the best fitting model for each voxel, we still have to
#    # calculate the coefficient of determination (R-squared) for each voxel. We
#    # start by calculating the total sum of squares (i.e. the deviation of the
#    # data from the mean). The mean of each time course:
#    vecFuncMean = np.mean(aryFunc, axis=0, dtype=np.float32)
#    # Deviation from the mean for each datapoint:
#    vecFuncDev = np.subtract(aryFunc, vecFuncMean[None, :], dtype=np.float32)
#
#    # Sum of squares:
#    vecSsTot = np.sum(np.power(vecFuncDev,
#                               2.0),
#                      axis=0)
#
#
#
#
#
#    # -------------------------------------------------------------------------
#    # *** Post-process results
#
#    print('------Post-processing results')
#
#    # Array for model parameters. At the moment, we have the indices of the
#    # best fitting models, so we need an array that tells us what model
#    # parameters these indices refer to.
#    aryMdl = np.zeros((varNumMdlsTtl, 3), dtype=np.float32)
#
#    # Model parameter can be represented as float32 as well:
#    vecMdlXpos = vecMdlXpos.astype(np.float32)
#    vecMdlYpos = vecMdlYpos.astype(np.float32)
#    vecMdlSd = vecMdlSd.astype(np.float32)
#
#    # The first column is to contain model x positions:
#    aryMdl[:, 0] = np.repeat(vecMdlXpos, int(varNumY * varNumPrfSizes))
#
#    # The second column is to contain model y positions:
#    aryMdl[:, 1] = np.repeat(
#                             np.tile(vecMdlYpos,
#                                     varNumX),
#                             varNumPrfSizes
#                             )
#
#    # The third column is to contain model pRF sizes:
#    aryMdl[:, 2] = np.tile(vecMdlSd, int(varNumX * varNumY))
#
#    # Earlier, we had removed models with a variance of zero. Thus, those
#    # models were ignored and are not present in the results. We remove them
#    # from the model-parameter-array:
#    aryMdl = aryMdl[vecLgcVar]
#
#    # Retrieve model parameters of 'winning' model for all voxels:
#    vecBstXpos = aryMdl[:, 0][vecResSsMinIdx]
#    vecBstYpos = aryMdl[:, 1][vecResSsMinIdx]
#    vecBstSd = aryMdl[:, 2][vecResSsMinIdx]
#
#    # Coefficient of determination (1 - ratio of (residual sum of squares by
#    # total sum of squares)):
#    vecBstR2 = np.subtract(1.0,
#                           np.divide(vecResSsMin,
#                                     vecSsTot)
#                           )
#
#    # Output list:
#    lstOut = [idxPrc,
#              vecBstXpos,
#              vecBstYpos,
#              vecBstSd,
#              vecBstR2,
#              np.zeros((varNumVox, (varNumBeta))).astype(np.float32)]
#
#    queOut.put(lstOut)
#
#print('---------Prepare pRF finding results for export')
#
## Create list for vectors with fitting results, in order to put the results
## into the correct order:
#lstResXpos = [None] * cfg.varPar
#lstResYpos = [None] * cfg.varPar
#lstResSd = [None] * cfg.varPar
#lstResR2 = [None] * cfg.varPar
#
## Put output into correct order:
#for idxRes in range(0, cfg.varPar):
#
#    # Index of results (first item in output list):
#    varTmpIdx = lstPrfRes[idxRes][0]
#
#    # Put fitting results into list, in correct order:
#    lstResXpos[varTmpIdx] = lstPrfRes[idxRes][1]
#    lstResYpos[varTmpIdx] = lstPrfRes[idxRes][2]
#    lstResSd[varTmpIdx] = lstPrfRes[idxRes][3]
#    lstResR2[varTmpIdx] = lstPrfRes[idxRes][4]
#
## Concatenate output vectors (into the same order as the voxels that were
## included in the fitting):
#aryBstXpos = np.zeros(0)
#aryBstYpos = np.zeros(0)
#aryBstSd = np.zeros(0)
#aryBstR2 = np.zeros(0)
#for idxRes in range(0, cfg.varPar):
#    aryBstXpos = np.append(aryBstXpos, lstResXpos[idxRes])
#    aryBstYpos = np.append(aryBstYpos, lstResYpos[idxRes])
#    aryBstSd = np.append(aryBstSd, lstResSd[idxRes])
#    aryBstR2 = np.append(aryBstR2, lstResR2[idxRes])
#
## Delete unneeded large objects:
#del(lstPrfRes)
#del(lstResXpos)
#del(lstResYpos)
#del(lstResSd)
#del(lstResR2)
#
## Put results form pRF finding into array (they originally needed to be saved
## in a list due to parallelisation). Voxels were selected for pRF model finding
## in two stages: First, a mask was applied. Second, voxels with low variance
## were removed. Voxels are put back into the original format accordingly.
#
## Number of voxels that were included in the mask:
#varNumVoxMsk = np.sum(aryLgcMsk)
#
## Array for pRF finding results, of the form aryPrfRes[voxel-count, 0:3], where
## the 2nd dimension contains the parameters of the best-fitting pRF model for
## the voxel, in the order (0) pRF-x-pos, (1) pRF-y-pos, (2) pRF-SD, (3) pRF-R2.
## At this step, only the voxels included in the mask are represented.
#aryPrfRes01 = np.zeros((varNumVoxMsk, 6), dtype=np.float32)
#
## Place voxels based on low-variance exlusion:
#aryPrfRes01[aryLgcVar, 0] = aryBstXpos
#aryPrfRes01[aryLgcVar, 1] = aryBstYpos
#aryPrfRes01[aryLgcVar, 2] = aryBstSd
#aryPrfRes01[aryLgcVar, 3] = aryBstR2
#
## Total number of voxels:
#varNumVoxTlt = (tplNiiShp[0] * tplNiiShp[1] * tplNiiShp[2])
#
## Place voxels based on mask-exclusion:
#aryPrfRes02 = np.zeros((varNumVoxTlt, 6), dtype=np.float32)
#aryPrfRes02[aryLgcMsk, 0] = aryPrfRes01[:, 0]
#aryPrfRes02[aryLgcMsk, 1] = aryPrfRes01[:, 1]
#aryPrfRes02[aryLgcMsk, 2] = aryPrfRes01[:, 2]
#aryPrfRes02[aryLgcMsk, 3] = aryPrfRes01[:, 3]
#
## Reshape pRF finding results into original image dimensions:
#aryPrfRes = np.reshape(aryPrfRes02,
#                       [tplNiiShp[0],
#                        tplNiiShp[1],
#                        tplNiiShp[2],
#                        6])
#
#del(aryPrfRes01)
#del(aryPrfRes02)
#
## Calculate polar angle map:
#aryPrfRes[:, :, :, 4] = np.arctan2(aryPrfRes[:, :, :, 1],
#                                   aryPrfRes[:, :, :, 0])
#
## Calculate eccentricity map (r = sqrt( x^2 + y^2 ) ):
#aryPrfRes[:, :, :, 5] = np.sqrt(np.add(np.power(aryPrfRes[:, :, :, 0],
#                                                2.0),
#                                       np.power(aryPrfRes[:, :, :, 1],
#                                                2.0)))
#
## List with name suffices of output images:
#lstNiiNames = ['_x_pos',
#               '_y_pos',
#               '_SD',
#               '_R2',
#               '_polar_angle',
#               '_eccentricity']
#
#print('---------Exporting results')
#
## Save nii results:
#for idxOut in range(0, 6):
#    # Create nii object for results:
#    niiOut = nb.Nifti1Image(aryPrfRes[:, :, :, idxOut],
#                            aryAff,
#                            header=hdrMsk
#                            )
#    # Save nii:
#    strTmp = (cfg.strPathOut + lstNiiNames[idxOut] + '.nii')
#    nb.save(niiOut, strTmp)
