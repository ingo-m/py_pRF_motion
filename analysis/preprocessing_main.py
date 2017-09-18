print('------Load & preprocess nii data')

# Load mask (to restrict model fitting):
aryMask, hdrMsk, affMsk = fncLoadNii(cfg.strPathNiiMask)

# Mask is loaded as float32, but is better represented as integer:
aryMask = np.array(aryMask).astype(np.int16)

# Number of non-zero voxels in mask:
# varNumVoxMsk = int(np.count_nonzero(aryMask))

# Dimensions of nii data:
vecNiiShp = aryMask.shape

# Total number of voxels:
varNumVoxTlt = (vecNiiShp[0] * vecNiiShp[1] * vecNiiShp[2])

# Reshape mask:
aryMask = np.reshape(aryMask, varNumVoxTlt)

# List for arrays with functional data (possibly several runs):
lstFunc = []

# Number of runs:
varNumRun = len(cfg.lstPathNiiFunc)

# Loop through runs and load data:
for idxRun in range(varNumRun):

    print(('------Preprocess run ' + str(idxRun + 1)))

    # Load 4D nii data:
    aryTmpFunc, _, _ = fncLoadLargeNii(cfg.lstPathNiiFunc[idxRun])

    # Dimensions of nii data (including temporal dimension; spatial dimensions
    # need to be the same for mask & functional data):
    vecNiiShp = aryTmpFunc.shape

    # Preprocessing of nii data:
    aryTmpFunc = funcPrfPrePrc(aryTmpFunc,
                               aryMask=aryMask,
                               lgcLinTrnd=True,
                               varSdSmthTmp=cfg.varSdSmthTmp,
                               varSdSmthSpt=cfg.varSdSmthSpt,
                               varIntCtf=cfg.varIntCtf,
                               varPar=cfg.varPar)

    # Reshape functional nii data, from now on of the form
    # aryTmpFunc[voxelCount, time]:
    aryTmpFunc = np.reshape(aryTmpFunc, [varNumVoxTlt, vecNiiShp[3]])

    # Convert intensities into z-scores. If there are several pRF runs, these
    # are concatenated. Z-scoring ensures that differences in mean image
    # intensity and/or variance between runs do not confound the analysis.
    # Possible enhancement: Explicitly model across-runs variance with a
    # nuisance regressor in the GLM.
    aryTmpStd = np.std(aryTmpFunc, axis=1)

    # In order to avoid devision by zero, only divide those voxels with a
    # standard deviation greater than zero:
    aryTmpLgc = np.greater(aryTmpStd.astype(np.float32),
                           np.array([0.0], dtype=np.float32)[0])
    # Z-scoring:
    aryTmpFunc[aryTmpLgc, :] = np.divide(aryTmpFunc[aryTmpLgc, :],
                                         aryTmpStd[aryTmpLgc, None])
    # Set voxels with a variance of zero to intensity zero:
    aryTmpLgc = np.not_equal(aryTmpLgc, True)
    aryTmpFunc[aryTmpLgc, :] = np.array([0.0], dtype=np.float32)[0]

    # Apply mask:
    aryLgcMsk = np.greater(aryMask.astype(np.int16),
                           np.array([0], dtype=np.int16)[0])
    aryTmpFunc = aryTmpFunc[aryLgcMsk, :]

    # Put preprocessed functional data of current run into list:
    lstFunc.append(np.copy(aryTmpFunc))
    del(aryTmpFunc)

# Put functional data from separate runs into one array. 2D array of the form
# aryFunc[voxelCount, time]
aryFunc = np.concatenate(lstFunc, axis=1).astype(np.float32, copy=False)
del(lstFunc)

# Voxels that are outside the brain and have no, or very little, signal should
# not be included in the pRF model finding. We take the variance over time and
# exclude voxels with a suspiciously low variance. Because the data given into
# the cython or GPU function has float32 precision, we calculate the variance
# on data with float32 precision.
aryFuncVar = np.var(aryFunc, axis=1, dtype=np.float32)

# Is the variance greater than zero?
aryLgcVar = np.greater(aryFuncVar,
                       np.array([0.0001]).astype(np.float32)[0])

# Array with functional data for which conditions (mask inclusion and cutoff
# value) are fullfilled:
aryFunc = aryFunc[aryLgcVar, :]

# Number of voxels for which pRF finding will be performed:
varNumVoxInc = aryFunc.shape[0]

print('---------Number of voxels on which pRF finding will be performed: '
      + str(varNumVoxInc))

print('---------Preprocess pRF time course models')

# Preprocessing of pRF time course models:
aryPrfTc = funcPrfPrePrc(aryPrfTc,
                         aryMask=np.array([]),
                         lgcLinTrnd=False,
                         varSdSmthTmp=cfg.varSdSmthTmp,
                         varSdSmthSpt=0.0,
                         varIntCtf=0.0,
                         varPar=cfg.varPar)
