import numpy as np
from dipy.viz import window, actor

from dipy.io.image import load_nifti, save_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table

import dipy.reconst.dti as dti

path_start = 'D:\\Anders\\mri\\45\\'
#dataset_numbers = ['103818', '105923', '111312']
dataset_numbers = ['114823']
#dataset_numbers = ['115320', '122317', '125525', '130518', '135528', '137128']
dir_name_end = '_3T_Diffusion_preproc\\'
path_end = '\\T1w\\Diffusion\\'

for dataset in dataset_numbers:
    print('Doing dataset', dataset)
    dname = path_start + dataset + dir_name_end + dataset + path_end
    fdwi = dname + 'data.nii.gz'
    fbval = dname + 'bvals'
    fbvec = dname + 'bvecs'


    data, affine, img = load_nifti(fdwi, return_img=True)


    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    gtab = gradient_table(bvals, bvecs)
    print('data.shape (%d, %d, %d, %d)' % data.shape)

    from dipy.segment.mask import median_otsu

    maskdata, mask = median_otsu(data, vol_idx=range(10, 50), median_radius=3,
                                numpass=1, autocrop=True, dilate=2)
    print('maskdata.shape (%d, %d, %d, %d)' % maskdata.shape)

    tenmodel = dti.TensorModel(gtab)

    tenfit = tenmodel.fit(maskdata)

    np.save('D:\\Anders\\mri\\45_numpy\\' + dataset + '_3T_Diffusion_quadratic_form',tenfit.quadratic_form)

