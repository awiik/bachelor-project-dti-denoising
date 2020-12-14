import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from dipy.viz import window, actor
import matplotlib.pyplot as plt

def make_sym_mtx(v):
    """Makes a symmetric 3x3 matrix, from 6 numbers"""
    res = np.ndarray((3,3))
    res[0,0] = v[0]
    res[1,0] = res[0,1] = v[1]
    res[2,0] = res[0,2] = v[2]
    res[1,1] = v[3]
    res[1,2] = res[2,1] = v[4]
    res[2,2] = v[5]
    return res

def make_6num(M):
    return np.array([M[0,0], M[0,1], M[0,2], M[1,1], M[1,2], M[2,2]])

def reconstruct_img(img):
    """Takes an image with 6-vector voxels, and returns the same image
    with matrix voxels"""
    k = 0 # I assume we are working on z=0
    x_sz = img.shape[0]
    y_sz = img.shape[1]
    res = np.ndarray((x_sz, y_sz, 1, 3, 3))
    for i in range(x_sz):
        for j in range(y_sz):
            res[i,j,k] = make_sym_mtx(img[i,j,k])
    return res

def ssim_index(y_true, y_pred):
    # Does ssim as tensorflow implements it
    img1 = K.sum(y_true, axis=-2) # Sum away z direction
    img2 = K.sum(y_pred, axis=-2) 
    img1 = tf.cast(img1+1, tf.float64) # Add 1 to keep values in interval [0,2]
    img2 = tf.cast(img2+1, tf.float64)
    return tf.image.ssim(img1, img2, 2)

def ssim_mean_diffusivity(y_true, y_pred):
#     k=0
#     img1 = (K.sum(y_true, axis=-2)).numpy() # Sum away z direction
#     img2 = (K.sum(y_pred, axis=-2)).numpy()
#     for i in range(img1.shape[0]):
#         for j in range(img1.shape[1]):
#             img1[i,j] = np.trace(make_sym_mtx(img1[i,j]))/3
#             img2[i,j] = np.trace(make_sym_mtx(img2[i,j]))/3
#     # The images are now 2x2 arrays, with each point being the mean diffusivity (average of eigenvalues)
#     # This is not negative
#     return tf.image.ssim(tf.convert_to_tensor(img1,dtype=tf.float64), tf.convert_to_tensor(img2,dtype=tf.float64), max(img1.max(), img2.max())
    pass

def ssim_test(y_true, y_pred):
    # Takes average ssim over 6 channels
    # Comfirmed my suspicion that ssim takes average over each channel.
    img1 = K.sum(y_true, axis=-2) 
    img2 = K.sum(y_pred, axis=-2) 
    img1 = tf.cast(img1, tf.float64)
    img2 = tf.cast(img2, tf.float64)
    res = 0

    for i in range(6):
        res += tf.image.ssim(img1[:,:,i:i+1], img2[:,:,i:i+1], 2).numpy()
    return res/6

def visualize(evals,evecs,viz_scale=0.5, fname='tensor_ellipsoids.png', size=(1000,1000)):
    # Do vizualisation
    interactive = True

    ren = window.Scene()

    from dipy.data import get_sphere
    #sphere = get_sphere('symmetric362')
    #sphere = get_sphere('repulsion724')
    sphere = get_sphere('symmetric642')

    # Calculate the colors. See dipy documentation.
    from dipy.reconst.dti import fractional_anisotropy, color_fa
    FA = fractional_anisotropy(evals)
    #print(FA)
    FA[np.isnan(FA)] = 0
    FA = np.clip(FA, 0, 1)
    RGB = color_fa(FA, evecs)
    k=0
    cfa = RGB[:, :, k:k+1]
    # Normalizing like this increases the contrast, but this will make the contrast different across plots
    #cfa /= cfa.max()

    # imgplot = plt.imshow(FA, cmap='gray')
    # plt.show()


    ren.add(actor.tensor_slicer(evals, evecs, sphere=sphere, scalar_colors=cfa, scale=viz_scale, norm=False))

    if interactive:
        window.show(ren)

    window.record(ren, n_frames=1, out_path=fname, size=(1000, 1000))