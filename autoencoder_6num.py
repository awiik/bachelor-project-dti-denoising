import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from dipy.viz import window, actor
from helper_functions import ssim_index, visualize, make_sym_mtx, reconstruct_img, ssim_mean_diffusivity

# import time
# start_time = time.time()

def loss_fun_6num(y_true, y_pred):
    tensor_of_norms = K.sum(tf.norm(y_true-y_pred, axis=-1), axis=-1)
    return tf.norm(tensor_of_norms, axis=[-2,-1])

def loss_fun_6num_new(y_true, y_pred):
    beta = 2
    gamma = 1
    lamb = 0.1 # Regularization strength
    beta_sum = 1/beta* K.sum(tf.norm(y_true-y_pred, axis=-1)**beta, axis=[1,2,3])
    # Note: y_pred (and y_true) are shape (batch_size, x, y, z, 6)
    lambda_sum = lamb*(K.sum(tf.norm(y_pred[:,:-1]-y_pred[:,1:], axis=-1)**gamma,axis=[1,2,3]) \
                 + K.sum(tf.norm(y_pred[:,:,:-1]-y_pred[:,:,1:]**gamma,axis=-1),axis=[1,2,3]))
    return beta_sum + lambda_sum

def loss_fun_6num_eig(y_true, y_pred):
    """Old loss function but new distance function"""
    # Same problem, matrix might be non-spd so root fails.
    tensor_of_norms = K.sum(dist_batch_naive(y_true,y_pred), axis=-1)
    return tf.norm(tensor_of_norms, axis=[-2,-1])

def loss_fun_6num_naive(y_true, y_pred):
    beta = 2
    gamma = 1
    lamb = 0.1 # Regularization strength
    beta_sum = 1/beta* K.sum(dist_batch_naive(y_true,y_pred)**beta, axis=[1,2,3])
    # Note: y_pred (and y_true) are shape (batch_size, x, y, z, 6)
    lambda_sum = lamb*(K.sum(dist_batch_naive(y_pred[:,:-1],y_pred[:,1:])**gamma, axis=[1,2,3]) \
                + K.sum(dist_batch_naive(y_pred[:,:,:-1],y_pred[:,:,1:]), axis=[1,2,3]))
    return beta_sum + lambda_sum

def dist_batch_naive(y_true, y_pred):
    """Takes batches y_true and y_pred, returns tensor of losses"""
    # We write this with map_fn. You might have to swap dtype for fn_output_signature in newer TF versions.
    return tf.map_fn(fn=dist_image, elems=(y_true, y_pred), dtype=tf.float32)

def dist_image(tup):
    """Takes image tuple (y_true, y_pred) and returns the loss"""
    return tf.map_fn(fn=dist_col, elems=tup, dtype=tf.float32)

def dist_col(tup):
    """Takes tuple of columns and returns the loss"""
    return tf.map_fn(fn=dist_z, elems=tup, dtype=tf.float32)

def dist_z(tup):
    """Takes tuple of columns and returns the loss"""
    return tf.map_fn(fn=dist_naive, elems=tup, dtype=tf.float32)

def dist_naive(tup):
    """Calculates distance according to 4.2.2 in Dissipative Schemes on Riemannian manifolds
    tup = (x,y)
    x and y are 6-vectors corresponding to a SPD 3x3 matrix"""
    x = tup[0] # Have to index with [0] because z=1
    y = tup[1]

    indices = tf.constant([[0], [1], [2], [4], [5], [8]])
    shape = tf.constant([9])
    A_upper_trig = tf.reshape(tf.scatter_nd(indices, x, shape), (3,3))
    A_diag = tf.linalg.band_part(A_upper_trig, 0, 0)
    A_strict_upper_trig = A_upper_trig - A_diag
    A=A_strict_upper_trig + tf.transpose(A_strict_upper_trig) + A_diag

    # Same for B
    B_upper_trig = tf.reshape(tf.scatter_nd(indices, y, shape), (3,3))
    B_diag = tf.linalg.band_part(B_upper_trig, 0, 0)
    B_strict_upper_trig = B_upper_trig - B_diag
    B=B_strict_upper_trig + tf.transpose(B_strict_upper_trig) + B_diag

    
    a_evals, a_evecs = tf.linalg.eigh(A)
    b_evals, b_evecs = tf.linalg.eigh(B)
    b_diag = tf.linalg.diag(K.abs(b_evals))
    B = tf.linalg.matmul(b_evecs, tf.linalg.matmul(b_diag, tf.transpose(b_evecs)))

    A_inv_root = a_evecs @ tf.linalg.diag(K.sqrt(K.abs(a_evals)**(-1))) @ tf.transpose(a_evecs)
    matrix_product = tf.linalg.matmul(A_inv_root, tf.linalg.matmul(B, A_inv_root))
    evals, evecs = tf.linalg.eigh(matrix_product)

    return K.sqrt(K.sum(K.log(evals)**2))

# Load data
#data_name = 'mris'
data_name = 'imgs'
training_noised_imgs = np.load('training_noised_' + data_name + '.npy')
training_imgs = np.load('training_' + data_name + '.npy')
testing_noised_imgs = np.load('testing_noised_' + data_name + '.npy')
testing_imgs = np.load('testing_' + data_name + '.npy')

train_ni = tf.convert_to_tensor(training_noised_imgs)
train_i = tf.convert_to_tensor(training_imgs)
test_ni = tf.convert_to_tensor(testing_noised_imgs)
test_i = tf.convert_to_tensor(testing_imgs)

# Normalize data, dividing by maximum absolute value
max_val = max([K.max(K.abs(train_ni)), K.max(K.abs(train_i)), K.max(K.abs(test_ni)), K.max(K.abs(test_i))])
train_ni = tf.math.multiply(training_noised_imgs, 1/max_val)
train_i = tf.math.multiply(training_imgs, 1/max_val)
test_ni = tf.math.multiply(testing_noised_imgs, 1/max_val)
test_i = tf.math.multiply(testing_imgs, 1/max_val)
# So now data should be values in [-1, 1]


x_size = training_noised_imgs.shape[1]
y_size = training_noised_imgs.shape[2]

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(x_size, y_size, 1, 6)),
    keras.layers.Dense(400, activation='tanh'),  
    keras.layers.Dense(x_size*y_size*1*6, activation='tanh'),
    keras.layers.Reshape((x_size, y_size, 1, 6))
])

model.summary()
print(train_i.shape)

model.compile(optimizer='adam', loss=loss_fun_6num_new) 
print('Compiled!')


model.fit(train_ni, train_i, epochs=10)

test_loss = model.evaluate(test_ni,  test_i, verbose=1)
print('\nTest loss:', test_loss)

prediction = model.predict(test_ni)

def average_ssim_test_data_dae(test_i, prediction):
    res = 0.0
    datapoints = test_i.shape[0]
    for index in range(test_i.shape[0]):
        res += ssim_index(test_i[index], prediction[index]).numpy()
    return res/datapoints


def visualize_result(index):
    print('loss_fun_6num of this result:',loss_fun_6num(test_i[index], prediction[index]))
    print(test_i[index].shape)
    print(prediction[index].shape)

    print('ssim of this result:',ssim_index(test_i[index], prediction[index]).numpy())
    original_image = reconstruct_img(test_i[index].numpy())
    noisy_image = reconstruct_img(test_ni[index].numpy())
    denoised_image = reconstruct_img(prediction[index]) 
    print(denoised_image.shape)
    print(noisy_image.shape)

    print('loss_fun_6num and ssim, ssim_m_d, with the untreated noisy image:')
    print(loss_fun_6num(test_i[index], test_ni[index]))
    print(ssim_index(test_i[index], test_ni[index]))
    print(ssim_mean_diffusivity(test_i[index], test_ni[index]))


    k = 0 # z slice


    ###### Print and test
    original_evals = np.zeros((x_size,y_size,1,3))
    original_evecs = np.zeros((x_size,y_size,1,3,3))
    denoised_evals = np.zeros((x_size,y_size,1,3))
    denoised_evecs = np.zeros((x_size,y_size,1,3,3))
    noisy_evals = np.zeros((x_size,y_size,1,3))
    noisy_evecs = np.zeros((x_size,y_size,1,3,3))

    for i in range(x_size):
        for j in range(y_size):
            #print(denoised_image[i,j,k])
            #print(np.linalg.eig(denoised_image[i,j,k]))

            # Calculate the evals/evecs and order them in descending order
            original_evals[i,j,k], original_evecs[i,j,k] = np.linalg.eigh(original_image[i,j,k])
            original_evals[i,j,k], original_evecs[i,j,k] = np.flip(original_evals[i,j,k]), np.fliplr(original_evecs[i,j,k])
            denoised_evals[i,j,k], denoised_evecs[i,j,k] = np.linalg.eigh(denoised_image[i,j,k])
            denoised_evals[i,j,k], denoised_evecs[i,j,k] = np.flip(denoised_evals[i,j,k]), np.fliplr(denoised_evecs[i,j,k])
            noisy_evals[i,j,k], noisy_evecs[i,j,k] = np.linalg.eigh(noisy_image[i,j,k])
            noisy_evals[i,j,k], noisy_evecs[i,j,k] = np.flip(noisy_evals[i,j,k]), np.fliplr(noisy_evecs[i,j,k])


    scale = (3 if (data_name == 'mris') else 1)
    visualize(original_evals, original_evecs, scale, 'original_ellipsoids.png', (600,600)) 
    visualize(noisy_evals, noisy_evecs, scale, 'noisy_ellipsoids.png', (600,600)) 
    visualize(denoised_evals, denoised_evecs, scale, 'denoised_ellipsoids.png', (600,600))
    

print('Average denoised ssim over test data:',average_ssim_test_data_dae(test_i, prediction))
print('Average untreated ssim over test data:',average_ssim_test_data_dae(test_i, test_ni))
# print('EXECUTION TIME: ', time.time()-start_time)

visualize_result(12) 
#visualize_result(367)
#visualize_result(33)
