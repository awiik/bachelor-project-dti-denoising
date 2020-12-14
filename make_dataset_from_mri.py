import numpy as np
from random import randint, uniform
from math import floor
from generate_training_data_6num import add_noise_to_spd, add_noise_to_spd_unbiased
from helper_functions import make_6num

# IDs of data sets from Human Connectom Project
training_numbers = ['103818', '105923', '111312', '114823', '115320', '122317', '125525', '130518', '135528']
testing_numbers = ['137128']

path_start = 'D:\\Anders\\mri\\45_numpy\\'
file_name_end = '_3T_Diffusion_quadratic_form.npy'

z_min = 10 # In this case i will try to take advantage of all the z-slices.
z_max = 90
x_pos = 30
y_pos = 40
x_size = 20
y_size = 20

training_noised_imgs = np.ndarray(((z_max-z_min)*9, x_size, y_size, 1, 6))
training_imgs = np.ndarray(((z_max-z_min)*9, x_size, y_size, 1, 6))
testing_noised_imgs = np.ndarray((z_max-z_min, x_size, y_size, 1, 6))
testing_imgs = np.ndarray((z_max-z_min, x_size, y_size, 1, 6))

index = 0
for number in training_numbers:
    print('Training dataset', number)
    data = np.load(path_start + number + file_name_end)
    for z_slice in range(z_min, z_max):
        if (z_slice % 10 == 0): print('z-slice:',z_slice)
        img = data[x_pos:x_pos+x_size, y_pos:y_pos+y_size, z_slice:z_slice+1]
        for i in range(x_size):
            for j in range(y_size):
                training_imgs[index,i,j,0] = make_6num(img[i,j,0])
                training_noised_imgs[index,i,j,0] = make_6num(add_noise_to_spd_unbiased(img[i,j,0]))
                #training_noised_imgs[index,i,j,0] = make_6num(add_noise_to_spd(img[i,j,0], noisemax=0.001))
        index += 1

index = 0
for number in testing_numbers:
    print('Testing dataset', number)
    data = np.load(path_start + number + file_name_end)
    for z_slice in range(z_min, z_max):
        img = data[x_pos:x_pos+x_size, y_pos:y_pos+y_size, z_slice:z_slice+1]
        for i in range(x_size):
            for j in range(y_size):
                testing_imgs[index,i,j,0] = make_6num(img[i,j,0])
                testing_noised_imgs[index,i,j,0] = make_6num(add_noise_to_spd_unbiased(img[i,j,0]))
                #testing_noised_imgs[index,i,j,0] = make_6num(add_noise_to_spd(img[i,j,0], noisemax=0.001))
        index += 1

np.save('training_noised_mris', training_noised_imgs)
np.save('training_mris', training_imgs)
np.save('testing_noised_mris', testing_noised_imgs)
np.save('testing_mris', testing_imgs)
