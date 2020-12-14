import numpy as np
from random import randint, uniform
from math import floor

"""
See generate_trainig_data.py for notes
"""

def main():
    #N = 1200 # Number of training images
    #M = 200 # Number of testing images
    N = 6000 # Number of training images
    M = 1000 # Number of testing images
    #N = 60000
    #M = 10000
    #x_size = 10 # Image dimensions
    #y_size = 10
    x_size = 20 # Image dimensions
    y_size = 20

    training_noised_imgs = np.ndarray((N, x_size, y_size, 1, 6))
    training_imgs = np.ndarray((N, x_size, y_size, 1, 6))
    testing_noised_imgs = np.ndarray((M, x_size, y_size, 1, 6))
    testing_imgs = np.ndarray((M, x_size, y_size, 1, 6))

 
    print('Generating training data...')
    for img_n in range(N):
        if (img_n % 1000 == 0): print(img_n)
        training_noised_imgs[img_n], training_imgs[img_n] = make_image(x_size,y_size)
    print('Done!')
    print('Generating testing data...')
    for img_n in range(M):
        if (img_n % 1000 == 0): print(img_n)
        testing_noised_imgs[img_n], testing_imgs[img_n] = make_image(x_size,y_size)
    print('Done!')

    np.save('training_noised_imgs', training_noised_imgs)
    np.save('training_imgs', training_imgs)
    np.save('testing_noised_imgs', testing_noised_imgs)
    np.save('testing_imgs', testing_imgs)


def make_image(x_size, y_size):
    """ Returns a noisy, and noise-less image as a tuple (noise, noise-less)
    """
    k = 0 #Z slice
    image =  np.ndarray((x_size,y_size,1,6)) 
    noised_image =  np.ndarray((x_size,y_size,1,6)) 

    c_x = floor(x_size/2)
    c_y = floor(y_size/2)

    r = randint(6,9)
    #r = 4 # simple case
    phase = uniform(0,np.pi)

    for i in range(x_size):
        for j in range(y_size):
            if np.sqrt(np.abs(i-c_x)**2 + np.abs(j-c_y)**2) <= r:
            #if True:
                ang = np.arctan2(j-c_y, i-c_x) - np.pi/2
                eig_vals = np.array([4,2,1]) + np.array([4,2,1])*np.sin(ang+phase)**2
                #eig_vals = np.array([4,2,1]) # simple case
                M = get_rotated_spd_matrix(ang, eig_vals)
                #image[i,j,k] = np.array([M[0,0], M[0,1], M[0,2], M[1,1], M[1,2], M[2,2]])
                #nM = add_noise_to_spd(M) # noised M
            else:
                M =  np.diag([0.5, 0.5, 0.25])

            image[i,j,k] = np.array([M[0,0], M[0,1], M[0,2], M[1,1], M[1,2], M[2,2]])
            #nM = add_noise_to_spd(M) # noised M
            nM = add_noise_to_spd_unbiased(M) # noised M
            noised_image[i,j,k] = np.array([nM[0,0], nM[0,1], nM[0,2], nM[1,1], nM[1,2], nM[2,2]])
    return (noised_image, image)

def get_rotate_z_matrix(a):
    """Rotation matrix. Axis z. Angle a."""
    return np.array([[np.cos(a),-np.sin(a),0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])

def get_rotated_spd_matrix(ang, evals):
    """Takes an angle, and three eigenvalues and gives a spd matrix
    where the ellipsoid is pointed in the direction of ang"""
    R = get_rotate_z_matrix(ang)
    return np.dot(R, np.dot(np.diag(evals), np.transpose(R)))

def add_noise_to_spd(spd, noisemax=2):
    u,s,vh = np.linalg.svd(spd)
    noise = np.diag([uniform(0,noisemax), uniform(0,noisemax), uniform(0,noisemax)])
    return np.dot(u, np.dot(np.diag(s)+noise, vh))

def add_noise_to_spd_unbiased(spd):
    # Adds uniform noise to eigenvalues, expected value 0. 
    u,s,vh = np.linalg.svd(spd)
    n_max = 1
    #strength = 1/2
    strength = 1/4
    # noise = np.diag([uniform(max(-s[0]*strength,-n_max),min(s[0]*strength,n_max)), 
    #                  uniform(max(-s[1]*strength,-n_max),min(s[1]*strength,n_max)), 
    #                  uniform(max(-s[2]*strength,-n_max),min(s[2]*strength,n_max)),])
    noise = np.diag([uniform(-s[0]*strength,s[0]*strength), 
                     uniform(-s[1]*strength,s[1]*strength), 
                     uniform(-s[2]*strength,s[2]*strength)])
    # noise = np.diag([np.random.normal(0,0.5),
    #                  np.random.normal(0,0.5),
    #                  np.random.normal(0,0.5)])
    return np.dot(u, np.dot(np.diag(s)+noise, vh))

if __name__ == '__main__':
    main()