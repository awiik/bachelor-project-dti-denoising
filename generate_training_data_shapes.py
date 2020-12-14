import numpy as np
from random import randint, uniform
from math import floor
from generate_training_data_6num import get_rotated_spd_matrix, add_noise_to_spd, add_noise_to_spd_unbiased


def main():
    #N = 1200 # Number of training images
    #M = 200 # Number of testing images
    N = 6000 # Number of training images
    M = 100 # Number of testing images
    #N = 60000
    #M = 10000
    # x_size = 10 # Image dimensions
    # y_size = 10
    x_size = 20 # Image dimensions
    y_size = 20

    training_noised_imgs = np.ndarray((N, x_size, y_size, 1, 6))
    training_imgs = np.ndarray((N, x_size, y_size, 1, 6))
    testing_noised_imgs = np.ndarray((M, x_size, y_size, 1, 6))
    testing_imgs = np.ndarray((M, x_size, y_size, 1, 6))

 
    print('Generating training data...')
    for img_n in range(N):
        if (img_n % 1000 == 0): print(img_n)
        training_noised_imgs[img_n], training_imgs[img_n] = make_rectangle_image(x_size,y_size)
    print('Done!')
    print('Generating testing data...')
    for img_n in range(M):
        if (img_n % 1000 == 0): print(img_n)
        testing_noised_imgs[img_n], testing_imgs[img_n] = make_box_image(x_size,y_size)
    print('Done!')

    np.save('training_noised_imgs', training_noised_imgs)
    np.save('training_imgs', training_imgs)
    np.save('testing_noised_imgs', testing_noised_imgs)
    np.save('testing_imgs', testing_imgs)

def make_rectangle_image(x_size, y_size):
    """ Returns a noisy, and noise-less image as a tuple (noise, noise-less)
    """
    k = 0 #Z slice
    image =  np.ndarray((x_size,y_size,1,6)) 
    noised_image =  np.ndarray((x_size,y_size,1,6)) 

    min_size = x_size // 3
    p_x = randint(0, x_size-1)
    p_y = randint(0, y_size-1)
    width = randint(min_size, x_size//2)
    height = randint(min_size, y_size//2)

    for i in range(x_size):
        for j in range(y_size):
            if (i in range(p_x, min(p_x+width, x_size))) and (j in range(p_y, min(p_y+height, y_size))):
                M = get_rotated_spd_matrix(np.pi/4, [4,2,1])
            else:
                M =  np.diag([0.5, 0.5, 0.25])

            image[i,j,k] = np.array([M[0,0], M[0,1], M[0,2], M[1,1], M[1,2], M[2,2]])
            nM = add_noise_to_spd_unbiased(M) # noised M
            #nM = add_noise_to_spd(M, noisemax=1) # noised M
            noised_image[i,j,k] = np.array([nM[0,0], nM[0,1], nM[0,2], nM[1,1], nM[1,2], nM[2,2]])

    return (noised_image, image)

def make_box_image(x_size, y_size):
    """ Returns a noisy, and noise-less image as a tuple (noise, noise-less)
    """
    k = 0 #Z slice
    image =  np.ndarray((x_size,y_size,1,6)) 
    noised_image =  np.ndarray((x_size,y_size,1,6)) 

    min_size = x_size // 3
    p_x = randint(0, x_size-1)
    p_y = randint(0, y_size-1)
    width = randint(min_size, x_size//2)
    height = randint(min_size, y_size//2)

    for i in range(x_size):
        for j in range(y_size):
            if (i in range(p_x, min(p_x+width, x_size))) and (j in range(p_y, min(p_y+height, y_size))) and \
               not (abs(i-(p_x + width//2)) <= width//4 and abs(j-(p_y + height//2)) <= height//4):
                M = get_rotated_spd_matrix(np.pi/4, [4,2,1])
            else:
                M =  np.diag([0.5, 0.5, 0.25])

            image[i,j,k] = np.array([M[0,0], M[0,1], M[0,2], M[1,1], M[1,2], M[2,2]])
            nM = add_noise_to_spd_unbiased(M) # noised M
            #nM = add_noise_to_spd(M, noisemax=1) # noised M
            noised_image[i,j,k] = np.array([nM[0,0], nM[0,1], nM[0,2], nM[1,1], nM[1,2], nM[2,2]])

    return (noised_image, image)


def add_box(image, x_size, y_size):
    """Modifies image"""
    min_size = x_size // 5
    p_x = randint(0, x_size-1)
    p_y = randint(0, y_size-1)
    width = randint(min_size, x_size - x_size//4)
    height = randint(min_size, y_size - y_size//4)
    k = 0
    for i in range(p_x, min(p_x+width, x_size)):
        for j in range(p_y, min(p_y+height, y_size)):
            if abs(i-(p_x + width//2)) <= width//4 and abs(j-(p_y + height//2)) <= height//4:
                continue
            image[i,j,k] = get_rotated_spd_matrix(np.pi/4, [4,2,1])


if __name__ == '__main__':
    main()