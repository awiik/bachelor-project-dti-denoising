# Noise removal in synthetically generated diffusion tensor imaging data using a denoising autoencoder

This repository contains code written for my bachelor project at NTNU.
The main purpose of the code is generating synthetic diffusion tensor imaging data with random noise added,
and removing the noise by means of a denoising autoencoder or with total variation denoising.

The full text of the thesis is provided in `thesis.pdf`

## Dependencies

To run the code you need the python packages NumPy, TensorFlow, Dipy and FURY.

Python 3.8 was used under Windows 10.

## Usage

Run `generate_training_data_6num.py` or `generate_training_data_shapes.py` to generate synthetic example datasets.

Then run `autoencoder_6num.py` which uses a denoising autoencoder to try to remove noise.
Alternatively, run `gradient_descent.py` which uses total variation denoising to try to remove noise.
Three images will be created showing the original, the noisy and the denoised versions of a sample image.

The files `make_dataset_from_mri.py` and `preprocess_dtis.py` were used to process real-world DTI data
from Human Connectome Project for use with the rest of the code.



