# VAE-on-MNSIT
This project implements a Variational Autoencoder (VAE) using Keras and TensorFlow on the MNIST dataset. It encodes images into a 2D latent space and reconstructs them via a decoder. The model is trained by minimizing reconstruction loss and KL divergence, enabling compressed and structured representation of handwritten digits.
Variational Autoencoder (VAE) on MNIST
This project implements a Variational Autoencoder (VAE) using Keras and TensorFlow, applied to the MNIST dataset of handwritten digits.

Overview
The VAE learns to encode input images into a 2D latent space, from which it reconstructs the original digits. By minimizing reconstruction loss and KL divergence, the model achieves a structured, compressed representation of the data.

Features
Uses convolutional layers in the encoder and decoder

2D latent space for visualization

Custom training loop with loss tracking

Trained on the MNIST dataset (loaded from a local .npz file)

Requirements
Python 3.x

TensorFlow

Keras

NumPy

