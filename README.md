# Variational_Autoencoder_Inverse_Design

This project leverages a Variational Autoencoder (VAE) architecture to generate novel
microstructure images for material science applications. The primary goal is to enhance
the understanding of microstructures by synthesizing realistic and diverse images that
can be used for simulation, analysis, and material design. The VAE model, a generative
deep learning approach, is trained on a dataset of existing microstructure images to learn
the underlying distribution of the material features, including grain patterns, porosity,
and other microstructural characteristics. By encoding the images into a latent space
and sampling from this space, the model generates new images that retain the statistical
properties of the training data while oering variations that could represent dierent
material congurations. The generated images are evaluated through visual inspection
and quantitative metrics to assess their quality and relevance. This work aims to push the
boundaries of computational material science, providing a tool for researchers to explore
new microstructural congurations and gain deeper insights into material behaviors.

There are two main files in this repository

## InverseDesignVariationalAutoencoder.py

This script implements a Variational Autoencoder (VAE) to learn the latent space representation of microstructures. It processes .tif images, encodes them into a lower-dimensional latent space, and reconstructs them through a decoder network. The main functionalities include:

1) Loading and preprocessing images: Converts grayscale images into a structured dataset.
2) Building the VAE model: Comprises an encoder, decoder, and reparameterization trick for latent space sampling.
3) Training the VAE: Optimized using binary cross-entropy loss and KL divergence.
4) Saving outputs: Includes original vs. reconstructed images, loss plots, latent space visualization, and interpolated images.
5) The script can be executed using command-line arguments to specify dataset paths and hyperparameters.

## InverseDesignBaysianOptimization.py

This script builds upon the trained VAE and performs Bayesian Optimization to generate microstructures with specific mechanical properties. The workflow includes:

1) Loading and preprocessing microstructure images.
2) Training the VAE model if not already trained.
3) Bayesian Optimization: Searches for an optimal latent vector that maximizes a target property (e.g., permeability).
4) Generating new microstructures: Decodes optimized latent vectors to produce synthetic microstructures.
5) Saving results, including optimized microstructures and visualization of the latent space.

Both scripts work together to learn, generate, and optimize microstructures for material design applications.
