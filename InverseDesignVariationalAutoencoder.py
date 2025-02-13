import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import re
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import argparse


# --------------------------- Load and Preprocess Images --------------------------- #
def read_tif_images_to_dataframe(folder_path, resize_shape=(128, 128)):
    """
    Reads all .tif image files from a directory, resizes them, and generates a Pandas DataFrame.

    Parameters:
        folder_path (str): Path to the folder containing .tif image files.
        resize_shape (tuple): Target size for resizing images (default is 128x128).

    Returns:
        pd.DataFrame: A DataFrame where each row corresponds to a flattened image.
    """
    images = []
    file_names = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.tif'):
            file_path = os.path.join(folder_path, file_name)
            try:
                image = Image.open(file_path)
                image = image.resize(resize_shape)  # Resize image
                image_array = np.array(image)

                if image_array.dtype == bool:
                    image_array = image_array.astype(int)  # Convert True/False to 1/0

                images.append(image_array.flatten())  # Flatten resized image
                file_names.append(file_name)
            except Exception as e:
                print(f"Error reading {file_name}: {e}")

    # Create DataFrame
    df = pd.DataFrame(images, index=file_names)
    return df


def display_images_from_dataframe(df, image_shapes, num_images=5):
    """
    Displays a few images from the DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame where each row is a flattened image.
        image_shapes (list): List of tuples indicating the shape of each image.
        num_images (int): Number of images to display (default is 5).
    """
    # Ensure we don't display more images than available
    num_images = min(num_images, len(df))

    for i in range(num_images):
        # Retrieve the flattened image and reshape it to its original shape
        image_array = df.iloc[i].values.astype(np.uint8)  # Convert to uint8 for proper display
        image_shape = image_shapes[i]
        reshaped_image = image_array.reshape(image_shape)

        # Plot the image
        plt.figure(figsize=(5, 5))
        plt.title(df.index[i])  # Use file name as title
        plt.imshow(reshaped_image, cmap="gray")
        plt.axis("off")
        plt.show()


def save_original_and_reconstructed(vae, images, save_directory, num_samples=5, filename_prefix="sample"):
    """
    Saves original and reconstructed images using the trained VAE.

    Parameters:
        vae (VAE): Trained Variational Autoencoder model.
        images (np.ndarray): Dataset containing input images.
        save_directory (str): Directory to save the images.
        num_samples (int): Number of images to visualize.
        filename_prefix (str): Prefix for the saved image filenames.
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)

    # Select `num_samples` random indices from the dataset
    random_indices = np.random.choice(len(images), num_samples, replace=False)
    sample_images = images[random_indices]

    # Encode and decode to get reconstructed images
    reconstructed_images = vae.decoder(vae.encoder(sample_images)[2]).numpy()

    for i in range(num_samples):
        plt.figure(figsize=(6, 3))

        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(sample_images[i], cmap="gray")
        plt.title("Original")
        plt.axis("off")

        # Reconstructed image
        plt.subplot(1, 2, 2)
        plt.imshow(reconstructed_images[i], cmap="gray")
        plt.title("Reconstructed")
        plt.axis("off")

        # Save the figure instead of displaying it
        save_path = os.path.join(save_directory, f"{filename_prefix}_{i+1}.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()  # Close the figure to free memory

    print(f"Original and reconstructed images saved in {save_directory}")



def save_loss_plot(history, save_directory, filename="loss_plot.png"):
    """
    Saves the training loss plot without displaying it.

    Parameters:
        history (tf.keras.callbacks.History): Training history containing loss values.
        save_directory (str): Directory where the plot will be saved.
        filename (str): Name of the saved loss plot image (default: "loss_plot.png").
    """
    # Extract loss from history
    loss = history.history['loss']
    # val_loss = history.history['val_loss']  # Uncomment if validation loss is needed

    # Create save directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)

    # Plot loss
    plt.figure(figsize=(8, 5))
    plt.plot(loss, label='Training Loss')
    # plt.plot(val_loss, label='Validation Loss')  # Uncomment if validation loss is included
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid()

    # Save the plot without displaying
    save_path = os.path.join(save_directory, filename)
    plt.savefig(save_path)
    plt.close()  # Close the figure to free memory

    print(f"Loss plot saved at: {save_path}")


def preprocess_images(df, image_shapes):
    """
    Preprocess the flattened images back to their original shape and normalize.
    """
    images = [df.iloc[i].to_numpy().reshape(image_shapes[i]) for i in range(len(df))]
    images = np.expand_dims(images, axis=-1)  # Add channel dimension for grayscale
    images = images.astype("float32")  # Normalize pixel values to [0, 1]
    return images

def save_latent_space_visualization(encoder, images, save_directory, filename="latent_space.png"):
    """
    Generates and saves the latent space visualization.

    Parameters:
        encoder (tf.keras.Model): The trained encoder model.
        images (np.ndarray): The dataset used for extracting latent space representations.
        save_directory (str): Directory where the plot will be saved.
        filename (str): Name of the saved plot image (default: "latent_space.png").
    """
    # Get latent space representations
    z_mean, z_log_var, _ = encoder.predict(images)

    # Create save directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)

    # Create the figure
    plt.figure(figsize=(12, 5))

    # Plot 1: Latent Space Scatter Plot
    plt.subplot(1, 2, 1)
    plt.scatter(z_mean[:, 62], z_mean[:, 63], alpha=0.5, color='blue')
    plt.title("Latent Space Visualization")
    plt.xlabel("z_mean[0]")
    plt.ylabel("z_mean[1]")

    # Plot 2: Distribution of Latent Dimensions
    plt.subplot(1, 2, 2)
    plt.hist(z_mean[:, 62], bins=30, alpha=0.5, label='z_mean[0]', color='red')
    plt.hist(z_mean[:, 63], bins=30, alpha=0.5, label='z_mean[1]', color='green')
    plt.title("Distribution of Latent Dimensions")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()

    # Adjust layout
    plt.tight_layout()

    # Save the figure instead of displaying it
    save_path = os.path.join(save_directory, filename)
    plt.savefig(save_path)
    plt.close()  # Close the figure to free memory

    print(f"Latent space visualization saved at: {save_path}")


def generate_and_save_middle_image(vae, images, save_directory, filename="middle_image.png"):
    """
    Generate a middle image by interpolating between two random images in the latent space and save it.

    Parameters:
        vae (VAE): The trained VAE model.
        images (np.ndarray): The training dataset.
        save_directory (str): Directory where the image will be saved.
        filename (str): Name of the saved image file (default: "middle_image.png").
    """
    # Ensure the save directory exists
    os.makedirs(save_directory, exist_ok=True)

    # Select two random images
    idx1, idx2 = np.random.choice(len(images), 2, replace=False)
    image1, image2 = images[idx1], images[idx2]

    # Encode the images to get latent representations
    z_mean1, _, _ = vae.encoder.predict(np.expand_dims(image1, axis=0))
    z_mean2, _, _ = vae.encoder.predict(np.expand_dims(image2, axis=0))

    # Interpolate between the two latent vectors
    z_middle = 0.5 * (z_mean1 + z_mean2)

    # Decode the interpolated latent vector to generate the middle image
    middle_image = vae.decoder.predict(z_middle)

    # Plot the original images and the middle image
    plt.figure(figsize=(9, 3))

    # Original image 1
    plt.subplot(1, 3, 1)
    plt.imshow(image1, cmap="gray")
    plt.title("Image 1")
    plt.axis("off")

    # Middle image
    plt.subplot(1, 3, 2)
    plt.imshow(middle_image[0], cmap="gray")
    plt.title("Middle Image")
    plt.axis("off")

    # Original image 2
    plt.subplot(1, 3, 3)
    plt.imshow(image2, cmap="gray")
    plt.title("Image 2")
    plt.axis("off")

    # Save the figure instead of displaying it
    save_path = os.path.join(save_directory, filename)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)  # Save with high resolution
    plt.close()  # Close figure to free memory

    print(f"Middle image saved at: {save_path}")


def save_optimized_microstructure(optimized_microstructure, save_directory, filename="optimized_microstructure.png"):
    """
    Saves the optimized microstructure image in the specified directory.

    Parameters:
        optimized_microstructure (np.ndarray): The generated microstructure image.
        save_directory (str): Directory where the image will be saved.
        filename (str): Name of the saved image file (default: "optimized_microstructure.png").
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)

    # Plot and save the optimized microstructure
    plt.figure(figsize=(5, 5))
    plt.imshow(optimized_microstructure[0, :, :, 0], cmap="gray")
    plt.title("Optimized Microstructure")

    # Save the figure instead of displaying it
    save_path = os.path.join(save_directory, filename)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)  # Save with high resolution
    plt.close()  # Close figure to free memory

    print(f"Optimized microstructure saved at: {save_path}")




# --------------------------- Define VAE Model --------------------------- #
# Define the encoder
class Sampling(layers.Layer):
    def call(self, inputs):
        """Reparameterization trick: sample from N(0, 1) scaled by the learned parameters."""
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_encoder(image_shape, latent_dim):
    encoder_inputs = layers.Input(shape=image_shape)
    x = layers.Conv2D(64, (3, 3), activation="relu", strides=2, padding="same")(encoder_inputs)
    x = layers.Conv2D(128, (3, 3), activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2D(256, (3, 3), activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = models.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    return encoder




def build_decoder(latent_dim, original_shape):

    decoder_inputs = layers.Input(shape=(latent_dim,))

    # Dense layer to match the size before reshaping
    x = layers.Dense(16 * 16 * 256, activation="relu")(decoder_inputs)  # Match the final downsampled shape in encoder
    x = layers.Reshape((16, 16, 256))(x)  # Reshape to spatial dimensions

    # Upsampling using Conv2DTranspose layers
    x = layers.Conv2DTranspose(256, (3, 3), activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(128, (3, 3), activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(64, (3, 3), activation="relu", strides=2, padding="same")(x)

    # Final layer to output the image with the original shape
    decoder_outputs = layers.Conv2DTranspose(original_shape[-1], (3, 3), activation="sigmoid", padding="same")(x)

    # Define the decoder model
    decoder = models.Model(decoder_inputs, decoder_outputs, name="decoder")

    return decoder



# Define the VAE
class VAE(models.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(
                tf.keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
            ))
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )
            total_loss = reconstruction_loss + 0.1* kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {"loss": total_loss, "reconstruction_loss": reconstruction_loss, "kl_loss": kl_loss}


# --------------------------- Main Execution --------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", required=True, help="Path to the folder containing .tif images")
    parser.add_argument("--save_directory", required=True, help="Directory to save outputs")
    parser.add_argument("--latent_dim", type=int, required=True, help="Latent space dimension")
    parser.add_argument("--image_shape", type=int, nargs=3, required=True, help="Image shape as three integers")
    parser.add_argument("--epochs", type=int, required=True, help="Number of epochs")
    args = parser.parse_args()
    
    # Load and preprocess images
    image_df = read_tif_images_to_dataframe(args.folder_path, resize_shape=(128, 128))
    image_shapes = [(128, 128) for _ in range(len(image_df))]

    images = preprocess_images(image_df, image_shapes)

    # Build VAE components
    encoder = build_encoder(tuple(args.image_shape), args.latent_dim)
    decoder = build_decoder(args.latent_dim, tuple(args.image_shape))
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=tf.keras.optimizers.Adam())

    # Train VAE
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=20, restore_best_weights=True)

    history= vae.fit(images,images, epochs=args.epochs, callbacks=[early_stopping])

    save_loss_plot(history, args.save_directory, filename="loss_plot.png")

    # Save latent space visualization
    save_latent_space_visualization(encoder, images, args.save_directory, filename="latent_space.png")

    # Show original and reconstructed images
    save_original_and_reconstructed(vae, images, args.save_directory, num_samples=5, filename_prefix="sample")

    # Generate and save middle image
    generate_and_save_middle_image(vae, images, args.save_directory, filename="middle_image.png")

    print("All processes completed successfully.")


# To run the above file run the command below. Edit the inputs as required
# python InverseDesignVariationalAutoencoder.py --folder_path "/home/uashfaq/InverseDesign/dataset_dominik" --save_directory "/home/uashfaq/InverseDesign/Inverse_design_UA_RN_DB_JS/Inverse_design_UA_RN_DB_JS/code/save_directory_dominik" --latent_dim 128 --image_shape 128 128 1 --epochs 2

