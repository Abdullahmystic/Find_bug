import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess the Sunflowers dataset
def preprocess_images(data):
    image = tf.image.resize(data['image'], (32, 32))  # Reduce image size for faster processing
    image = (image - 127.5) / 127.5  # Normalize to [-1, 1]
    return image

# Load and filter the dataset
dataset, metadata = tfds.load("tf_flowers", split="train", with_info=True)
sunflowers_dataset = dataset.filter(lambda x: tf.reduce_any(x["label"] == 0))  # Filter for sunflowers
sunflowers_dataset = sunflowers_dataset.map(preprocess_images).shuffle(500).batch(32)  # Smaller batch size

# Build the Generator
def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(8 * 8 * 128, input_shape=(100,)),
        layers.Reshape((8, 8, 128)),
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', activation='relu'),
        layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', activation='relu'),
        layers.Conv2DTranspose(3, (4, 4), padding='same', activation='tanh')  # Output image
    ])
    return model

# Build the Discriminator
def build_discriminator():
    model = tf.keras.Sequential([
        layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same', input_shape=(32, 32, 3)),
        layers.LeakyReLU(0.2),
        layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same'),
        layers.LeakyReLU(0.2),
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')  # Output: real (1) or fake (0)
    ])
    return model

# Compile the GAN
generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002), loss='binary_crossentropy', metrics=['accuracy'])

# GAN model
discriminator.trainable = False
gan_input = layers.Input(shape=(100,))
generated_image = generator(gan_input)
gan_output = discriminator(generated_image)
gan = tf.keras.Model(gan_input, gan_output)
gan.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002), loss='binary_crossentropy')

# Training Loop
def train_gan(epochs=5000, batch_size=32):
    for epoch in range(epochs):
        # Train Discriminator
        real_images = next(iter(sunflowers_dataset))
        batch_size = real_images.shape[0]
        random_latent_vectors = np.random.normal(0, 1, (batch_size, 100))
        fake_images = generator.predict(random_latent_vectors)

        real_labels = np.ones((batch_size, 1)) - 0.1  # Label smoothing for real images
        fake_labels = np.zeros((batch_size, 1))

        d_loss_real = discriminator.train_on_batch(real_images, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)

        # Train Generator
        misleading_labels = np.ones((batch_size, 1))  # Train generator to trick discriminator
        g_loss = gan.train_on_batch(random_latent_vectors, misleading_labels)

        # Display progress
        if epoch % 500 == 0:
            print(f"Epoch {epoch}: D Loss Real: {d_loss_real[0]}, D Loss Fake: {d_loss_fake[0]}, G Loss: {g_loss}")
            save_generated_images(epoch)

# Function to save and display generated images
def save_generated_images(epoch):
    random_latent_vectors = np.random.normal(0, 1, (16, 100))
    generated_images = generator.predict(random_latent_vectors)
    generated_images = (generated_images * 127.5 + 127.5).astype(np.uint8)

    plt.figure(figsize=(4, 4))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(generated_images[i])
        plt.axis('off')
    plt.savefig(f"generated_images_epoch_{epoch}.png")
    plt.show()

# Train the GAN
train_gan(epochs=3000)  # Reduce epochs to fit Colab runtime
