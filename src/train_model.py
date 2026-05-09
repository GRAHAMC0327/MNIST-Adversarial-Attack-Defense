import tensorflow as tf
from tensorflow.keras import layers, models

# --- 1. LOAD DATA ---
print("Loading MNIST data...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# --- 2. PREPROCESS ---
# Scale pixels to and reshape for the CNN (28x28 images with 1 color channel)
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# --- 3. BUILD THE BRAIN (CNN) ---
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax') # 10 outputs for digits 0-9
])

# --- 4. COMPILE (The Adam Optimizer) ---
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# --- 5. TRAIN ---
print("Starting training...")
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

# --- 6. SAVE ---
model.save('mnist_model.h5')
print("Finished! Your model is saved as mnist_model.h5")
