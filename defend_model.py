import tensorflow as tf
import numpy as np

# 1. Load the "Victim" model and data
model = tf.keras.models.load_model('mnist_model.h5')
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# 2. Function to create adversarial images for training
def create_adversarial_batch(images, labels, eps=0.1):
    images = tf.cast(images, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(images)
        pred = model(images)
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, pred)
    grad = tape.gradient(loss, images)
    signed_grad = tf.sign(grad)
    return images + (eps * signed_grad)

# 3. Create a "Mixed" Dataset (Clean + Attacked)
print("Generating adversarial images for training...")
# We'll take a portion of the training data and attack it
x_adv_train = create_adversarial_batch(x_train[:10000], y_train[:10000])
x_combined = np.concatenate([x_train, x_adv_train])
y_combined = np.concatenate([y_train, y_train[:10000]])

# 4. Retrain (The Defense)
print("Retraining the model to be robust...")
model.fit(x_combined, y_combined, epochs=3, batch_size=64)

# 5. Save the new "Defended" model
model.save('defended_model.h5')
print("Success! Your defended model is saved as defended_model.h5")