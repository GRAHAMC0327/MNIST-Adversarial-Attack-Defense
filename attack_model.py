import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# --- 1. SETUP ---
# Load your trained model from Step 1
model = tf.keras.models.load_model('mnist_model.h5')
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# --- 2. THE FGSM ATTACK FUNCTION ---
def create_adversarial_pattern(input_image, input_label):
    input_image = tf.cast(input_image, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = model(input_image)
        loss = tf.keras.losses.sparse_categorical_crossentropy(input_label, prediction)
    
    # Get the gradients: How does the loss change based on the pixels?
    gradient = tape.gradient(loss, input_image)
    # Get the 'sign' of the gradient (either +1 or -1)
    signed_grad = tf.sign(gradient)
    return signed_grad

# --- 3. EXECUTE THE ATTACK ---
image = x_test[0:1] # Pick the first test image
label = y_test[0:1] # The real answer (e.g., '7')

# Try different 'Epsilons' (Strength of the attack)
epsilons = [0, 0.2, 0.3, 0.5]
plt.figure(figsize=(12, 3))

for i, eps in enumerate(epsilons):
    # Formula: Adversarial_Image = Original + (Epsilon * Noise)
    noise = create_adversarial_pattern(image, label)
    adv_x = image + (eps * noise)
    adv_x = tf.clip_by_value(adv_x, 0, 1) # Keep pixels between 0 and 1
    
    # Have the model guess the new noisy image
    prediction = model.predict(adv_x)
    pred_label = np.argmax(prediction)
    
    # Plotting the results
    plt.subplot(1, len(epsilons), i + 1)
    plt.imshow(adv_x[0].numpy().reshape(28, 28), cmap='gray')
    plt.title(f"Eps: {eps}\nPred: {pred_label}")
    plt.axis('off')

print("Displaying attack results...")
plt.show()