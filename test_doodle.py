import numpy as np
import pickle
from PIL import Image
from train_network import Dense_Layer, Activation_ReLU, Activation_Softmax
import os

# Classes in same order as training
CLASSES = ['headphones', 'hexagon', 'pizza', 'bucket', 'clock']

# Load trained weights
with open('weights/model_weights.pkl', 'rb') as f:
    weights = pickle.load(f)

# Load and preprocess input image
def preprocess_image(filepath):
    img = Image.open(filepath).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))               # Resize to 28x28
    img = np.array(img).astype(np.float32)

    # Invert if necessary (user-drawn usually black-on-white)
    if img.mean() > 127:
        img = 255 - img

    img = img / 255.0                        # Normalize
    img = img.reshape(1, 784)               # Flatten
    return img

# Recreate the model architecture
layer1 = Dense_Layer(784, 128)
activation1 = Activation_ReLU()
layer2 = Dense_Layer(128, 5)
activation2 = Activation_Softmax()

# Load weights
layer1.weights = weights['layer1_weights']
layer1.biases = weights['layer1_biases']
layer2.weights = weights['layer2_weights']
layer2.biases = weights['layer2_biases']

# Predict
def predict(image_path):
    x = preprocess_image(image_path)

    layer1.forward(x)
    activation1.forward(layer1.output)
    layer2.forward(activation1.output)
    activation2.forward(layer2.output)

    probs = activation2.output[0]  # Shape (5,)
    sorted_indices = np.argsort(probs)[::-1]

    print(f"\n📷 Prediction for: {image_path}\n")
    for idx in sorted_indices:
        print(f"{CLASSES[idx]:>7}: {probs[idx]*100:.2f}%")

    print(f"\n🎯 Final Prediction: {CLASSES[sorted_indices[0]]} ✅")

if __name__ == "__main__":
    input_path = "input.png"
    if not os.path.exists(input_path):
        print(f"❌ '{input_path}' not found.")
    else:
        predict(input_path)
