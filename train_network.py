import numpy as np
import nnfs
import random
import matplotlib.pyplot as plt
from load_data import load_quickdraw_data, train_test_split
import pickle
import os

random.seed(0)
nnfs.init()

# Dense Layer
class Dense_Layer:
    def __init__(self, n_inputs, n_neurons, weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self, n_inputs):
        self.inputs = n_inputs
        self.output = np.dot(n_inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dinputs = np.dot(dvalues, self.weights.T)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        if self.weight_regularizer_l1 > 0:
            self.dweights += self.weight_regularizer_l1 * np.sign(self.weights)
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        if self.bias_regularizer_l1 > 0:
            self.dbiases += self.bias_regularizer_l1 * np.sign(self.biases)
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

# ReLU Activation
class Activation_ReLU:
    def forward(self, n_inputs):
        self.inputs = n_inputs
        self.output = np.maximum(0, n_inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

# Dropout
class Dropout:
    def __init__(self, rate):
        self.rate = rate

    def forward(self, n_inputs):
        self.inputs = n_inputs
        self.binary_mask = np.random.binomial(1, self.rate, size=n_inputs.shape) / self.rate
        self.output = n_inputs * self.binary_mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask

# Base Loss
class Loss:
    def calculate(self, output, y, layers=[]):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        reg_loss = 0
        for layer in layers:
            if hasattr(layer, 'weights'):
                if layer.weight_regularizer_l1 > 0:
                    reg_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
                if layer.weight_regularizer_l2 > 0:
                    reg_loss += layer.weight_regularizer_l2 * np.sum(layer.weights ** 2)
                if layer.bias_regularizer_l1 > 0:
                    reg_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
                if layer.bias_regularizer_l2 > 0:
                    reg_loss += layer.bias_regularizer_l2 * np.sum(layer.biases ** 2)
        return data_loss + reg_loss

# Categorical Crossentropy
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        elif len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(len(y_pred_clipped)), y_true]
        losses = -np.log(correct_confidences)
        return losses

# Adam Optimizer
class Optimized_ADAM:
    def __init__(self, learning_rate=0.005, decay=1e-3, epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / (1 + self.decay * self.iterations)

    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

        updated_weight_momentums = layer.weight_momentums / (1 - self.beta_1**(self.iterations + 1))
        updated_bias_momentums = layer.bias_momentums / (1 - self.beta_1**(self.iterations + 1))

        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * (layer.dweights**2)
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * (layer.dbiases**2)

        updated_weight_cache = layer.weight_cache / (1 - self.beta_2**(self.iterations + 1))
        updated_bias_cache = layer.bias_cache / (1 - self.beta_2**(self.iterations + 1))

        layer.weights += -self.current_learning_rate * updated_weight_momentums / (np.sqrt(updated_weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * updated_bias_momentums / (np.sqrt(updated_bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1

# Softmax Activation
class Activation_Softmax:
    def forward(self, n_inputs):
        self.inputs = n_inputs
        exp_values = np.exp(n_inputs - np.max(n_inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

# Combined Softmax + Loss
class Activation_Softmax_Loss_CategoricalCrossentropy:
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true, layers=[layer1, layer2])

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples

# Global layers
layer1 = None
layer2 = None

def train_network():
    global layer1, layer2
    X, y, _ = load_quickdraw_data(num_samples_per_class=5000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    layer1 = Dense_Layer(784, 128, weight_regularizer_l1=1e-6, weight_regularizer_l2=1e-6,
                         bias_regularizer_l1=1e-6, bias_regularizer_l2=1e-6)
    activation1 = Activation_ReLU()
    dropout = Dropout(0.8)
    layer2 = Dense_Layer(128, 5)
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
    optimizer = Optimized_ADAM(learning_rate=0.005, decay=1e-5)

    train_accuracies = []
    train_losses = []

    for epoch in range(100):
        layer1.forward(X_train)
        activation1.forward(layer1.output)
        dropout.forward(activation1.output)
        layer2.forward(dropout.output)
        loss = loss_activation.forward(layer2.output, y_train)

        predictions = np.argmax(loss_activation.output, axis=1)
        y_true = np.argmax(y_train, axis=1)
        accuracy = np.mean(predictions == y_true)

        train_accuracies.append(accuracy)
        train_losses.append(loss)

        if epoch % 2 == 0:
            print(f"epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f}, lr: {optimizer.current_learning_rate:.5f}")

        loss_activation.backward(loss_activation.output, y_train)
        layer2.backward(loss_activation.dinputs)
        dropout.backward(layer2.dinputs)
        activation1.backward(dropout.dinputs)
        layer1.backward(activation1.dinputs)

        optimizer.pre_update_params()
        optimizer.update_params(layer1)
        optimizer.update_params(layer2)
        optimizer.post_update_params()

    # Test
    layer1.forward(X_test)
    activation1.forward(layer1.output)
    layer2.forward(activation1.output)
    loss = loss_activation.forward(layer2.output, y_test)
    predictions = np.argmax(loss_activation.output, axis=1)
    y_true = np.argmax(y_test, axis=1)
    accuracy = np.mean(predictions == y_true)
    print(f"Test on test data — acc: {accuracy:.3f}, loss: {loss:.3f}")

    # Save weights
    os.makedirs('weights', exist_ok=True)
    weights = {
        'layer1_weights': layer1.weights,
        'layer1_biases': layer1.biases,
        'layer2_weights': layer2.weights,
        'layer2_biases': layer2.biases
    }
    with open('weights/model_weights.pkl', 'wb') as f:
        pickle.dump(weights, f)

    # Plot training graphs
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_accuracies, label="Training Accuracy", color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label="Training Loss", color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_plots.png")
    plt.show()

    return layer1, activation1, dropout, layer2, loss_activation

if __name__ == "__main__":
    os.makedirs('weights', exist_ok=True)
    train_network()
