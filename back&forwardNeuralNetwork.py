import numpy as np

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Neural Network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        # Initialize weights and biases
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Weights (input → hidden) & (hidden → output)
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.1
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.1
        self.bias_output = np.zeros((1, output_size))

    # Forward propagation
    def forward(self, X):
        self.input_layer = X
        self.hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = relu(self.hidden_layer_input)  # Activation
        
        self.final_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.final_output = sigmoid(self.final_input)  # Activation
        
        return self.final_output

    # Backpropagation
    def backward(self, X, y, output):
        # Compute error
        error = y - output
        d_output = error * sigmoid_derivative(output)  # Gradient of sigmoid
        
        # Compute error in hidden layer
        error_hidden = d_output.dot(self.weights_hidden_output.T)
        d_hidden = error_hidden * relu_derivative(self.hidden_layer_output)  # Gradient of ReLU
        
        # Update weights and biases (Gradient Descent)
        self.weights_hidden_output += self.hidden_layer_output.T.dot(d_output) * self.learning_rate
        self.bias_output += np.sum(d_output, axis=0, keepdims=True) * self.learning_rate
        self.weights_input_hidden += X.T.dot(d_hidden) * self.learning_rate
        self.bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * self.learning_rate

    # Train function
    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            
            if epoch % 100 == 0:
                loss = np.mean(np.square(y - output))
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

# Example usage
if __name__ == "__main__":
    # Create a dataset (XOR problem)
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([[0], [1], [1], [0]])  # Expected XOR outputs
    
    # Create Neural Network
    nn = NeuralNetwork(input_size=2, hidden_size=5, output_size=1, learning_rate=0.1)

    # Train the network
    nn.train(X_train, y_train, epochs=1000)

    # Test after training
    for i in range(4):
        print(f"Input: {X_train[i]} -> Predicted: {nn.forward(X_train[i])}")
