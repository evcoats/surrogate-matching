import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Parameters for data generation
input_size = 3     # Number of input features
time_steps = 50     # Number of time steps in the sequence
batch_size = 100    # Batch size for training the surrogate model
hidden_size_rnn = 80     # Hidden size for the surrogate RNN

# Step 1: Generate Input Sequences with 1 Probability 2/5
def generate_input_sequences(batch_size, time_steps, input_size):
    # Generate random binary sequences where 1 has probability 0.4
    probability = 0.1  
    sequences = np.random.binomial(1, probability, size=(batch_size, time_steps, input_size))
    return torch.tensor(sequences, dtype=torch.float32)

input_sequences = generate_input_sequences(batch_size, time_steps, input_size)

# Function to initialize sparse weights with ones
def initialize_sparse_weights(layer, sparsity):
    with torch.no_grad():
        weight = layer.weight
        size = weight.size()
        # Create a mask with the desired sparsity
        mask = (torch.rand(size) < sparsity).float()
        # Set weights to one where mask is one, zero elsewhere
        weight.data = mask

# Step 2: Define the Fixed Multilayer SNN Model with Sparse Weights
class LIFNeuron(nn.Module):
    def __init__(self, threshold=1.0, decay=0.9):
        super(LIFNeuron, self).__init__()
        self.threshold = threshold
        self.decay = decay

    def forward(self, input, mem):
        mem = mem * self.decay + input
        spike = (mem >= self.threshold).float()
        mem = mem * (1 - spike)  # Reset mem to 0 after spike
        return spike, mem

class MultilayerSNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, thresholds, decays, sparsity):
        super(MultilayerSNN, self).__init__()
        self.layers = nn.ModuleList()
        self.neurons = nn.ModuleList()
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.num_layers = len(layer_sizes) - 1
        for i in range(self.num_layers):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1], bias=False)
            self.layers.append(layer)
            self.neurons.append(LIFNeuron(threshold=thresholds[i], decay=decays[i]))
            # Initialize weights to be sparse with ones
            initialize_sparse_weights(layer, sparsity)
        # Freeze weights
        for param in self.parameters():
            param.requires_grad = False  # Freeze weights

    def forward(self, inputs):
        batch_size = inputs.size(0)
        time_steps = inputs.size(1)
        mems = [torch.zeros(batch_size, layer.out_features) for layer in self.layers]
        outputs = []
        for t in range(time_steps):
            x = inputs[:, t, :]
            for i in range(self.num_layers):
                if i == 0:
                    h = self.layers[i](x)
                else:
                    h = self.layers[i](spike)
                spike, mems[i] = self.neurons[i](h, mems[i])
            outputs.append(spike.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)  # Shape: (batch_size, time_steps, output_size)
        return outputs

# Step 3: Define the Surrogate RNN Model
class SurrogateRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SurrogateRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()  # To constrain outputs between 0 and 1

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out)
        out = self.sigmoid(out)
        return out

# Function to train the surrogate model
def train_surrogate_model(surrogate_model, input_sequences, snn_outputs, num_epochs, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(surrogate_model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = surrogate_model(input_sequences)
        loss = criterion(outputs, snn_outputs)
        loss.backward()
        optimizer.step()
        # Optionally print progress
        # if (epoch + 1) % 10 == 0:
        #     print(f"Surrogate Model Training Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")
    return surrogate_model

# Function to optimize inputs using the surrogate model
def optimize_inputs(surrogate_model, desired_outputs, input_size, time_steps, input_optimization_lr, num_input_epochs):
    optimized_inputs = torch.rand(1, time_steps, input_size, requires_grad=True)
    input_optimizer = torch.optim.Adam([optimized_inputs], lr=input_optimization_lr)
    criterion = nn.MSELoss()
    for epoch in range(num_input_epochs):
        input_optimizer.zero_grad()
        surrogate_outputs = surrogate_model(optimized_inputs)
        loss = criterion(surrogate_outputs, desired_outputs[:1])
        loss.backward()
        input_optimizer.step()
        # with torch.no_grad():
        #     optimized_inputs.clamp_(0, 1)  # Ensure inputs are within valid range
        # Optionally print progress
        # if (epoch + 1) % 50 == 0:
        #     print(f"Input Optimization Epoch [{epoch+1}/{num_input_epochs}], Loss: {loss.item():.6f}")
    return optimized_inputs.detach()

# Step 4: Hyperparameter Optimization Loop
def hyperparameter_optimization(num_iterations=10):
    best_accuracy = 0.0
    best_hyperparams = None
    results = []

    for iteration in range(num_iterations):
        print(f"\nIteration {iteration+1}/{num_iterations}")

        # Sample hyperparameters
        num_layers = random.choice([1,2,3])
        hidden_sizes_snn = [random.randint(30,70) for _ in range(num_layers)]
        thresholds = [random.uniform(0.5, 2.0) for _ in range(num_layers + 1)]
        decays = [random.uniform(0.2, 1.0) for _ in range(num_layers + 1)]
        input_optimization_lr = random.choice([0.01])
        input_optimization_epochs = random.choice([1000])
        sparsity = random.uniform(0.1, 0.9)  # Sparsity between 10% and 50%

        print(f"Hyperparameters:")
        print(f"  Number of Layers: {num_layers}")
        print(f"  Hidden Sizes: {hidden_sizes_snn}")
        print(f"  Thresholds: {thresholds}")
        print(f"  Decays: {decays}")
        print(f"  Sparsity: {sparsity:.2f}")
        print(f"  Input Optimization LR: {input_optimization_lr}")
        print(f"  Input Optimization Epochs: {input_optimization_epochs}")

        # Instantiate the Fixed SNN with sampled hyperparameters
        snn_model = MultilayerSNN(input_size, hidden_sizes_snn, input_size, thresholds, decays, sparsity)
        snn_model.eval()  # Ensure the SNN is in evaluation mode

        # Step 5: Get SNN Outputs
        with torch.no_grad():
            snn_outputs = snn_model(input_sequences)

        # Step 6: Train the Surrogate RNN Model
        surrogate_model = SurrogateRNN(input_size, hidden_size_rnn, input_size)
        surrogate_model = train_surrogate_model(surrogate_model, input_sequences, snn_outputs, num_epochs=3000)

        # Step 7: Input Optimization Using the Surrogate Model
        desired_outputs = input_sequences.clone()  # Identity mapping task
        optimized_inputs = optimize_inputs(surrogate_model, desired_outputs, input_size, time_steps, input_optimization_lr, input_optimization_epochs)

        # Step 8: Test the Optimized Inputs on the Fixed SNN
        binary_optimized_inputs = (optimized_inputs >= 0.5).float()
        with torch.no_grad():
            snn_outputs_test = snn_model(binary_optimized_inputs)

        # Step 9: Evaluate Performance
        desired_outputs_np = desired_outputs[:1].numpy()
        snn_outputs_np = snn_outputs_test.numpy()
        accuracy = np.mean(snn_outputs_np == desired_outputs_np)
        print(f"Accuracy of SNN Output: {accuracy * 100:.2f}%")

        # Record the results
        results.append({
            'iteration': iteration + 1,
            'hyperparameters': {
                'num_layers': num_layers,
                'hidden_sizes': hidden_sizes_snn,
                'thresholds': thresholds,
                'decays': decays,
                'sparsity': sparsity,
                'input_optimization_lr': input_optimization_lr,
                'input_optimization_epochs': input_optimization_epochs
            },
            'accuracy': accuracy
        })

        # Update best hyperparameters
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_hyperparams = results[-1]['hyperparameters']

    print("\nBest Hyperparameters:")
    print(f"  Accuracy: {best_accuracy * 100:.2f}%")
    print(f"  Hyperparameters: {best_hyperparams}")

    return results, best_hyperparams

# Run the hyperparameter optimization
num_iterations = 88 # Adjust as needed
results, best_hyperparams = hyperparameter_optimization(num_iterations)

# Optional: Plotting accuracies
accuracies = [result['accuracy'] for result in results]
plt.figure(figsize=(10, 4))
plt.plot(range(1, num_iterations + 1), accuracies, marker='o')
plt.title('Hyperparameter Optimization Results')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

# After finding the best hyperparameters, retrain the surrogate model and perform input optimization

print("\nRetraining the model with the best hyperparameters...")

# Instantiate the Fixed SNN with best hyperparameters
snn_model = MultilayerSNN(
    input_size,
    best_hyperparams['hidden_sizes'],
    input_size,
    best_hyperparams['thresholds'],
    best_hyperparams['decays'],
    best_hyperparams['sparsity']
)
snn_model.eval()  # Ensure the SNN is in evaluation mode

# Step 1: Get SNN Outputs with Best Hyperparameters
with torch.no_grad():
    snn_outputs = snn_model(input_sequences)

# Step 2: Define the Surrogate RNN Model
surrogate_model = SurrogateRNN(input_size, hidden_size_rnn, input_size)

# Step 3: Train the Surrogate RNN Model with Best Hyperparameters
surrogate_model = train_surrogate_model(surrogate_model, input_sequences, snn_outputs, num_epochs=1000)

# Step 4: Input Optimization Using the Surrogate Model
desired_outputs = input_sequences.clone()  # Identity mapping task
optimized_inputs = optimize_inputs(surrogate_model, desired_outputs, input_size, time_steps, best_hyperparams['input_optimization_lr'], best_hyperparams['input_optimization_epochs'])

# Step 5: Test the Optimized Inputs on the Fixed SNN
binary_optimized_inputs = (optimized_inputs >= 0.5).float()
with torch.no_grad():
    snn_outputs_test = snn_model(binary_optimized_inputs)

# Step 6: Evaluate Performance
desired_outputs_np = desired_outputs.numpy()
snn_outputs_np = snn_outputs_test.numpy()
accuracy = np.mean(snn_outputs_np == desired_outputs_np)
print(f"\nFinal Accuracy of SNN Output with Best Hyperparameters: {accuracy * 100:.2f}%")

# Step 7: Graph the Output of the SNN on an Example
time_steps_range = range(time_steps)
for i in range(input_size):
    plt.figure(figsize=(10, 2))
    plt.plot(time_steps_range, desired_outputs_np[0, :, i], label='Desired Output', linestyle='--', marker='o')
    plt.plot(time_steps_range, snn_outputs_np[0, :, i], label='SNN Output', linestyle='-', marker='x')
    plt.title(f'SNN Output vs Desired Output - Feature {i+1}')
    plt.xlabel('Time Step')
    plt.ylabel('Output Value')
    plt.legend()
    plt.show()
