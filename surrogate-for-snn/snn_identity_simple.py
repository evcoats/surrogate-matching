import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Parameters
input_size = 5   # Number of input features
output_size = 5     # Number of output features (same as input_size for identity mapping)
time_steps = 50     # Number of time steps in the sequence
batch_size = 200    # Batch size for training the surrogate model
hidden_size = 100    # Hidden size for the surrogate RNN

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Step 1: Generate Input Sequences
def generate_input_sequences(batch_size, time_steps, input_size):
    # Generate random binary sequences (0 or 1)
    sequences = np.random.randint(0, 2, size=(batch_size, time_steps, input_size))
    return torch.tensor(sequences, dtype=torch.float32)

input_sequences = generate_input_sequences(batch_size, time_steps, input_size)

# Step 2: Define the Modified SNN Model
class LIFNeuron(nn.Module):
    def __init__(self, threshold=1.5, decay=0.9):  # Increased threshold to 1.5
        super(LIFNeuron, self).__init__()
        self.threshold = threshold
        self.decay = decay

    def forward(self, input, mem):
        mem = mem * self.decay + input
        spike = (mem >= self.threshold).float()
        mem = mem * (1 - spike)  # Reset mem to 0 after spike
        return spike, mem

class SimpleSNN(nn.Module):
    def __init__(self, input_size, output_size, threshold=1.5, decay=0.9):
        super(SimpleSNN, self).__init__()
        self.neuron = LIFNeuron(threshold, decay)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        mem = torch.zeros(batch_size, inputs.size(2))  # Initialize membrane potential
        outputs = []
        for t in range(inputs.size(1)):
            x = inputs[:, t, :]
            s, mem = self.neuron(x, mem)
            outputs.append(s.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)  # Shape: (batch_size, time_steps, output_size)
        return outputs

# Instantiate the fixed SNN model with higher threshold and decay
snn_model = SimpleSNN(input_size, output_size, threshold=1.5, decay=0.9)
snn_model.eval()  # Ensure the SNN is not in training mode

# Step 3: Get SNN Outputs
with torch.no_grad():
    snn_outputs = snn_model(input_sequences)
    # No decoding needed as outputs are already in binary form

# Step 4: Define the Surrogate RNN Model
class IdentityRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(IdentityRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, nonlinearity='relu')
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()  # To constrain outputs between 0 and 1

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out)
        out = self.sigmoid(out)
        return out

# Step 5: Train the Surrogate RNN Model
train_inputs = input_sequences  # Shape: (batch_size, time_steps, input_size)
train_targets = snn_outputs     # SNN outputs as targets

surrogate_model = IdentityRNN(input_size, hidden_size, output_size)
criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary outputs
optimizer = torch.optim.Adam(surrogate_model.parameters(), lr=0.001)
num_epochs = 5000

for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = surrogate_model(train_inputs)
    loss = criterion(outputs, train_targets)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 500 == 0:
        print(f"Surrogate Model Training Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")

# Step 6: Define the Desired Outputs
# For identity mapping, let's use a sequence of all ones
desired_outputs = torch.zeros(1, time_steps, output_size)
desired_outputs[0, ::5, :] = 1  # Set every 5th time step to 1

# Step 7: Initialize Inputs to be Optimized
optimized_inputs = torch.rand(1, time_steps, input_size, requires_grad=True)

# Step 8: Input Optimization Using the Surrogate Model
input_optimizer = torch.optim.Adam([optimized_inputs], lr=0.01)
num_input_epochs = 5000
criterion = nn.MSELoss()

lambda_reg = 0.01  # Regularization strength

for epoch in range(num_input_epochs):
    input_optimizer.zero_grad()
    surrogate_outputs = surrogate_model(optimized_inputs)
    loss = criterion(surrogate_outputs, desired_outputs)
    # Add regularization term
    reg_loss = lambda_reg * torch.mean(optimized_inputs)
    total_loss = loss + reg_loss
    total_loss.backward()
    input_optimizer.step()
    with torch.no_grad():
        optimized_inputs.clamp_(0, 1)  # Ensure inputs are within valid range
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{num_input_epochs}], Loss: {loss.item():.6f}, Reg Loss: {reg_loss.item():.6f}")

# Step 9: Prepare Inputs for the SNN
# Threshold the optimized inputs to get binary inputs for the SNN
binary_optimized_inputs = (optimized_inputs.detach() >= 0.5).float()

# Step 10: Test the Optimized Inputs on the Fixed SNN
with torch.no_grad():
    snn_outputs_test = snn_model(binary_optimized_inputs)

# Step 11: Evaluation and Visualization
desired_outputs_np = desired_outputs.numpy()
snn_outputs_np = snn_outputs_test.numpy()

# Compute Accuracy
accuracy = np.mean(snn_outputs_np == desired_outputs_np)
print(f"Accuracy of SNN Output: {accuracy * 100:.2f}%")

# Plot the Results
for i in range(output_size):
    plt.figure(figsize=(10, 2))
    plt.plot(range(time_steps), desired_outputs_np[0, :, i], label='Desired Output', linestyle='--', marker='o')
    plt.plot(range(time_steps), snn_outputs_np[0, :, i], label='SNN Output', linestyle='-', marker='x')
    plt.title(f'Identity Mapping with Reduced Spiking - Output Dimension {i+1}')
    plt.xlabel('Time Step')
    plt.ylabel('Output Value')
    plt.legend()
    plt.show()
