import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Define the target RNN model (unknown parameters)
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
            
    def forward(self, x, h0=None):
        out, hn = self.rnn(x, h0)
        out = self.fc(out)
        return out, hn

# Parameters
input_size = 10
hidden_size = 50  # Increased hidden size for better capacity
output_size = 5
sequence_length = 15
batch_size = 200  # Increased batch size for better training
num_layers = 2    # Increased number of layers for better capacity

# Step 1: Data Preparation
torch.manual_seed(42)  # For reproducibility

# Initialize the target RNN model (unknown parameters)
target_model = RNNModel(input_size, hidden_size, output_size, num_layers)

# Generate random inputs and get target outputs
inputs = torch.randn(batch_size, sequence_length, input_size)
with torch.no_grad():
    target_outputs, _ = target_model(inputs)

# Move data to the appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
inputs = inputs.to(device)
target_outputs = target_outputs.to(device)

# Step 2: Implementing the LSTM Surrogate Model
class SurrogateModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(SurrogateModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, output_size)
        
    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        x = self.fc1(out)
        out = self.fc2(x)
        return out, (hn, cn)

# Instantiate the surrogate model
surrogate_model = SurrogateModel(input_size, hidden_size, output_size, num_layers).to(device)

# Step 3: Training the Surrogate Model
criterion = nn.MSELoss()
optimizer = optim.Adam(surrogate_model.parameters(), lr=0.001)
num_epochs = 1000  # Increased number of epochs

surrogate_model.train()
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs_pred, _ = surrogate_model(inputs)
    loss = criterion(outputs_pred, target_outputs)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")

# Step 4: Optimize Inputs Using the Surrogate Model
# Generate desired outputs (sinusoidal function)
def generate_desired_output(batch_size, sequence_length, output_size):
    t = torch.linspace(0, 2 * torch.pi, sequence_length)
    desired_output = torch.zeros(batch_size, sequence_length, output_size)
    for i in range(output_size):
        # Create sine waves with different frequencies and phases
        desired_output[:, :, i] = torch.sin(t + i * torch.pi / 4)
    return desired_output

desired_outputs = generate_desired_output(100, sequence_length, output_size).to(device)

# Initialize inputs to be optimized
optimized_inputs = torch.randn(100, sequence_length, input_size, requires_grad=True, device=device)

# Define optimizer for inputs
input_optimizer = optim.Adam([optimized_inputs], lr=0.1)
num_input_epochs = 2000 # Increased number of epochs for input optimization
criterion = nn.MSELoss()

for epoch in range(num_input_epochs):
    input_optimizer.zero_grad()
    surrogate_outputs, _ = surrogate_model(optimized_inputs)
    loss = criterion(surrogate_outputs, desired_outputs)
    # Add regularization to keep inputs within reasonable bounds
    lambda_reg = 0
    reg_loss = lambda_reg * torch.norm(optimized_inputs)
    total_loss = loss + reg_loss
    total_loss.backward()
    input_optimizer.step()
    # # Clamp inputs to a reasonable range
    # with torch.no_grad():
    #     optimized_inputs.clamp_(-1.0, 1.0)
    if (epoch + 1) % 100 == 0:
        print(f"Input Optimization Epoch [{epoch+1}/{num_input_epochs}], Loss: {loss.item():.6f}")

# Step 5: Run the Optimized Inputs Through the Target RNN
with torch.no_grad():
    target_model.to(device)
    target_model_outputs, _ = target_model(optimized_inputs)

# Step 6: Visualization
# Convert tensors to numpy arrays for plotting
desired_outputs_np = desired_outputs.cpu().detach().numpy()[0]
surrogate_outputs_np = surrogate_outputs.cpu().detach().numpy()[0]
target_outputs_np = target_model_outputs.cpu().detach().numpy()[0]

for i in range(output_size):
    plt.figure(figsize=(10, 4))
    plt.plot(desired_outputs_np[:, i], label='Desired Output', linestyle='--')
    plt.plot(surrogate_outputs_np[:, i], label='Surrogate Model Output')
    plt.plot(target_outputs_np[:, i], label='Target Model Output')
    plt.title(f'Output Dimension {i+1}')
    plt.xlabel('Time Step')
    plt.ylabel('Output Value')
    plt.legend()
    plt.show()
