import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Parameters
input_size = 10
output_size = 5
sequence_length = 15
batch_size = 1000 # Number of samples for training the surrogate model

torch.manual_seed(42)  # For reproducibility

# Define the target RNN model (unknown parameters)
class TargetRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TargetRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out)
        return out

hidden_size = 20
target_rnn = TargetRNN(input_size, hidden_size, output_size)

# Generate random inputs
inputs = torch.randn(batch_size, sequence_length, input_size)
with torch.no_grad():
    outputs = target_rnn(inputs)

# Convert data to numpy arrays for kernel estimation
inputs_np = inputs.numpy()
outputs_np = outputs.numpy()

# Step 1: Kernel Estimation
# We will estimate first and second-order Volterra kernels

# Flatten the data
X = inputs_np.reshape(-1, input_size)  # Shape: (batch_size * sequence_length, input_size)
Y = outputs_np.reshape(-1, output_size)  # Shape: (batch_size * sequence_length, output_size)

# Number of data points
N = X.shape[0]

# Estimate first-order kernel (linear term)
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, Y)
h1 = lin_reg.coef_.T  # Shape: (input_size, output_size)


# Estimate second-order kernel (quadratic term)
# For computational tractability, we'll assume the kernels are diagonal

# Prepare data for second-order kernel estimation
X_quad = X ** 2  # Element-wise square

# Stack linear and quadratic terms
X_combined = np.hstack((X, X_quad))  # Shape: (N, 2 * input_size)

# Fit linear regression on combined data
lin_reg_quad = LinearRegression()
lin_reg_quad.fit(X_combined, Y)
coefficients = lin_reg_quad.coef_.T  # Shape: (2 * input_size, output_size)

# Extract kernels (corrected slicing)
h1 = coefficients[:input_size, :]       # Shape: (input_size, output_size)
h2_diag = coefficients[input_size:, :]  # Shape: (input_size, output_size)

# Step 2: Implement Volterra Series Surrogate Model
class VolterraSurrogate(nn.Module):
    def __init__(self, input_size, output_size, h1, h2_diag):
        super(VolterraSurrogate, self).__init__()
        self.h1 = torch.tensor(h1, dtype=torch.float32)        # Shape: (input_size, output_size)
        self.h2_diag = torch.tensor(h2_diag, dtype=torch.float32)  # Shape: (input_size, output_size)
    
    def forward(self, x):
        x_flat = x.view(-1, input_size)  # Shape: (batch_size * sequence_length, input_size)
        y_lin = torch.matmul(x_flat, self.h1)                  # Shape: (batch_size * sequence_length, output_size)
        y_quad = torch.matmul(x_flat ** 2, self.h2_diag)       # Same shape
        y = y_lin + y_quad
        y = y.view(x.size(0), x.size(1), -1)
        return y

# Instantiate the surrogate model
surrogate_model = VolterraSurrogate(input_size, output_size, h1, h2_diag)

# Move models to the appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
surrogate_model.to(device)
target_rnn.to(device)

# Step 3: Input Optimization
# Generate desired outputs (e.g., sine wave)
def generate_desired_output(batch_size, sequence_length, output_size):
    t = np.linspace(0, 2 * np.pi, sequence_length)
    desired_output = np.zeros((batch_size, sequence_length, output_size))
    for i in range(output_size):
        desired_output[:, :, i] = np.sin(t + i * np.pi / 4)
    return desired_output

desired_outputs_np = generate_desired_output(100, sequence_length, output_size)
desired_outputs = torch.tensor(desired_outputs_np, dtype=torch.float32).to(device)

# Initialize inputs to be optimized
optimized_inputs = torch.randn(100, sequence_length, input_size, requires_grad=True, device=device)

# Define optimizer for inputs
input_optimizer = optim.Adam([optimized_inputs], lr=0.01)
num_input_epochs = 1000
criterion = nn.MSELoss()

for epoch in range(num_input_epochs):
    input_optimizer.zero_grad()
    surrogate_outputs = surrogate_model(optimized_inputs)
    loss = criterion(surrogate_outputs, desired_outputs)
    loss.backward()
    input_optimizer.step()
    # Optionally clamp inputs to a reasonable range
    with torch.no_grad():
        optimized_inputs.clamp_(-3.0, 3.0)
    if (epoch + 1) % 100 == 0:
        print(f"Input Optimization Epoch [{epoch+1}/{num_input_epochs}], Loss: {loss.item():.6f}")

# Step 4: Test on the Target RNN
with torch.no_grad():
    target_outputs = target_rnn(optimized_inputs)

# Step 5: Visualization
# Convert tensors to numpy arrays for plotting
desired_outputs_np = desired_outputs.cpu().numpy()[0]
surrogate_outputs_np = surrogate_outputs.cpu().detach().numpy()[0]
target_outputs_np = target_outputs.cpu().numpy()[0]

for i in range(output_size):
    plt.figure(figsize=(10, 4))
    plt.plot(desired_outputs_np[:, i], label='Desired Output', linestyle='--')
    plt.plot(surrogate_outputs_np[:, i], label='Surrogate Model Output')
    plt.plot(target_outputs_np[:, i], label='Target RNN Output')
    plt.title(f'Output Dimension {i+1}')
    plt.xlabel('Time Step')
    plt.ylabel('Output Value')
    plt.legend()
    plt.show()

# Evaluate Performance

print(criterion(desired_outputs,target_outputs).item())
