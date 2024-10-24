import torch
import torch.nn as nn
import torch.optim as optim

# Define the RNN model for both target and surrogate
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, h0=None):
        out, hn = self.rnn(x, h0)
        out = self.fc(out)
        return out, hn

# Step 1: Initialize RNN with unknown parameters to represent the target process
input_size = 10
hidden_size = 20
output_size = 5
sequence_length = 15
batch_size = 3

# Initialize the target RNN with unknown parameters
target_rnn = RNNModel(input_size, hidden_size, output_size)

# Step 2: Sample inputs/outputs from the target process
inputs = torch.randn(sequence_length, batch_size, input_size)
inputs.requires_grad = True  # Enable gradient tracking for inputs

# Get outputs from the target process
with torch.no_grad():
    target_outputs, _ = target_rnn(inputs.clone())

# Step 3: Initialize surrogate RNN with model assumptions, but random parameters
surrogate_rnn = RNNModel(input_size, hidden_size, output_size)

# Step 4: Match surrogate to target using loss between surrogate outputs and target outputs
criterion = nn.MSELoss()
optimizer = optim.Adam(surrogate_rnn.parameters(), lr=0.001)

# Training loop for surrogate model
num_epochs = 1000
for epoch in range(num_epochs):
    # Zero gradients
    optimizer.zero_grad()
    if inputs.grad is not None:
        inputs.grad.zero_()
    
    # Forward pass
    surrogate_outputs, _ = surrogate_rnn(inputs)
    
    # Compute loss
    loss = criterion(surrogate_outputs, target_outputs)
    
    # Backward pass
    loss.backward()
    
    # Update parameters
    optimizer.step()
    
    if (epoch+1) % 200 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")


inputs_test = torch.randn(sequence_length, batch_size, input_size)
inputs_test.requires_grad = True

# Step 5: Sample gradients for both target and surrogate
# Compute gradients of outputs with respect to inputs for the target model
inputs_test.grad = None  # Reset gradients
target_outputs, _ = target_rnn(inputs_test)
target_scalar_output = target_outputs.sum()
target_scalar_output.backward(retain_graph=True)
target_inputs_grad = inputs_test.grad.clone()

# Compute gradients of outputs with respect to inputs for the surrogate model
inputs_test.grad.zero_()
surrogate_outputs, _ = surrogate_rnn(inputs_test)
surrogate_scalar_output = surrogate_outputs.sum()
surrogate_scalar_output.backward()
surrogate_inputs_grad = inputs_test.grad.clone()

# Step 6: Compare the dot product of these gradients
# Flatten gradients
target_inputs_grad_flat = target_inputs_grad.view(-1)
surrogate_inputs_grad_flat = surrogate_inputs_grad.view(-1)

# Compute dot product
dot_product = torch.dot(target_inputs_grad_flat, surrogate_inputs_grad_flat)
print("\nDot product of gradients:", dot_product.item())

# Check the sign of the dot product
if dot_product.item() > 0:
    print("Gradients are in the same direction.")
elif dot_product.item() == 0:
    print("Gradients are orthogonal.")
else:
    print("Gradients are in different directions.")
