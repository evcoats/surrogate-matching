import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# Define the RNN model class
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, h0=None):
        # x shape: (batch_size, seq_length, input_size)
        out, hn = self.rnn(x, h0)  # out: (batch_size, seq_length, hidden_size)
        out = self.fc(out)         # out: (batch_size, seq_length, output_size)
        return out, hn

# Function to create inputs
def create_inputs(sequence_length, batch_size, input_size):
    inputs = torch.randn(batch_size, sequence_length, input_size)
    return inputs

# Function to initialize the target RNN
def target_rnn(input_size, hidden_size, output_size, num_layers=1):
    # Initialize the target RNN with random parameters
    model = RNNModel(input_size, hidden_size, output_size, num_layers)
    return model

# Function to initialize the surrogate RNN
def surrogate_rnn(input_size, hidden_size, output_size, num_layers=1):
    # Initialize the surrogate RNN with random parameters
    model = RNNModel(input_size, hidden_size, output_size, num_layers)
    return model

# Define the loss criterion
def criterion():
    return nn.MSELoss()

# Example usage:
def run_matching_perturbed():
    # Parameters
    input_size = 10
    hidden_size = 20
    output_size = 5
    sequence_length = 15
    batch_size = 3
    num_layers = 1
    
    # Create inputs
    inputs = create_inputs(sequence_length, batch_size, input_size)
    
    # Initialize models
    target_model = target_rnn(input_size, hidden_size, output_size, num_layers)
    surrogate_model = surrogate_rnn(input_size, hidden_size, output_size, num_layers)
    
    # Initialize loss function
    loss_fn = criterion()
        
    # Training loop with combined methods
    lambda_reg = 0   # Regularization coefficient
    lambda_diff = 10   # Difference loss coefficient
    noise_std = 0.01    # Input noise standard deviation
    epsilon = 1e-4      # Perturbation magnitude

    num_epochs = 1000
    learning_rate = 0.001
    optimizer = torch.optim.Adam(surrogate_model.parameters(), lr=learning_rate)


    for param in surrogate_model.parameters():
        param.requires_grad = True

    for param in target_model.parameters():
        param.requires_grad = True


    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        
        # Add noise to inputs
        inputs_noisy = inputs 
        
        # noise_std * torch.randn_like(inputs)

        inputs_noisy.requires_grad = True
        
        # Get outputs from surrogate model
        surrogate_outputs, _ = surrogate_model(inputs_noisy)
        target_outputs, _ = target_model(inputs_noisy)
        loss = loss_fn(surrogate_outputs, target_outputs)
        
        # Compute surrogate gradients w.r.t inputs
        surrogate_outputs_flat = surrogate_outputs.view(-1)
        grad_outputs = torch.ones_like(surrogate_outputs_flat)
        surrogate_gradients = torch.autograd.grad(
            outputs=surrogate_outputs_flat,
            inputs=inputs_noisy,
            grad_outputs=grad_outputs,
            create_graph=True
        )[0]
        
        # Gradient regularization
        grad_norm = surrogate_gradients.norm()
        
        # Implicit gradient alignment via loss function
        inputs_perturbed = inputs_noisy + epsilon * torch.randn_like(inputs_noisy)
        with torch.no_grad():
            target_outputs_orig, _ = target_model(inputs_noisy)
            target_outputs_perturbed, _ = target_model(inputs_perturbed)
        surrogate_outputs_perturbed, _ = surrogate_model(inputs_perturbed)
        
        target_differences = target_outputs_perturbed - target_outputs_orig
        surrogate_differences = surrogate_outputs_perturbed - surrogate_outputs
        difference_loss = loss_fn(surrogate_differences, target_differences)
        
        # Total loss
        total_loss = loss + lambda_reg * grad_norm + lambda_diff * difference_loss
        total_loss.backward()
        optimizer.step()
        
        if (epoch+1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss.item():.6f}")

        # Ensure inputs require gradients
    inputs.requires_grad = True

    # Compute gradients for the target model
    target_outputs, _ = target_model(inputs)
    target_scalar_output = target_outputs.sum()
    target_scalar_output.backward(retain_graph=True)
    target_gradients = inputs.grad.clone()

    # Reset gradients
    inputs.grad.zero_()

    # Compute gradients for the surrogate model
    surrogate_outputs, _ = surrogate_model(inputs)
    surrogate_scalar_output = surrogate_outputs.sum()
    surrogate_scalar_output.backward()
    surrogate_gradients = inputs.grad.clone()

    # Compare gradients
    print(target_gradients.view(-1).shape)

    dot_product = torch.dot(target_gradients.view(-1), surrogate_gradients.view(-1))
    print("Dot Product of Gradients:", dot_product.item())
    if dot_product.item() > 0:
        print("Gradients are in the same direction.")
    elif dot_product.item() == 0:
        print("Gradients are orthogonal.")
    else:
        print("Gradients are in opposite directions.")

    target_grad_flat = target_gradients.view(-1)
    surrogate_grad_flat = surrogate_gradients.view(-1)

    # Compute cosine similarity
    cosine_sim = F.cosine_similarity(target_grad_flat, surrogate_grad_flat, dim=0)
    print("Cosine Similarity:", cosine_sim.item())

    norm_product = torch.norm(target_grad_flat) * torch.norm(surrogate_grad_flat)
# Clamp the cosine value to the valid range [-1, 1]
    cosine_of_angle = torch.clamp(dot_product / norm_product, -1.0, 1.0)

    # Compute angle in radians
    angle_rad = torch.acos(cosine_of_angle)
    # Convert to degrees
    angle_deg = torch.rad2deg(angle_rad)

    print("Angle between gradients (degrees):", angle_deg.item())

    return (dot_product,cosine_sim,angle_deg)







            

    
