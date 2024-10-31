import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Define the RNN model class
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, h0=None):
        out, hn = self.rnn(x, h0)  # out: (batch_size, seq_length, hidden_size)
        out = self.fc(out)         # out: (batch_size, seq_length, output_size)
        return out, hn
    
def closedLoopSinusoid(batch_size_initial,batch_size_inputs_model):
    # Parameters
    input_size = 10
    hidden_size = 20
    output_size = 5
    sequence_length = 15
    num_layers = 1

    # Step 1: Initialize target and surrogate models
    torch.manual_seed(42)  # For reproducibility
    target_model = RNNModel(input_size, hidden_size, output_size, num_layers)
    surrogate_model = RNNModel(input_size, hidden_size, output_size, num_layers)

    # Step 2: Train the surrogate model to match the target model
    # Generate random inputs and get target outputs
    inputs = torch.randn(batch_size_initial, sequence_length, input_size)
    with torch.no_grad():
        target_outputs, _ = target_model(inputs)

    # Training the surrogate model
    criterion = nn.MSELoss()
    optimizer = optim.Adam(surrogate_model.parameters(), lr=0.001)
    num_epochs = 2000

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs, _ = surrogate_model(inputs)
        loss = criterion(outputs, target_outputs)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 200 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")

    # Step 3: Define the desired outputs (e.g., sinusoid function)
    def generate_desired_output(batch_size_inputs_model, sequence_length, output_size):
        t = torch.linspace(0, 2 * torch.pi, sequence_length)
        desired_output = torch.zeros(batch_size_inputs_model, sequence_length, output_size)
        for i in range(output_size):
            # Create sine waves with different frequencies and phases
            desired_output[0, :, i] = torch.sin(t + i * torch.pi / 4)
        return desired_output

    desired_outputs = generate_desired_output(batch_size_inputs_model, sequence_length, output_size)

    # Step 4: Optimize inputs using the surrogate model
    # Initialize inputs (starting from random values or zeros)
    optimized_inputs = torch.randn(batch_size_inputs_model, sequence_length, input_size, requires_grad=True)

    # Define optimizer for inputs
    input_optimizer = optim.Adam([optimized_inputs], lr=0.01)
    num_input_epochs = 2000

    for epoch in range(num_input_epochs):
        input_optimizer.zero_grad()
        surrogate_outputs, _ = surrogate_model(optimized_inputs)
        loss = criterion(surrogate_outputs, desired_outputs)
        loss.backward()
        input_optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f"Input Optimization Epoch [{epoch+1}/{num_input_epochs}], Loss: {loss.item():.6f}")

    # Step 5: Run the optimized inputs through the target model
    with torch.no_grad():
        target_model_outputs, _ = target_model(optimized_inputs)

    # Step 6: Visualization
    import matplotlib.pyplot as plt

    # Convert tensors to numpy arrays for plotting
    desired_outputs_np = desired_outputs.detach().numpy()[0]
    surrogate_outputs_np = surrogate_outputs.detach().numpy()[0]
    target_outputs_np = target_model_outputs.detach().numpy()[0]

    print(criterion(desired_outputs,target_model_outputs).item())


    # Plot the outputs
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


closedLoopSinusoid(batch_size_initial=1,batch_size_inputs_model=10)
closedLoopSinusoid(batch_size_initial=10,batch_size_inputs_model=10)
closedLoopSinusoid(batch_size_initial=100,batch_size_inputs_model=10)
closedLoopSinusoid(batch_size_initial=1000,batch_size_inputs_model=10)
