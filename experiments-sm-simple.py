from surrogate_matching_simple import run_matching_simple
import itertools
import numpy as np

# Hyperparameters to optimize
learning_rates = [0.001]
batch_sizes = [1]
num_epochs_list = [1000]
surrogate_hidden_sizes = [10, 20, 30, 40, 100]

# Constants
input_size = 10
hidden_size = 20
output_size = 5
sequence_length = 15
inputs_num_run = 300
num_runs_per_combination = 40

# Store results
results = []

# Hyperparameter optimization loop
for lr, batch_size, num_epochs, surrogate_hidden_size in itertools.product(learning_rates, batch_sizes, num_epochs_list, surrogate_hidden_sizes):
    # Run the matching function with current hyperparameters
    dp = 0
    cos = 0
    angl = 0

    for i in range(num_runs_per_combination):
        dot_product_sum, cosine_similarity, angle = run_matching_simple(
            input_size=input_size,
            hidden_size=hidden_size,
            surrogate_hidden_size = surrogate_hidden_size,
            output_size=output_size,
            sequence_length=sequence_length,
            inputs_num_run=inputs_num_run,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=lr
        )

        dp+=dot_product_sum
        cos += cosine_similarity
        angl += angle
    dp /= num_runs_per_combination
    cos /= num_runs_per_combination
    angl /= num_runs_per_combination
    
    # Store the results
    results.append({
        'learning_rate': lr,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'dot_product_sum': dp,
        'surrogate_size': surrogate_hidden_size,
        'cosine_similarity': cos,
        'angle': angl
    })
    
    # Print progress
    print(f"lr: {lr}, batch_size: {batch_size}, num_epochs: {num_epochs}, "
          f"cosine_similarity: {cos:.4f}, surrogate_size:{surrogate_hidden_size},angle: {angl:.2f} degrees")

# Find the hyperparameters that maximize cosine similarity
best_result = max(results, key=lambda x: x['cosine_similarity'])

print("\nBest Hyperparameters:")
print(f"Learning Rate: {best_result['learning_rate']}")
print(f"Batch Size: {best_result['batch_size']}")
print(f"Surrogate Size: {best_result['surrogate_size']}")
print(f"Number of Epochs: {best_result['num_epochs']}")
print(f"Cosine Similarity: {best_result['cosine_similarity']:.4f}")
print(f"Angle between Gradients: {best_result['angle']:.2f} degrees")
