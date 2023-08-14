import numpy as np

def dropout_layer(x, dropout_rate):
    # Create a random binary mask with the same shape as x
    dropout_mask = np.random.rand(*x.shape) > dropout_rate
    # Apply the mask to the input, scaling it by the inverse of the dropout rate
    return x * dropout_mask / (1 - dropout_rate)

def dropconnect_layer(weights, input_data, dropconnect_rate):
    # Create a random binary mask with the same shape as the weights
    dropconnect_mask = np.random.rand(*weights.shape) > dropconnect_rate
    # Apply the mask to the weights
    masked_weights = weights * dropconnect_mask
    # Perform a matrix multiplication with the masked weights and input data
    return np.dot(input_data, masked_weights)


# Example usage for dropout
input_data = np.array([[0.1, 0.5, 0.2], 
                      [0.8, 0.6, 0.7], 
                      [0.9, 0.3, 0.4]])
dropout_rate = 0.5
output_data = dropout_layer(input_data, dropout_rate)
print(output_data)

# Example usage for drop-connect
weights = np.random.randn(3, 4)
print(weights)
print("\n")
dropconnect_rate = 0.5
output_data=  dropconnect_layer(weights, input_data, dropconnect_rate)
print(output_data)