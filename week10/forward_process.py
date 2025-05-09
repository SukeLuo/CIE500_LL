# %%
import torch
import torch.nn as nn
import torch.nn.functional as F


# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(2, 3)  # Input layer to hidden layer
        self.fc2 = nn.Linear(3, 1)  # Hidden layer to output layer

    def forward(self, x):
        x = self.fc1(x)  # Linear transformation
        x = F.relu(x)  # Non-linearity
        x = self.fc2(x)  # Output layer
        return x


# Instantiate the model
model = SimpleNet()

# Create a dummy input tensor (batch of 2 samples, each with 4 features)
input_tensor = torch.tensor([1.0, 2.0])

# Perform a forward pass
output = model(input_tensor)

print("Input:\n", input_tensor)
print("Output:\n", output)

print("Weights of fc1:\n", model.fc1.weight.data)
print("Biases of fc1:\n", model.fc1.bias.data)
