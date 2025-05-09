'''''
# %%
# Basic setting

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

# Note: Each of these 10 feature variables have been mean centered and scaled by the standard deviation times the square root of n_samples (i.e. the sum of squares of each column totals 1).
# https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset

diabetes = load_diabetes()
# print(diabetes)
# print(diabetes.keys())
print(f'Factors of Diabetes: {diabetes['feature_names']}')

X = diabetes.data
y = diabetes.target
print(X[:5,:])
print(y[:5])
print(type(X))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

class DiabetesNetwork(nn.Module):
    def __init__(self):
        super(DiabetesNetwork, self).__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = DiabetesNetwork()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 500
losses = []

for i in range(epochs):
    y_pred = model(X_train)
    loss = torch.sqrt(criterion(y_pred, y_train)) # RMSE
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i+1) % 10 == 0:
        print(f'Epoch {(i+1)}/{epochs}, Loss = {loss.item():.4f}')

with torch.no_grad():
    y_val = model(X_test)
    val_loss = torch.sqrt(criterion(y_val, y_test))
    print(f'Epoch {i+1}, Training Loss = {loss.item():.4f}, Validation Loss = {val_loss.item():.4f}')

'''''
# %%
# batch and standarization were added into the codes.

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

diabetes = load_diabetes()
print(f'Factors of Diabetes: {diabetes['feature_names']}')

X = diabetes.data
y = diabetes.target
print(X[:5,:])
print(y[:5])
print(type(X))

# train_data, test_data = train_test_split(diabetes, test_size=0.2, random_state=42)
# train_loader = DataLoader(train_data,batch_size=10, shuffle=True)
# test_loader = DataLoader(test_data,batch_size=10, shuffle=False)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # Ensure y is a column vector
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Convert the data into PyTorch Geometric Data objects
train_data = [Data(x=X_train[i].view(1, -1), y=y_train[i].view(1, -1)) for i in range(len(X_train))]
test_data = [Data(x=X_test[i].view(1, -1), y=y_test[i].view(1, -1)) for i in range(len(X_test))]

# Create DataLoaders
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10, shuffle=False)

class DiabetesNetwork(nn.Module):
    def __init__(self):
        super(DiabetesNetwork, self).__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = DiabetesNetwork()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 500
train_losses = []
test_losses = []

def train(data):
    model.train()
    optimizer.zero_grad()
    y_pred = model(data.x)
    loss = torch.sqrt(criterion(y_pred, data.y))
    loss.backward()
    optimizer.step()
    return loss.item()

def test(data):
    model.eval()
    y_val = model(data.x)
    test_loss = torch.sqrt(criterion(y_val, data.y))
    return test_loss.item()

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    for i, x_train in enumerate(train_loader):
        batch_loss = train(x_train)
        epoch_loss += batch_loss
    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    with torch.no_grad():
        test_epoch_loss = 0.0
        for x_val in test_loader:
            test_loss = test(x_val)
            test_epoch_loss += test_loss
    avg_test_loss = test_epoch_loss / len(test_loader)
    test_losses.append(avg_test_loss)

    print(f'Epoch {epoch+1}/500, Training Loss = {avg_train_loss:.4f}, Test Loss = {avg_test_loss:.4f}')

# Standardize the output as well