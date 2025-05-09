#%%
import wntr
import torch
import numpy as np
import torch.nn as nn
from torch_geometric.nn import GCNConv, NNConv
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

inp_file = "/Users/sukeluo/Documents/UB/Courses/25Spring/CIE500/CIE500_LL/week7/Net3.inp"
wn = wntr.network.WaterNetworkModel(inp_file)
wn.options.time.report_timestep = 3600*2
wn.options.time.duration = 3600*24*3

sim = wntr.sim.WNTRSimulator(wn)
results = sim.run_sim()

# Extract node pressure, flowrate
pressure = results.node["pressure"]
Q_flowrate = results.link["flowrate"]

# Extract nodes and edges
nodes = list(wn.node_name_list)
node_mapping = {name: idx for idx, name in enumerate(nodes)}
inv_node_mapping = {idx: name for name, idx in node_mapping.items()}

# Filter to only include pipes (exclude pumps, valves, etc.)
pipes = [name for name, link in wn.links() if link.link_type == 'Pipe']
pipe_mapping = {name: idx for idx, name in enumerate(pipes)}
inv_pipe_mapping = {idx: name for name, idx in pipe_mapping.items()}

print(f"Total links in network: {len(wn.link_name_list)}")
print(f"Number of pipes (excluding pumps/valves): {len(pipes)}")

# First collect all roughness values
roughness_label = []
for idx, (link_name, link) in enumerate(wn.links()):
    if link.link_type == 'Pipe':
        roughness_label.append(link.roughness)

print("\nOriginal roughness values:")
print(f"First 5 values: {roughness_label[:5]}")
print(f"Last 5 values: {roughness_label[-5:]}")

# Now modify the roughness values for known pipes

# Modify roughness values for known pipes to create more variance
modified_roughness = roughness_label.copy()
for idx, coeff in enumerate(modified_roughness):
    # 90% chance to modify
    r = np.random.random()
    if r < 0.9:
        # Create ranges with more variance but still somewhat realistic
        if r < 0.3:  # 30% chance to be lower range
            modified_roughness[idx] = np.random.uniform(100, 130)  # Lower range
        elif r < 0.7:  # 40% chance to be middle range
            modified_roughness[idx] = np.random.uniform(130, 150)  # Middle range
        else:  # 30% chance to be higher range
            modified_roughness[idx] = np.random.uniform(150, 180)  # Higher range

print("\nModified roughness values:")
print(f"First 5 values: {modified_roughness[:5]}")
print(f"Last 5 values: {modified_roughness[-5:]}")

print("\nRoughness distribution before modification:")
print(f"Min: {min(roughness_label):.2f}")
print(f"Max: {max(roughness_label):.2f}")
print(f"Mean: {np.mean(roughness_label):.2f}")
print(f"Std: {np.std(roughness_label):.2f}")

print("\nRoughness distribution after modification:")
print(f"Min: {min(modified_roughness):.2f}")
print(f"Max: {max(modified_roughness):.2f}")
print(f"Mean: {np.mean(modified_roughness):.2f}")
print(f"Std: {np.std(modified_roughness):.2f}")

# Use modified roughness for known pipes
roughness_label = modified_roughness

#%%
edges = []
edge_attrs = []
# Loop through each link in the model
for idx, (link_name, link) in enumerate(wn.links()):
    if link.link_type == 'Pipe':
        src_idx = node_mapping[link.start_node_name]
        dst_idx = node_mapping[link.end_node_name]
        length = link.length
        diameter = link.diameter
        roughness = roughness_label[idx]  # Use the modified roughness
        flowrate = abs(Q_flowrate[link_name].mean())
        
        if flowrate > 0 and diameter > 0 and roughness > 0:
            # Hazen-Williams for head loss calculation
            headloss = (10.67 * length * flowrate**1.852) / (roughness**1.852 * diameter**4.87)
        else:
            headloss = 0.0
        # Append edge
        edges.append((src_idx, dst_idx))
        # If an undirected graph, append (dst_idx, src_idx)
        # Append edge attributes
        edge_attrs.append([length, diameter, flowrate, headloss])

#%%
# Convert edge attributes to a PyTorch tensor (shape = [num_edges, num_edge_features])
edge_attr = torch.tensor(np.array(edge_attrs), dtype=torch.float)

# Add elevation,demand as node features
elevations = []
demands = []
for node in nodes:
    node_obj = wn.get_node(node)
    if node_obj.node_type == "Junction":
        ele = node_obj.elevation
        if len(node_obj.demand_timeseries_list) > 0:
            # Use the base value of the first demand pattern
            dem = node_obj.demand_timeseries_list[0].base_value
        else:
            # If no demand pattern exists, default to 0
            dem = 0.0
    elif node_obj.node_type == "Tank":
        ele = node_obj.elevation
    elif node_obj.node_type == "Reservoir":
        # a Reservoir node typically does not store its head as a direct numeric attribute (node_obj.head)
        ele = node_obj.head_timeseries.base_value
    else:
        ele = 0  # or some default
        dem = 0.0

    # print(node, node_obj.node_type, ele)  # Debug print if there is any "None" value
    elevations.append(ele)
    demands.append(dem)
#%%
# Convert to a PyTorch tensor
elevation_tensor = torch.tensor(elevations, dtype=torch.float).view(-1, 1) # Shape: (num_nodes, 1)
demand_tensor = torch.tensor(demands, dtype=torch.float).view(-1, 1)
node_features = torch.cat([elevation_tensor, demand_tensor], dim=1)

# Make edge attr into node fea, while node fea into edge attr in data object

# Convert edge list to a PyTorch tensor (shape = [2, num_edges])
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() # Original edge index

# Create new edge_index for pipe connections
pipe_edges = []
for i, (src, dst) in enumerate(edges):
    # Get all pipes connected to source node
    for j, (src2, dst2) in enumerate(edges):
        if i != j:  # Don't connect pipe to itself
            # Check if pipes share a node
            if (src2 == src or dst2 == src or src2 == dst or dst2 == dst):
                pipe_edges.append((i, j))

# Remove duplicate edges
pipe_edges = list(set(pipe_edges)) # undirected graph

# Convert to tensor
pipe_edge_index = torch.tensor(pipe_edges, dtype=torch.long).t().contiguous()

print("\nEdge index shapes:")
print(f"Original edge_index shape: {edge_index.shape}")
print(f"New pipe_edge_index shape: {pipe_edge_index.shape}")
# Find both ends node features of each edge(which is new edge_attr)
edge_node_features = []
for i in pipe_edge_index[0]:
    pipe_name = inv_pipe_mapping[i.item()]
    pipe_obj = wn.get_link(pipe_name)
    src_node = pipe_obj.start_node_name
    dst_node = pipe_obj.end_node_name
    print(pipe_name, src_node, dst_node)
    src_node_idx = node_mapping[src_node]
    dst_node_idx = node_mapping[dst_node]
    src_node_feat = node_features[src_node_idx]
    dst_node_feat = node_features[dst_node_idx]
    print(src_node_feat, dst_node_feat)
    edge_node_features.append([src_node_feat, dst_node_feat])
edge_node_features = [torch.cat([a, b]).tolist() for a, b in edge_node_features]
edge_node_features = torch.tensor(edge_node_features, dtype=torch.float)
# Split source and target indices
src_nodes = edge_index[0] # start node, shape: [num_edges]
dst_nodes = edge_index[1] # end node

# Gather node features for start and end nodes of each edge
src_feats = node_features[src_nodes] # shape: [num_edges, num_node_features]
dst_feats = node_features[dst_nodes] # same shape

# Concatenate: [src_features || dst_features]
# edge_node_features = torch.cat([src_feats, dst_feats], dim=1) # shape: [num_edges, 2*num_node_features]

# Data validation checks
print(f"Number of pipes (nodes in GNN): {len(pipes)}")
print(f"Number of pipe connections (edges in GNN): {edge_index.shape[1]}")
print(f"Pipe feature shape (node features in GNN): {edge_attr.shape}")
print(f"Pipe connection feature shape (edge features in GNN): {edge_node_features.shape}")
print(f"Roughness label shape: {edge_attr.shape}")

# Create data object
print("\nData dimensions check:")
print(f"edge_attr shape: {edge_attr.shape}")
print(f"edge_index shape: {pipe_edge_index.shape}")
print(f"edge_index max index: {pipe_edge_index.max().item()}")
print(f"edge_node_features shape: {edge_node_features.shape}")
print(f"label shape: {edge_attr.shape}")

# Create a single data object for all pipes
data = Data(
    x=edge_attr,
    edge_index=pipe_edge_index,
    edge_attr=edge_node_features,
    y=torch.tensor(np.array(roughness_label).reshape(-1, 1), dtype=torch.float)
)

# Split data into training and validation sets
train_mask = torch.zeros(len(pipes), dtype=torch.bool)
train_indices = np.random.choice(len(pipes), int(0.7 * len(pipes)), replace=False)
train_mask[train_indices] = True
val_mask = ~train_mask
val_indices = np.setdiff1d(np.arange(len(pipes)), train_indices)

# --- Feature Normalization ---
scaler_feat = StandardScaler()
edge_attr_np = np.array(edge_attrs)
scaler_feat.fit(edge_attr_np[train_indices])  # Fit only on training data
edge_attr_norm = scaler_feat.transform(edge_attr_np)
edge_attr = torch.tensor(edge_attr_norm, dtype=torch.float)

# --- Label Normalization ---
scaler_label = StandardScaler()
label_np = np.array(roughness_label).reshape(-1, 1)
scaler_label.fit(label_np[train_indices])  # Fit only on training data
label_norm = scaler_label.transform(label_np)
label = torch.tensor(label_norm, dtype=torch.float)

# Use normalized features for edge_node_features
src_feats = node_features[src_nodes]
dst_feats = node_features[dst_nodes]
edge_node_features = torch.cat([src_feats, dst_feats], dim=1)
'''''
class RoughnessNNConvGNN(nn.Module):
    def __init__(self, in_channels, edge_in_channels, hidden_channels):
        super().__init__()
        # Kernel network: maps edge features to a weight matrix
        self.nn = nn.Sequential(
            nn.Linear(edge_in_channels, hidden_channels * in_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels * in_channels, hidden_channels * in_channels) # weight matrix for the edge
        )
        self.conv1 = NNConv(in_channels, hidden_channels, self.nn, aggr='mean')
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x
'''''
# Try the simplest neural network later
class RoughnessConvGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels1, hidden_channels2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels1)
        self.conv2 = GCNConv(hidden_channels1, hidden_channels2)
        # self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_channels2, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        # x = self.bn1(x)
        x = self.fc(x)
        return x


# Initialize NNConv model and training
# model = RoughnessNNConvGNN(
#     in_channels=edge_attr.shape[1],
#     edge_in_channels=edge_node_features.shape[1],
#     hidden_channels=64
# )
model = RoughnessConvGNN(in_channels=edge_attr.shape[1], hidden_channels1=64, hidden_channels2=128)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
criterion = nn.MSELoss()

# Training loop for NNConv model
best_val_loss = float('inf')
patience = 20
# patience_counter = 0
best_model_state = None

for epoch in range(1000):
    model.train()
    optimizer.zero_grad()
    output = model(edge_attr, data.edge_index)
    loss = criterion(output[train_mask], label[train_mask])
    loss.backward()
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    print(f'Epoch 1000/{epoch+1} training loss: {loss}')

    # Validation
    model.eval()
    with torch.no_grad():
        val_output = model(edge_attr, data.edge_index)
        val_loss = criterion(val_output[val_mask], label[val_mask])
        print(f'evaluation loss: {val_loss}')

        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     best_model_state = model.state_dict()
        #     patience_counter = 0
        # else:
        #     patience_counter += 1
        # if epoch % 10 == 0:
        #     print(f'Epoch {epoch}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss.item():.4f}')
        # scheduler.step(val_loss)
        # if patience_counter >= patience:
        #     print(f"Early stopping at epoch {epoch}")
        #     break

# Load best model
# model.load_state_dict(best_model_state)

# Evaluation
model.eval()
with torch.no_grad():
    val_pred = model(edge_attr, data.edge_index)[val_mask]
    val_true = label[val_mask]
    val_pred_denorm = scaler_label.inverse_transform(val_pred.detach().numpy())
    val_true_denorm = scaler_label.inverse_transform(val_true.detach().numpy())
    print("\nSample predictions (denormalized):", val_pred_denorm[:5].flatten())
    print("Sample true labels (denormalized):", val_true_denorm[:5].flatten())
    print("\nPrediction statistics (denormalized):")
    print(f"Min predicted: {val_pred_denorm.min():.2f}")
    print(f"Max predicted: {val_pred_denorm.max():.2f}")
    print(f"Mean predicted: {val_pred_denorm.mean():.2f}")
    print(f"Min true: {val_true_denorm.min():.2f}")
    print(f"Max true: {val_true_denorm.max():.2f}")
    print(f"Mean true: {val_true_denorm.mean():.2f}")
    r2 = r2_score(val_true_denorm, val_pred_denorm)
    mae = mean_absolute_error(val_true_denorm, val_pred_denorm)
    rmse = np.sqrt(mean_squared_error(val_true_denorm, val_pred_denorm))
    print(f"\nEvaluation Metrics for Validation Set:")
    print(f"R² Score: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

plt.figure(figsize=(10, 8))
plt.scatter(val_true_denorm, val_pred_denorm, alpha=0.5)
plt.plot([val_true_denorm.min(), val_true_denorm.max()], [val_true_denorm.min(), val_true_denorm.max()], 'r--', label='Perfect Prediction')
plt.xlabel("Actual Roughness")
plt.ylabel("Predicted Roughness")
plt.title("NNConv GNN: Predicted vs Actual Pipe Roughness (Validation Set)")
plt.grid(True)
plt.legend()
plt.show()

errors = val_pred_denorm - val_true_denorm
plt.figure(figsize=(10, 6))
plt.hist(errors, bins=30, alpha=0.7)
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Distribution of Prediction Errors (NNConv)")
plt.grid(True)
plt.show()
# %%
