import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
# TSNE is dimension reduction algorithm
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

dataset = Planetoid(root='week13/data/Planetoid', name='Cora', transform=NormalizeFeatures())

print()
print(f'Dataset:{dataset}')