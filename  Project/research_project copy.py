#%%
import wntr
import torch
import numpy as np
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

inp_file = "/Users/sukeluo/Documents/UB/Courses/25Spring/CIE500/CIE500_LL/Assignment/ Project/ky15 EPANET/ky15.inp"
wn = wntr.network.WaterNetworkModel(inp_file)
# %%
