# File: scripts/model.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATNet(torch.nn.Module):
    """
    JSDoc: 3-Layer Graph Attention Network with Residual Connections.
    Designed for NLoS Path Loss regression at 28 GHz.
    """
    def __init__(self, num_features=3, hidden_channels=64, heads=4):
        super(GATNet, self).__init__()
        # Layer 1: Input to Hidden
        self.conv1 = GATConv(num_features, hidden_channels, heads=heads, concat=False)
        
        # Layer 2: Hidden to Hidden
        self.conv2 = GATConv(hidden_channels, hidden_channels, heads=heads, concat=False)
        
        # Layer 3: Hidden to Output
        self.conv3 = GATConv(hidden_channels, hidden_channels, heads=heads, concat=False)
        
        # Regression Head
        self.lin = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        # Logic Flow: Aggregate neighbors -> Add Residual -> Activation
        
        # Block 1
        h1 = self.conv1(x, edge_index)
        h1 = F.elu(h1)
        
        # Block 2 (Residual Connection)
        h2 = self.conv2(h1, edge_index)
        h2 = F.elu(h2)
        h2 = h2 + h1 # Skip Connection to prevent oversmoothing
        
        # Block 3 (Residual Connection)
        h3 = self.conv3(h2, edge_index)
        h3 = F.elu(h3)
        h3 = h3 + h2
        
        return self.lin(h3)