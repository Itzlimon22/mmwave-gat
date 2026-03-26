# File: scripts/build_graph.py
import torch
import pandas as pd
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors

def build_topology_graph(csv_path="../data/mirpur_data.csv", k=4):
    """
    Converts tabular GIS data into a KNN spatial graph.
    """
    df = pd.read_csv(csv_path)
    
    # Node features: [x, y, d_2d]
    x = torch.tensor(df[['x', 'y', 'distance']].values, dtype=torch.float)
    y = torch.tensor(df['path_loss'].values, dtype=torch.float).view(-1, 1)
    
    # Build edges using KNN (K=4 as per paper)
    coords = df[['x', 'y']].values
    knn = NearestNeighbors(n_neighbors=k+1).fit(coords)
    _, indices = knn.kneighbors(coords)
    
    edge_list = []
    for i, neighbors in enumerate(indices):
        for j in neighbors[1:]: # Skip self
            edge_list.append([i, j])
            
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    data = Data(x=x, edge_index=edge_index, y=y)
    torch.save(data, '../data/processed_graph.pt')
    print("✓ Created: ../data/processed_graph.pt")

if __name__ == "__main__":
    build_topology_graph()