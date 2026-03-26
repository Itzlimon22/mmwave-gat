# File: scripts/train.py
import os
import torch
import torch.nn.functional as F
from model import GATNet
from torch_geometric.data import Data
import matplotlib.pyplot as plt

def train_gat():
    """
    Docstring: Main training pipeline for the Graph-Informed mmWave Regression model.
    Implements Early Stopping and Gradient Clipping for stability.
    """
    
    # --- 1. Setup & Device ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    # --- 2. Load Data ---
    data_path = '../data/processed_graph.pt'
    if not os.path.exists(data_path):
        return print(f"Error: {data_path} not found. Run build_graph.py first.")
    
    data = torch.load(data_path).to(device)

    # --- 3. Split Masks ---
    # Creating simple 80/10/10 split masks for the 10,000 nodes
    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes)
    
    data.train_mask = indices[:8000]
    data.val_mask = indices[8000:9000]
    data.test_mask = indices[9000:]

    # --- 4. Initialize Model, Optimizer, & Loss ---
    model = GATNet(num_features=3, hidden_channels=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = torch.nn.MSELoss()

    # --- 5. Training Loop ---
    best_val_loss = float('inf')
    patience = 20
    trigger_times = 0
    history = []

    print("Starting Training...")
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        
        # Forward Pass
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        
        # Backward Pass
        loss.backward()
        
        # Professional Polish: Gradient Clipping (Standard norm = 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_out = model(data.x, data.edge_index)
            val_loss = criterion(val_out[data.val_mask], data.y[data.val_mask])
            history.append(val_loss.item())

        if epoch % 10 == 0:
            print(f'Epoch {epoch:03d} | Train Loss: {loss:.4f} | Val Loss: {val_loss:.4f}')

        # --- Early Stopping Logic ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
            # Save the best weights
            torch.save(model.state_dict(), '../models/gat_path_loss_model.pth')
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping at epoch {epoch}. Model weights saved to /models.")
                break

    # --- 6. Final Test Performance ---
    model.load_state_dict(torch.load('../models/gat_path_loss_model.pth'))
    model.eval()
    with torch.no_grad():
        test_out = model(data.x, data.edge_index)
        test_rmse = torch.sqrt(criterion(test_out[data.test_mask], data.y[data.test_mask]))
        print(f"\nFinal Test RMSE: {test_rmse:.2f} dB")

    # Optional: Save a quick loss plot for the GitHub docs/ folder
    plt.plot(history)
    plt.title("Validation Loss Trend")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.savefig('../docs/training_loss.png')

if __name__ == "__main__":
    train_gat()