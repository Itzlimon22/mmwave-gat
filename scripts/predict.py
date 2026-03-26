# File: scripts/predict.py
import torch
from model import GATNet
from sklearn.metrics import mean_squared_error
import numpy as np

def run_evaluation():
    # 1. Load Data
    data = torch.load('../data/processed_graph.pt')
    
    # 2. Load Model
    model = GATNet(num_features=3, hidden_channels=64)
    model.load_state_dict(torch.load('../models/gat_path_loss_model.pth', map_location='cpu'))
    model.eval()
    
    # 3. Predict
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        rmse = np.sqrt(mean_squared_error(data.y.numpy(), out.numpy()))
        
    print(f"✓ Evaluation Complete. Final RMSE: {rmse:.2f} dB")

if __name__ == "__main__":
    run_evaluation()