# Graph-Informed Regression Networks for Millimeter-Wave Propagation Prediction in NLoS Urban Scenarios

**Authors:** Limon Howlader, Ragib Mahmud, Fahim Marsad Shuvo, and Arnob Shahriar  
**Affiliation:** Dept. of Electrical, Electronic and Communication Engineering, Military Institute of Science and Technology (MIST)

---

## Project Overview

Reliable communication in 5G and 6G networks depends heavily on accurate path loss prediction in the 28 GHz millimeter-wave (mmWave) band. However, at these high frequencies, signal propagation is extremely sensitive to physical blockages, diffraction around corners, and absorption by concrete facades. 

Traditional empirical models (like the 3GPP UMi) often yield errors as high as 7-10 dB because they lack site-specific geometric awareness. Deterministic ray tracing is highly accurate but computationally expensive, scaling at O(N^3) complexity. This project bridges that gap by using a Graph Attention Network (GAT) that learns the underlying topology of an urban environment to provide ray-tracing levels of accuracy at near-instantaneous inference speeds.

---

## System Architecture

The pipeline transitions from raw geographical data to a trained neural surrogate in three distinct phases:

1. Digital Twin Construction: 1,745 buildings were extracted from OpenStreetMap (OSM) for the Mirpur DOHS area and extruded into 3D polygons with ITU-R concrete dielectric properties.
2. Deterministic Simulation: Using the Sionna 2.0 physics engine, we simulated 10,000 spatial samples with up to 5-bounce Non-Line-of-Sight (NLoS) interactions.
3. Graph Learning: Spatial coordinates were converted into a K-Nearest Neighbor (K=4) graph. A deep GAT with residual connections was then trained to regress the path loss.

[Image Placeholder: System architecture diagram showing GIS data flowing to GAT output]

---

## Dataset: The Mirpur DOHS Digital Twin

We specifically chose Mirpur DOHS, Dhaka, due to its complex residential morphology. The maze-like streets and varying building heights (6-11 stories) represent the most challenging "Urban Canyon" scenarios for mmWave deployment.

### Simulation Parameters

| Parameter | Value |
| :--- | :--- |
| Carrier Frequency | 28 GHz |
| Transmitter Height | 30 meters |
| Receiver Height | 1.5 meters |
| Max NLoS Interactions | 5 Bounces |
| Material | ITU-R Standard Concrete |
| Total Samples | 10,000 |

[Image Placeholder: 3D ray tracing simulation render in the Mirpur urban canyon]

---

## Model Architecture: Deep GAT

Our model is built on the principle that RF propagation is fundamentally a topological problem. By representing receivers as nodes in a graph, the GAT can pass messages along street corridors, effectively modeling waveguiding and shadowing without requiring brute-force physics calculations.

[Image Placeholder: Graph attention network layers and message passing mechanism]

### Key Architectural Features

* Graph Attention Mechanism: The model assigns dynamic weights to neighboring nodes, identifying which spatial directions contribute most to the received signal at any given coordinate.
* Residual (Skip) Connections: We implemented skip connections to prevent the oversmoothing common in deep Graph Neural Networks. This ensures the foundational distance metric persists through the deep network layers.
* Geometric Features: Nodes are parameterized by strictly geometric features [x, y, distance], forcing the network to learn the physics of the environment rather than memorizing fixed locations.

---

## Results and Benchmarking

The Graph-Informed approach achieved a Root-Mean-Square Error (RMSE) of 2.15 dB, a significant improvement over both empirical formulas and standard non-graph Machine Learning models.

| Model | RMSE (dB) | Spatial Awareness |
| :--- | :--- | :--- |
| 3GPP UMi (Empirical) | ~7.20 | None (Distance Only) |
| Multilayer Perceptron (MLP) | 5.84 | Implicit (Coordinates) |
| Proposed Deep GAT | 2.15 | Explicit (Graph Topology) |

[Image Placeholder: Predicted vs actual path loss scatter plot with diagonal perfect-accuracy line]

As demonstrated in our testing, the GAT successfully models the high-attenuation regions (110-120 dB) where the standard 3GPP models typically fail due to complex shadowing in the Mirpur street grid.

---

## Repository Structure

```text
/mmwave-gat
├── /data                # Raw CSV datasets and processed .pt graph files
├── /docs                # Figures, diagrams, and the final IEEE PDF
├── /models              # Saved .pth model weights (2.15 dB RMSE)
├── /notebooks           # Initial EDA and training experiments
├── /scripts             # Production-ready Python modules
│   ├── model.py         # GAT Architecture definition
│   ├── build_graph.py   # KNN Graph construction logic
│   ├── train.py         # Training loop with Early Stopping
│   └── predict.py       # Inference and evaluation script
└── README.md
```

---

## Installation and Usage

### 1. Prerequisites
Ensure you have Python 3.8+ and the following libraries installed:
```bash
pip install torch torch-geometric pandas scikit-learn numpy matplotlib
```

### 2. Build the Graph
Convert the raw Mirpur CSV data into a topology-aware graph object:
```bash
python scripts/build_graph.py
```

### 3. Run Inference
Use the pre-trained weights in the /models folder to verify the 2.15 dB RMSE:
```bash
python scripts/predict.py
```

---

## Citation

If you use this work, the architecture, or the Mirpur DOHS digital twin dataset in your research, please cite our study:

```bibtex
@article{Howlader2026GraphInformed,
  title={Graph-Informed Regression Networks for Millimeter-Wave Propagation Prediction in NLoS Urban Scenarios},
  author={Howlader, Limon and Mahmud, Ragib and Shuvo, Fahim Marsad and Shahriar, Arnob},
  journal={Military Institute of Science and Technology (MIST)},
  year={2026}
}
```
