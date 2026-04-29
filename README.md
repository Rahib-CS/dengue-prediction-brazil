🦟 Dengue Outbreak Prediction — Brazil (GNN & MLP)
Predicting monthly dengue cases across Brazilian states using Graph Neural Networks and Multi-Layer Perceptrons on epidemiological and meteorological data.

📌 Project Overview
This project applies two deep learning models to forecast dengue case counts across 27 Brazilian Federative Units (UFs) using a monthly dataset. The models learn from historical case trends, weather patterns, and spatial relationships between states.

Model	Architecture	MAE	R² Score
MLP	3-layer MLP (128→64→32) with BatchNorm & Dropout	1,257	0.9194
Hybrid GNN	GATv2 (spatial) + MLP (temporal) fused	987	0.9464
📁 Repository Structure
text
📦 dengue-prediction-brazil
 ┣ 📓 MLP-2-4.ipynb           # MLP model notebook
 ┣ 📓 GNN-2.ipynb             # Hybrid GNN-MLP model notebook
 ┣ 📊 Brazil_UF_dengue_monthly.csv  # Dataset
 ┗ 📄 README.md
🔬 Models
1. MLP (Multi-Layer Perceptron)
A feedforward neural network trained on per-state temporal features.

Architecture:

text
Input (8 features) → Linear(128) → BN → ReLU → Dropout(0.2)
                   → Linear(64)  → BN → ReLU → Dropout(0.1)
                   → Linear(32)  → ReLU
                   → Linear(1)   [Log Cases Output]
2. Hybrid GNN-MLP (Graph Attention Network)
A spatial-temporal model that encodes state adjacency as a graph, allowing information to flow between neighboring states.

Architecture:

Path A (GNN): GATv2Conv(9→128, heads=4) → GATv2Conv(512→64, heads=2)

Path B (MLP): Linear(128) → Linear(128)

Fusion: Concatenate → Linear(64) → Output

Graph Structure: Brazilian states (UFs) as nodes, sharing borders as edges.

🧪 Feature Engineering
Feature	Description
cases_lag1	Previous month's log-cases (autoregression)
cases_rolling3	3-month rolling average of log-cases
cases_lag12	12-month lag (yearly memory)
precip_lag1	Previous month's precipitation
precip_rolling3	3-month rolling precipitation average
temp_lag1	Previous month's temperature
month_sin / month_cos	Cyclic encoding of seasonality
PopTotalUF	State population (scale factor)
📦 Requirements
bash
pip install torch torch-geometric pandas numpy scikit-learn matplotlib
For PyG (PyTorch Geometric):

bash
pip install torch-scatter torch-sparse torch-geometric
🚀 Running the Models
Clone this repository

Place Brazil_UF_dengue_monthly.csv in the root directory

Open notebooks in Google Colab or a local Jupyter environment

Run all cells in order

📈 Results
Both models are trained on data up to 2018 and evaluated on post-2018 data (temporal split — no data leakage).

The Hybrid GNN outperforms the standalone MLP by capturing spatial diffusion of dengue cases between neighboring states — matching real-world epidemiological dynamics.

📚 Dataset
Source: Brazilian monthly dengue surveillance data

Coverage: All 27 Brazilian states (UFs)

Features: Dengue cases, temperature, precipitation, population
