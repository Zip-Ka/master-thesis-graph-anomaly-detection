# Temporal Graph Neural Networks for Anomaly Detection

Master's Thesis - Implementation of GraphSAGE-based anomaly detection on dynamic networks

## 🎯 Overview

This repository contains the implementation of my Master's thesis on detecting anomalies in temporal graph data using Graph Neural Networks. The project addresses class imbalance through novel graph-based oversampling techniques and evaluates performance across multiple time periods.

### Key Features

- **Advanced GNN Architecture**: Custom GraphSAGE implementation with:
  - Multi-layer aggregation strategies (mean, max, sum)
  - Residual connections for better gradient flow
  - Edge dropout for regularization
  - Batch normalization

- **Class Imbalance Handling**:
  - Graph-SMOTE: Synthetic node generation in graph space
  - Focal Loss with tunable gamma parameter
  - Dynamic class weighting
  - Minority edge reweighting

- **Robust Evaluation**:
  - Temporal train/validation/test splits
  - Comprehensive metrics (F1, AUROC, PR-AUC, MCC, Kappa)
  - Optimal threshold selection via composite scoring
  - Per-timestep performance analysis

- **Hyperparameter Optimization**:
  - Optuna-based search with Hyperband pruning
  - 50+ parameter combinations evaluated
  - Automatic best model selection

- **Experiment Tracking**:
  - Weights & Biases integration
  - Metrics logging and visualization
  - Model checkpointing

## 📊 Results

| Metric | Validation | Test |
|--------|-----------|------|
| Composite Score | 0.XX | 0.XX |
| Balanced Accuracy | 0.XX | 0.XX |
| Class 1 Recall | 0.XX | 0.XX |
| Class 1 F1 | 0.XX | 0.XX |
| AUROC | 0.XX | 0.XX |
| PR-AUC | 0.XX | 0.XX |

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
CUDA 11.7+ (for GPU support)
```

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/thesis-graph-anomaly-detection.git
cd thesis-graph-anomaly-detection

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from src.training.trainer import EnhancedModelTrainer
from pathlib import Path

# Initialize trainer
trainer = EnhancedModelTrainer(
    checkpoint_dir=Path('data/processed'),
    wandb_project='your-project',
    wandb_entity='your-entity',
    oversample_ratio=50.0
)

# Run hyperparameter search
best_params = trainer.run_parameter_search(n_trials=50)

# Train final model
model, metrics = trainer.train_final_model_with_timesteps(best_params)

# Extract embeddings
embeddings_df = trainer.extract_node_embeddings(
    model=model,
    output_path="embeddings.parquet"
)
```

## 📁 Project Structure

```
thesis-graph-anomaly-detection/
├── README.md
├── requirements.txt
├── .gitignore
├── config/
│   └── best_hyperparameters.yaml
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── graphsage.py          # GraphSAGE architecture
│   │   └── layers.py              # Custom graph convolution layers
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_handler.py        # Data loading and preprocessing
│   │   └── graph_smote.py         # Graph-based SMOTE implementation
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py             # Main training logic
│   │   ├── losses.py              # Custom loss functions
│   │   └── metrics.py             # Evaluation metrics
│   └── utils/
│       ├── __init__.py
│       └── visualization.py       # Plotting utilities
├── notebooks/
│   └── analysis.ipynb             # Exploratory analysis
└── tests/
    └── test_models.py
```

## 🔧 Configuration

### Best Hyperparameters

The optimal hyperparameters found through Optuna search:

```yaml
num_layers: 1
base_channels: 240
dropout: 0.435
learning_rate: 4.89e-05
weight_decay: 0.00158
residual: false
edge_dropout: 0.0177
focal_gamma: 2.79
class_weight_factor: 1.03
aggregation: sum
num_neighbors: [5]
```

### Environment Variables

Create a `.env` file:

```bash
WANDB_PROJECT=your-project-name
WANDB_ENTITY=your-entity-name
WANDB_API_KEY=your-api-key
DATA_PATH=/path/to/data
```

## 📈 Methodology

### Data Pipeline

1. **Preprocessing**: Node features normalized, edges validated
2. **Temporal Splitting**: 
   - Train: Timesteps 1-36
   - Validation: Timesteps 37-41
   - Test: Timesteps 42-49
3. **Oversampling**: Graph-SMOTE applied only to training data
4. **Neighbor Sampling**: Dynamic neighborhood sampling per layer

### Graph-SMOTE Algorithm

Novel implementation of SMOTE for graph-structured data:

1. Identify minority class nodes
2. Find k-hop neighbors of same class
3. Generate synthetic nodes via interpolation
4. Create edges to parent neighborhoods
5. Weight synthetic edges appropriately

### Model Architecture

```
Input Features (n_features)
    ↓
GraphSAGE Layer 1 (aggregation + transform)
    ↓
ReLU + BatchNorm + Dropout
    ↓
[Optional: Additional GraphSAGE Layers]
    ↓
Linear Classification Layer (2 classes)
    ↓
Softmax
```

### Loss Function

Enhanced focal loss with label smoothing:

```
L = -α * (1 - p_t)^γ * log(p_t)
```

where:
- `α`: Class weights
- `γ`: Focusing parameter (2.79)
- `p_t`: Model probability for true class

## 🧪 Experiments

### Running Hyperparameter Search

```bash
python scripts/run_optimization.py --n-trials 50 --output config/
```

### Training with Best Parameters

```bash
python scripts/train_final.py --config config/best_hyperparameters.yaml
```

### Evaluation

```bash
python scripts/evaluate.py --model-path models/best_model.pt --test-timesteps 42-49
```

## 📊 Visualization

Generate performance plots:

```python
from src.utils.visualization import MetricsVisualizer

visualizer = MetricsVisualizer(wandb_run=wandb_run)
visualizer.plot_confusion_matrix(y_true, y_pred, "Test Set")
visualizer.plot_pr_curve(y_true, y_prob, "Precision-Recall")
```

## 🤝 Contributing

This is academic research code. If you find issues or have suggestions:

1. Open an issue
2. Describe the problem or enhancement
3. Include code snippets if applicable

## 🙏 Acknowledgments

- PyTorch Geometric team for graph neural network implementations
- Optuna developers for hyperparameter optimization framework
- Weights & Biases for experiment tracking tools

## 📧 Contact

- **Author**: Your Name
- **Email**: your.email@example.com
- **LinkedIn**: [Your Profile](https://linkedin.com/in/yourprofile)
- **Thesis**: [Link to PDF when available]

---

**Note**: This repository contains code from my Master's thesis.