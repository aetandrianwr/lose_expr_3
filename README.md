# Next-Location Prediction System

Production-grade implementation of next-location prediction using PyTorch and Transformer architecture.

## Features

- **Pure Transformer Architecture**: No RNN/LSTM, fully attention-based
- **Parameter Efficient**: <500K params (Geolife), <1M params (DIY)
- **Multi-Feature Fusion**: Location, user, temporal, and duration features
- **GPU Accelerated**: Full CUDA support
- **Production Ready**: Modular code, comprehensive metrics, checkpointing

## Project Structure

```
.
├── configs/
│   ├── geolife_config.yaml
│   └── diy_config.yaml
├── src/
│   ├── data/
│   │   └── dataset.py
│   ├── models/
│   │   └── transformer_model.py
│   └── utils/
│       ├── metrics.py
│       └── trainer.py
├── data/
│   ├── geolife/
│   └── diy/
├── checkpoints/
├── train.py
└── README.md
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU
- See requirements.txt for full dependencies

## Installation

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pyyaml tqdm scikit-learn numpy
```

## Usage

### Train Geolife Model

```bash
python train.py --config configs/geolife_config.yaml
```

### Train DIY Model

```bash
python train.py --config configs/diy_config.yaml
```

### Resume Training

```bash
python train.py --config configs/geolife_config.yaml --resume checkpoints/geolife/best.pt
```

## Model Architecture

**Efficient Transformer Design:**
- Multi-head self-attention (no recurrence)
- Feature fusion: location + user + temporal embeddings
- Positional encoding for sequence order
- Pre-norm architecture for stability
- Parameter sharing for efficiency

## Performance Targets

- **Geolife**: >40% Test Acc@1, <500K parameters
- **DIY**: >45% Test Acc@1, <1M parameters

## Metrics

- Acc@1, Acc@5, Acc@10
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG)

## Author

Research-grade implementation following best practices.
