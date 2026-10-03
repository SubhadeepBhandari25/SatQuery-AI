# Remote Sensing Model Adaptation Pipeline

SatQuery AI includes a reproducible adaptation script in `backend/training/`:

## Adaptation Architecture
- **Backbone:** ResNet18 feature extractor.
- **Adapter Head:** Multi-layer projection bottleneck (256 dimensions) with Dropout and multi-label BCEWithLogits loss.
- **Dataset:** `BigEarthNetSyntheticDataset` implementing Corine Land Cover multi-hot labels.

## Running the Adaptation Script
```bash
cd backend
python training/train_adapter.py
```

The script trains the remote sensing classification adapter and saves the checkpoint to:
`backend/models/checkpoints/bigearthnet_adapter.pt`
