# SENTINEL Computer Vision

Computer vision work for the SENTINEL autonomous RC car project.

## Included

- `01_dataset_loader.ipynb` — TuSimple preprocessing and label generation
- `02_classical_pipeline.ipynb` — classical lane-detection baseline
- `03_custom_cnn.ipynb` — custom multi-task CNN trained from scratch
- `04_transfer_learning.ipynb` — MobileNetV2 transfer-learning model
- `05_experiments.ipynb` — controlled experiments for the CIE-552 report
- `06_deployment.ipynb` — export, quantization, and benchmark workflow
- `sentinel_dataset/collect_real_data.py` — real-data collection utility
- `Computer Vision Term Project.pdf` — project specification

## GitHub Repository

Public repo: https://github.com/AhmedNasr5804/sentinel-computer-vision

## Notes

- Large datasets are intentionally excluded from version control.
- Generated outputs in `processed/` are not meant to be committed.
- The notebooks are designed to run in order, starting from notebook 01.