# SENTINEL Computer Vision

Computer vision work for the SENTINEL autonomous RC car project. This repository contains the lane-detection pipeline, model training notebooks, deployment notes, and supporting scripts used for the CIE-552 computer vision deliverables.

## Project Goal

The computer vision subsystem estimates lane position from front-camera images and produces steering-relevant outputs for the autonomous RC car. The work is organized around three model families and one deployment path:

- a classical lane-detection baseline
- a custom CNN trained from scratch
- a MobileNetV2 transfer-learning model
- export and quantization for deployment on the Raspberry Pi 4

The notebooks are meant to be run in sequence, with notebook 01 generating the dataset labels consumed by the later notebooks.

## Repository Contents

### Notebooks

- `01_dataset_loader.ipynb` — converts TuSimple lane labels into the project’s polynomial schema, resizes images to 320×180, and writes train/validation/test splits.
- `02_classical_pipeline.ipynb` — implements the non-deep-learning baseline using HSV thresholding, Canny edges, ROI masking, Hough lines, HOG features, and an SVM classifier.
- `03_custom_cnn.ipynb` — trains a multi-task convolutional network from scratch to predict `has_lane` and the left/right lane polynomials.
- `04_transfer_learning.ipynb` — fine-tunes a MobileNetV2-based model for the same lane-detection tasks.
- `05_experiments.ipynb` — runs controlled experiments for the report, including color-space comparison, augmentation impact, and distortion robustness.
- `06_deployment.ipynb` — exports the best model, performs TFLite quantization, and benchmarks inference performance.

### Supporting Files

- `sentinel_dataset/collect_real_data.py` — helper script for collecting or organizing real training samples.
- `Computer Vision Term Project.pdf` — project specification and constraints.
- `README.md` — this guide.
- `.gitignore` — excludes generated and bulky artifacts from version control.

## Recommended Workflow

1. Run `01_dataset_loader.ipynb` first to generate `processed/labels_train.json`, `processed/labels_val.json`, and `processed/labels_test.json`.
2. Use `02_classical_pipeline.ipynb` to establish a classical baseline and evaluate it on the test split.
3. Train the custom CNN in `03_custom_cnn.ipynb` and save the best weights and metrics.
4. Run `04_transfer_learning.ipynb` to compare a pretrained backbone against the scratch model.
5. Use `05_experiments.ipynb` to generate the controlled comparison charts required by the course.
6. Finish with `06_deployment.ipynb` to export the selected model and measure inference performance.

## Dataset and Output Layout

The repository assumes the following local folders when working with the notebooks:

- `datasets/` — source datasets such as TuSimple and CEW
- `processed/` — generated images, labels, metrics, plots, checkpoints, and export artifacts
- `archive/` — older or backup datasets not intended for version control

These directories can be large and are intentionally ignored by git. The repo tracks the notebooks and lightweight project files only.

## Environment Notes

The notebooks were designed for a Python notebook environment with the following packages available:

- `numpy`
- `opencv-python`
- `matplotlib`
- `pandas`
- `torch`
- `torchvision`
- `scikit-learn`
- `tqdm`
- `Pillow`

Some notebooks also expect `tflite-runtime` or export-related tooling when you reach the deployment stage.

## Project Conventions

- All images are handled at 320×180 unless a notebook explicitly states otherwise.
- Lane-polynomial coefficients use the project convention `x = a*y^2 + b*y + c`.
- Generated outputs should stay in `processed/` and not be committed.
- Raw datasets should remain in `datasets/` and not be committed.
- The notebooks are intended to be reproducible from top to bottom on a fresh kernel once the dataset folders are present locally.

## GitHub Repository

Public repo: https://github.com/AhmedNasr5804/sentinel-computer-vision

## Status

This repository is the computer-vision portion of the broader SENTINEL autonomous RC car project. It is focused on CIE-552 deliverables, with supporting hooks for deployment into the full system later in the project.
