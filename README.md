# BeeFreeAgro

<div align="center">
  <img src="data/synth/imgs/1.jpg" alt="beefreeagro logo" width="300"/>
</div>

## Overview
This repository contains code for creating synthetic datasets, training a custom sticker detector model, and testing the model on new images.

## Repository Structure
```
beefreeagro/
├── data/
│   ├── raw/
│   │   ├── stickers/    # Place sticker images here
│   │   └── background/  # Place your background images here
│   └── synth/           # Generated synthetic dataset
│       ├── imgs/        # Images with stickers placed on backgrounds
│       └── annotations.txt  # Generated annotations for training
├── models/              # Trained models will be saved here
├── scripts/
│   ├── init-cpu.bat     # Windows setup script for CPU
│   └── init-gpu.bat     # Windows setup script for GPU
├── create_data.py       # Script for synthetic dataset creation
├── train.py             # Training script
├── test.py              # Testing/inference script
├── stickers_env.yml     # Conda environment file
├── requirements.txt     # Python dependencies
├── Dockerfile           # For Docker-based setup
└── README.md            # This file
```

## Setup Instructions

### Option 1: Using Conda (Mac/Linux)
```bash
# Create and activate conda environment
conda env create -f stickers_env.yml
conda activate stickers_env
```

### Option 2: Using Docker (Any OS)
```bash
# Build Docker image
docker build -t beefreeagro .

# Run container with GPU support
docker run -it --rm --gpus all --shm-size=1g -v /path/to/local/repo:/beefree beefreeagro

# For Windows, use:
docker run -it --rm --gpus all --shm-size=1g -v C:\path\to\local\repo:/beefree beefreeagro

# Verify GPU availability inside container
python -c "import torch; print(torch.cuda.is_available())"
```

### Option 3: Windows with Virtual Environment
For Windows machines, use the provided batch scripts:

```bash
# For GPU support
.\scripts\init-gpu.bat

# For CPU only
.\scripts\init-cpu.bat
```

## Usage Instructions

### 1. Prepare Raw Data
- Place sticker images in `data/raw/stickers/`
- Place background images in `data/raw/background/`

You can [download raw data here](https://drive.google.com/file/d/13Z_CTtKU9mfbX-AH2COw5HBsGW6boltu/view?usp=sharing)

### 2. Generate Synthetic Dataset
```bash
python create_data.py --output_dir data/synth/imgs --annotations_file data/synth/annotations.txt
```

You can download the synthetic dataset here

This will:
- Generate synthetic images with stickers placed on backgrounds
- Create annotation file in the format required for training

### 3. Train the Model
```bash
# For GPU training
python train.py --data_dir data/synth/imgs --annotations data/synth/annotations.txt --epochs 50 --model_size n --device cuda

# For CPU training
python train.py --data_dir data/synth/imgs --annotations data/synth/annotations.txt --epochs 50 --model_size n --device cpu
```

Parameters:
- `--data_dir`: Directory with training images
- `--annotations`: Path to annotations file
- `--epochs`: Number of training epochs
- `--model_size`: YOLOv8 model size (n, s, m, l, x)
- `--device`: Training device (cuda, cpu)

### 4. Test/Inference
```bash
python test.py test_dir result_dir --model models/stickers_detector/weights/best.pt --conf 0.25
```

Parameters:
- `test_dir`: Directory with test images
- `result_dir`: Directory to save detection results
- `--model`: Path to trained model
- `--conf`: Confidence threshold for detections

Output format:
```
0.png (40,21)-(230,300) sad
0.jpg (440,421)-(523,530) happy
...
```

## Detailed File Descriptions

### create_data.py
Creates a synthetic dataset by placing sticker images on background images with various transformations and generates corresponding annotations.

### train.py
Trains a YOLOv8 model on the synthetic dataset with customizable parameters.

### test.py
Performs inference on test images, saves visualized detections, and outputs bounding box information.

### Dockerfile
Sets up a CUDA-enabled environment for GPU training.

### stickers_env.yml & requirements.txt
Dependency definitions for conda and pip environments.

## System Requirements
- Python 3.10
- CUDA-compatible GPU (optional, for faster training)
- Docker (optional)