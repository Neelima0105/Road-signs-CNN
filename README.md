# Road Sign Image Classification with a Custom CNN (PyTorch)

A convolutional neural network built from scratch in PyTorch to classify road/traffic sign images from a labeled image dataset.

## Features

• Custom 4-block CNN (32 to 256 channels) with batch normalization and ReLU  
• Data augmentation (random horizontal flip, rotation) for better generalization  
• Dropout-regularized fully connected classifier head  
• Train/validation split with per-epoch loss and accuracy tracking  
• Evaluation via classification report and confusion matrix visualization  
• Best-model checkpointing

## Tech Stack

Python, PyTorch, Torchvision, scikit-learn, Matplotlib, Pillow

## Project Structure

```
Road-signs-CNN/
src/
  model.py        CNN architecture
  train.py        Training loop (train/val split, augmentation)
  evaluate.py     Classification report and confusion matrix
  prediction.py   Single-image inference
  models/         Saved checkpoints
  outputs/        Generated plots
requirements.txt
```

## Setup and Run

```bash
pip install -r requirements.txt
cd src
python train.py
python evaluate.py
python prediction.py
```

Note: update the dataset path in src/train.py to point to your local image dataset, organized ImageFolder-style with one subfolder per class.

## Resume Bullet

Built a custom CNN in PyTorch for multi-class road sign image classification, implementing data augmentation, batch normalization, dropout regularization, and evaluation via confusion matrices and classification reports.
