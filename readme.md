# Human Activity Recognition using ConvLSTM and LRCN Models

This project implements video-based human activity recognition using two deep learning architectures:
- ConvLSTM (Convolutional Long Short-Term Memory)
- LRCN (Long-term Recurrent Convolutional Networks)

Both models are built and trained to classify human activities from video sequences.  
The project includes model definitions, training scripts, a Jupyter notebook for exploration, and pre-trained weights for evaluation.

## Project Structure

```
├── index.ipynb                # Main Jupyter notebook (full pipeline: data loading, training, evaluation)
├── model_convLSTM.py          # ConvLSTM model definition
├── model_LRCN.py              # LRCN model definition
├── train_convLSTM.py          # Training script for ConvLSTM
├── train_LRCN.py              # Training script for LRCN
├── weights/                   # Directory containing saved model weights (.h5 files)
│   ├── convlstm_weights.h5
│   └── lrcn_weights.h5
├── requirements.txt           # Required Python packages
└── README.md                  # Project documentation
```

## Models Overview

### 1. ConvLSTM Model
- Combines convolutional layers with LSTM layers to capture both spatial and temporal dependencies in video frames.
- Ideal for sequence modeling of spatial features.

File: model_convLSTM.py  
Training script: train_convLSTM.py

### 2. LRCN Model
- Sequential model that uses CNNs for spatial feature extraction and LSTM for temporal sequence modeling.
- Popular for video classification tasks.

File: model_LRCN.py  
Training script: train_LRCN.py

## Setup and Installation

1. Clone the repository

git clone <repository_url>
cd <project_directory>

2. Install dependencies

Ensure you have Python 3.8+ installed. Then run:

pip install -r requirements.txt

## Usage

### Option 1: Jupyter Notebook
Run index.ipynb to see the full pipeline — from data preprocessing, model training, to evaluation and visualization.

### Option 2: Command Line Scripts
- Train ConvLSTM model

python train_convLSTM.py

- Train LRCN model

python train_LRCN.py

Pre-trained weights are available in the weights/ directory.

## Pre-trained Weights

You can use the provided .h5 weights to skip training and directly evaluate the models.

model.load_weights('weights/convlstm_weights.h5')

## Requirements

All required dependencies are listed in requirements.txt.

Example:

numpy
tensorflow
keras
opencv-python
matplotlib
scikit-learn

## Future Work

- Add more architectures for comparison (e.g., 3D CNNs, Transformer-based models).
- Implement data augmentation techniques.
- Deploy the model as a web application or REST API.
- Hyperparameter optimization for improved accuracy.

## License

This project is licensed under the MIT License.

## Contributions

Contributions, issues, and feature requests are welcome!  
Feel free to open a pull request or raise an issue.

## Contact

For any inquiries or collaborations, please contact:

Akash Rana