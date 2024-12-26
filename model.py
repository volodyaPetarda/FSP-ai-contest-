import os
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm  # We will use timm for loading pre-trained ConvNeXt or other models
import pandas as pd
import timm
import torch
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from timm import create_model
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch.utils.data import DataLoader, TensorDataset

import torch.nn as nn
import torch.optim as optim

class ImageClassifier:
    def __init__(self, model_path: str = "model.jit", model_name: str = 'convnext_base', num_classes: int = 10):
        """
        Initialize the ImageClassifier.

        Args:
            model_path (str): Path to the .pth model file.
            model_name (str): Name of the model architecture to load.
            num_classes (int): Number of output classes (for classification).
        """
        self.device_ = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_ = None
        self._load_model(model_path, model_name, num_classes)
        self._define_preprocessing()

    def _load_model(self, model_path: str, model_name: str, num_classes: int):
        """
        Load the model from the .pth file.

        Args:
            model_path (str): Path to the .pth model file.
            model_name (str): Model architecture name.
            num_classes (int): Number of output classes.
        """
        # Load the TorchScript model
        model_fname = os.path.join(os.path.dirname(__file__), model_path)  # os.path.dirname(__file__)
        self.model_ = torch.jit.load(model_fname).to(self.device_)

        # Ensure the model is in evaluation mode
        self.model_.eval()
        print(f'Model loaded successfully from "{model_path}".')

    def _define_preprocessing(self):
        """Define preprocessing pipeline for input images."""
        self.preprocess_ = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess a single image.

        Args:
            image (np.ndarray): Input image in HWC format with pixel values in [0, 255].

        Returns:
            torch.Tensor: Preprocessed image as a PyTorch tensor.
        """
        processed = self.preprocess_(image=image)['image']
        return processed.to(self.device_)

    def predict(self, image: np.ndarray) -> int:
        """
        Predict the class of a single image.

        Args:
            image (np.ndarray): Input image in HWC format.

        Returns:
            int: Predicted class index.
        """
        image_tensor = self._preprocess(image)
        image_tensor = torch.unsqueeze(image_tensor, 0)  # Add batch dimension
        with torch.no_grad():
            outputs = self.model_(image_tensor) * torch.tensor([1.6494, 0.7575, 1.5691, 1.4699, 1.0274, 0.5894, 0.8560, 0.6961, 1.0450,
        1.9083]).to(self.device_)
            _, predicted = torch.max(outputs, dim=1)
        return predicted.item()

    def predict_batch(self, images: np.ndarray) -> np.ndarray:
        """
        Predict the class of a batch of images.

        Args:
            images (np.ndarray): Batch of images in NHWC format.

        Returns:
            np.ndarray: Array of predicted class indices.
        """
        image_tensors = torch.stack([self._preprocess(image) for image in images])
        with torch.no_grad():
            outputs = self.model_(image_tensors)
            _, predicted = torch.max(outputs, dim=1)
        return predicted.cpu().numpy()
