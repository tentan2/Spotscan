"""
Model Manager Module
Handles loading, training, and management of ML models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
from transformers import ViTForImageClassification, ViTImageProcessor
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import pickle
from datetime import datetime

class ModelManager:
    """Manages all ML models for Spotscan"""
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize model manager
        
        Args:
            models_dir: Directory to store models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model registry
        self.models = {}
        self.model_metadata = {}
        
    def create_food_classifier(self, num_classes: int = 101) -> nn.Module:
        """
        Create a food classification model
        
        Args:
            num_classes: Number of food classes
            
        Returns:
            PyTorch model
        """
        # Use ResNet50 as base model
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Modify final layer for food classification
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        return model.to(self.device)
    
    def create_vit_classifier(self, model_name: str = "google/vit-base-patch16-224", 
                            num_classes: int = 101) -> Tuple[nn.Module, ViTImageProcessor]:
        """
        Create a Vision Transformer classifier
        
        Args:
            model_name: Hugging Face model name
            num_classes: Number of output classes
            
        Returns:
            Tuple of (ViT model, image processor)
        """
        try:
            # Load ViT model
            model = ViTForImageClassification.from_pretrained(
                model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
            
            # Load image processor
            processor = ViTImageProcessor.from_pretrained(model_name)
            
            return model.to(self.device), processor
            
        except Exception as e:
            print(f"Error creating ViT model: {e}")
            raise
    
    def create_freshness_detector(self) -> nn.Module:
        """
        Create a freshness/spoilage detection model
        
        Returns:
            PyTorch model for freshness detection
        """
        # Use a custom CNN for freshness detection
        model = nn.Sequential(
            # Convolutional layers
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Flatten and dense layers
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3)  # fresh, spoiling, spoiled
        )
        
        return model.to(self.device)
    
    def create_ripeness_predictor(self) -> nn.Module:
        """
        Create a ripeness prediction model
        
        Returns:
            PyTorch model for ripeness prediction
        """
        # Use EfficientNet for ripeness prediction
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # Modify classifier for ripeness stages
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 3)  # unripe, ripe, overripe
        )
        
        return model.to(self.device)
    
    def create_texture_analyzer(self) -> nn.Module:
        """
        Create a texture analysis model
        
        Returns:
            PyTorch model for texture analysis
        """
        model = nn.Sequential(
            # Input: 224x224x3
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Flatten(),
            nn.Linear(256 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10)  # 10 texture categories
        )
        
        return model.to(self.device)
    
    def save_model(self, model: nn.Module, model_name: str, metadata: Dict = None):
        """
        Save model and metadata
        
        Args:
            model: PyTorch model to save
            model_name: Name for the model
            metadata: Model metadata
        """
        model_path = self.models_dir / f"{model_name}.pth"
        torch.save(model.state_dict(), model_path)
        
        # Save metadata
        if metadata is None:
            metadata = {
                'created_at': datetime.now().isoformat(),
                'model_type': model_name,
                'device': str(self.device)
            }
        
        metadata_path = self.models_dir / f"{model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model saved: {model_path}")
    
    def load_model(self, model_name: str, model_class: nn.Module) -> nn.Module:
        """
        Load model from disk
        
        Args:
            model_name: Name of the model to load
            model_class: Model class to instantiate
            
        Returns:
            Loaded PyTorch model
        """
        model_path = self.models_dir / f"{model_name}.pth"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = model_class.to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        # Load metadata
        metadata_path = self.models_dir / f"{model_name}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            self.model_metadata[model_name] = metadata
        
        model.eval()
        return model
    
    def train_model(self, model: nn.Module, train_loader: DataLoader, 
                   val_loader: DataLoader, num_epochs: int = 10, 
                   learning_rate: float = 0.001) -> Dict:
        """
        Train a model
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            
        Returns:
            Training history
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
        
        train_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, targets) in enumerate(train_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += targets.size(0)
                train_correct += (predicted == targets).sum().item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, targets in val_loader:
                    data, targets = data.to(self.device), targets.to(self.device)
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += targets.size(0)
                    val_correct += (predicted == targets).sum().item()
            
            # Calculate metrics
            train_loss_avg = train_loss / len(train_loader)
            train_acc = 100 * train_correct / train_total
            val_loss_avg = val_loss / len(val_loader)
            val_acc = 100 * val_correct / val_total
            
            # Update history
            train_history['train_loss'].append(train_loss_avg)
            train_history['train_acc'].append(train_acc)
            train_history['val_loss'].append(val_loss_avg)
            train_history['val_acc'].append(val_acc)
            
            # Learning rate scheduling
            scheduler.step(val_loss_avg)
            
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.2f}%')
            print('-' * 50)
        
        return train_history
    
    def evaluate_model(self, model: nn.Module, test_loader: DataLoader) -> Dict:
        """
        Evaluate model performance
        
        Args:
            model: Model to evaluate
            test_loader: Test data loader
            
        Returns:
            Evaluation metrics
        """
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = model(data)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        accuracy = 100 * correct / total
        avg_loss = test_loss / len(test_loader)
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'predictions': all_predictions,
            'targets': all_targets
        }
    
    def get_model_info(self, model_name: str) -> Dict:
        """
        Get information about a saved model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model information
        """
        metadata_path = self.models_dir / f"{model_name}_metadata.json"
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return metadata
        else:
            return {"error": f"Model metadata not found for {model_name}"}
    
    def list_models(self) -> List[str]:
        """
        List all available models
        
        Returns:
            List of model names
        """
        model_files = list(self.models_dir.glob("*.pth"))
        model_names = [f.stem for f in model_files if not f.stem.endswith("_metadata")]
        return model_names
