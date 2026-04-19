"""
Food Detection Module
Core computer vision functionality for detecting and classifying food items
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, List, Dict, Optional
import json

class FoodDetector:
    """Main food detection and classification system"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the food detector
        
        Args:
            model_path: Path to pre-trained model file
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.transform = self._get_transforms()
        self.food_classes = self._load_food_classes()
        
    def _load_model(self, model_path: Optional[str]) -> nn.Module:
        """Load or create the food classification model"""
        if model_path and torch.cuda.is_available():
            try:
                model = torch.load(model_path, map_location=self.device)
                return model
            except:
                pass
        
        # Use ResNet50 pre-trained on ImageNet as base
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Modify for food classification (101 classes for Food-101)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 101)
        
        return model.to(self.device)
    
    def _get_transforms(self) -> transforms.Compose:
        """Get image preprocessing transforms"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _load_food_classes(self) -> List[str]:
        """Load Food-101 class names"""
        # Food-101 class names
        return [
            'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare',
            'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito',
            'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake',
            'ceviche', 'cheesecake', 'cheese_plate', 'chicken_curry', 'chicken_quesadilla',
            'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder',
            'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes',
            'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict',
            'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras',
            'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice',
            'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich',
            'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup',
            'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna',
            'lobster_roll_sandwich', 'lobster_bisque', 'macaroni_and_cheese', 'macarons', 'misso_soup',
            'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters',
            'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck',
            'pho', 'pizza', 'pork_chop', 'ramen', 'red_velvet_cake',
            'risotto', 'samosa', 'sashimi', 'scallops', 'seaweed_salad',
            'shrimp_and_grits', 'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak',
            'strawberry_shortcake', 'sushi', 'tacos', 'tiramisu', 'waffles'
        ]
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model input
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV)
            
        Returns:
            Preprocessed tensor
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Apply transforms
        tensor = self.transform(pil_image)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)
    
    def detect_food(self, image: np.ndarray, confidence_threshold: float = 0.5) -> Dict:
        """
        Detect and classify food in image
        
        Args:
            image: Input image as numpy array
            confidence_threshold: Minimum confidence for detection
            
        Returns:
            Dictionary containing detection results
        """
        # Preprocess image
        input_tensor = self.preprocess_image(image)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Perform inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
        # Get top predictions
        top_prob, top_class = torch.max(probabilities, 0)
        
        # Get all predictions above threshold
        predictions = []
        for i, prob in enumerate(probabilities):
            if prob > confidence_threshold:
                predictions.append({
                    'class': self.food_classes[i],
                    'confidence': float(prob),
                    'class_id': i
                })
        
        # Sort by confidence
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            'top_prediction': {
                'class': self.food_classes[top_class],
                'confidence': float(top_prob),
                'class_id': int(top_class)
            },
            'all_predictions': predictions,
            'image_shape': image.shape
        }
    
    def get_food_regions(self, image: np.ndarray) -> List[Dict]:
        """
        Detect multiple food regions in image using object detection
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detected food regions with bounding boxes
        """
        # This would use a more advanced object detection model like YOLO or Faster R-CNN
        # For now, return the entire image as one region
        h, w = image.shape[:2]
        
        return [{
            'bbox': [0, 0, w, h],  # [x, y, width, height]
            'confidence': 1.0,
            'class': 'food_item'
        }]
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract feature embeddings from image
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Feature vector
        """
        # Preprocess image
        input_tensor = self.preprocess_image(image)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Extract features from penultimate layer
        with torch.no_grad():
            # Remove the final classification layer
            feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
            features = feature_extractor(input_tensor)
            
        # Flatten and convert to numpy
        features = features.flatten().cpu().numpy()
        
        return features
