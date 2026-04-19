"""
Google Vision Transformer (ViT) Analyzer for Food Classification
Uses pre-trained ViT models for advanced food recognition and classification
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

class ViTAnalyzer:
    """Google Vision Transformer analyzer for food classification"""
    
    def __init__(self, model_name: str = "google/vit-base-patch16-224", 
                 model_path: Optional[str] = None):
        """
        Initialize ViT analyzer
        
        Args:
            model_name: Hugging Face model name for ViT
            model_path: Local path to fine-tuned model (optional)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        
        # Food-101 class labels (101 food categories)
        self.food_classes = self._load_food_classes()
        
        # Initialize model and processor
        self.model = self._load_model(model_path)
        self.processor = self._load_processor()
        
        # Confidence threshold for predictions
        self.confidence_threshold = 0.5
        
        logging.info(f"ViT Analyzer initialized with model: {model_name}")
    
    def _load_food_classes(self) -> List[str]:
        """Load Food-101 class labels"""
        return [
            'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare',
            'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito',
            'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake',
            'ceviche', 'cheese_cake', 'cheese_plate', 'chicken_curry', 'chicken_quesadilla',
            'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder',
            'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes',
            'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict',
            'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras',
            'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice',
            'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich',
            'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup',
            'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna',
            'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup',
            'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters',
            'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck',
            'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib',
            'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto',
            'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits',
            'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake',
            'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare',
            'waffles'
        ]
    
    def _load_model(self, model_path: Optional[str]) -> ViTForImageClassification:
        """Load ViT model"""
        try:
            if model_path and Path(model_path).exists():
                # Load fine-tuned local model
                model = ViTForImageClassification.from_pretrained(model_path)
            else:
                # Load pre-trained model from Hugging Face
                model = ViTForImageClassification.from_pretrained(
                    self.model_name,
                    num_labels=len(self.food_classes),
                    ignore_mismatched_sizes=True
                )
            
            model.to(self.device)
            model.eval()
            return model
            
        except Exception as e:
            logging.error(f"Error loading ViT model: {e}")
            raise
    
    def _load_processor(self) -> ViTImageProcessor:
        """Load ViT image processor"""
        try:
            processor = ViTImageProcessor.from_pretrained(self.model_name)
            return processor
        except Exception as e:
            logging.error(f"Error loading ViT processor: {e}")
            raise
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for ViT
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed tensor
        """
        # Convert numpy array to PIL Image
        if len(image.shape) == 3 and image.shape[2] == 3:
            # RGB image
            pil_image = Image.fromarray(image.astype(np.uint8))
        else:
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_image.astype(np.uint8))
            else:
                # Grayscale to RGB
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                pil_image = Image.fromarray(rgb_image.astype(np.uint8))
        
        # Process image using ViT processor
        inputs = self.processor(pil_image, return_tensors="pt")
        
        # Move to device
        pixel_values = inputs['pixel_values'].to(self.device)
        
        return pixel_values
    
    def analyze(self, image: np.ndarray, top_k: int = 5) -> Dict:
        """
        Analyze food image using ViT
        
        Args:
            image: Input image as numpy array
            top_k: Number of top predictions to return
            
        Returns:
            Analysis results with predictions and confidence scores
        """
        try:
            # Preprocess image
            pixel_values = self.preprocess_image(image)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(pixel_values)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
            
            # Get top predictions
            top_probs, top_indices = torch.topk(probabilities, min(top_k, len(self.food_classes)))
            
            # Prepare results
            predictions = []
            for i in range(top_k):
                idx = top_indices[0][i].item()
                prob = top_probs[0][i].item()
                class_name = self.food_classes[idx]
                
                predictions.append({
                    'class': class_name,
                    'confidence': prob,
                    'index': idx
                })
            
            # Filter by confidence threshold
            filtered_predictions = [p for p in predictions if p['confidence'] >= self.confidence_threshold]
            
            results = {
                'model': 'ViT',
                'predictions': filtered_predictions,
                'top_prediction': filtered_predictions[0] if filtered_predictions else None,
                'confidence_scores': {p['class']: p['confidence'] for p in predictions},
                'input_shape': image.shape,
                'processing_time': None  # Could add timing if needed
            }
            
            return results
            
        except Exception as e:
            logging.error(f"Error in ViT analysis: {e}")
            return {
                'model': 'ViT',
                'error': str(e),
                'predictions': [],
                'top_prediction': None
            }
    
    def get_food_category(self, prediction: Dict) -> str:
        """
        Get broader food category from specific prediction
        
        Args:
            prediction: Prediction dictionary
            
        Returns:
            Broader food category
        """
        if not prediction:
            return 'unknown'
        
        food_class = prediction['class'].lower()
        
        # Category mapping
        category_map = {
            'desserts': ['cake', 'pudding', 'mousse', 'creme', 'tart', 'pie', 'cookies', 'donuts', 'waffles', 'pancakes'],
            'main_dishes': ['chicken', 'beef', 'pork', 'fish', 'salmon', 'steak', 'ribs', 'tartare', 'carpaccio'],
            'appetizers': ['salad', 'soup', 'wings', 'calamari', 'oysters', 'scallops', 'mussels', 'bruschetta'],
            'asian_cuisine': ['sushi', 'pho', 'ramen', 'pad_thai', 'bibimbap', 'takoyaki', 'gyoza', 'dumplings'],
            'italian_cuisine': ['pizza', 'pasta', 'lasagna', 'ravioli', 'risotto', 'gnocchi', 'carbonara', 'bolognese'],
            'mexican_cuisine': ['tacos', 'quesadilla', 'burrito', 'nachos', 'huevos', 'guacamole'],
            'breakfast': ['eggs', 'pancakes', 'waffles', 'french_toast', 'omelette'],
            'sides': ['fries', 'onion_rings', 'garlic_bread', 'macaroni', 'rice', 'potato'],
            'seafood': ['fish', 'salmon', 'shrimp', 'crab', 'lobster', 'mussels', 'oysters', 'scallops']
        }
        
        for category, foods in category_map.items():
            if any(food in food_class for food in foods):
                return category
        
        return 'other'
    
    def fine_tune_info(self) -> Dict:
        """
        Get information about fine-tuning the model
        
        Returns:
            Fine-tuning information and recommendations
        """
        return {
            'current_model': self.model_name,
            'num_classes': len(self.food_classes),
            'dataset_recommendation': 'Food-101 dataset',
            'training_tips': [
                'Use Food-101 dataset for food-specific fine-tuning',
                'Consider data augmentation for better generalization',
                'Start with a lower learning rate (1e-5)',
                'Use early stopping to prevent overfitting',
                'Fine-tune on a subset first, then full dataset'
            ],
            'hyperparameters': {
                'learning_rate': '1e-5 to 5e-5',
                'batch_size': '16-32',
                'epochs': '10-20',
                'weight_decay': '0.01'
            }
        }
