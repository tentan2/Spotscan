"""
Visual Calorie Estimator Module
Estimates calories for whole fruits and restaurant foods using computer vision
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path
import json

class VisualCalorieEstimator:
    """Estimates calories using visual analysis and food databases"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize visual calorie estimator
        
        Args:
            model_path: Path to pre-trained model
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.scaler = StandardScaler()
        
        # Fruit-specific calorie data (per 100g)
        self.fruit_calories = {
            'apple': 52, 'banana': 89, 'orange': 47, 'strawberry': 32,
            'grape': 69, 'watermelon': 30, 'pineapple': 50, 'mango': 60,
            'pear': 57, 'peach': 39, 'plum': 46, 'cherry': 50,
            'kiwi': 61, 'lemon': 29, 'lime': 30, 'avocado': 160,
            'blueberry': 57, 'raspberry': 52, 'blackberry': 43,
            'cranberry': 46, 'fig': 74, 'pomegranate': 83,
            'papaya': 43, 'coconut': 354, 'durian': 147
        }
        
        # Restaurant meal calorie ranges (per serving)
        self.restaurant_calories = {
            'burger': 250-800, 'pizza': 200-400, 'pasta': 300-700,
            'salad': 150-400, 'sandwich': 200-600, 'steak': 300-800,
            'chicken': 200-500, 'fish': 200-600, 'rice': 200-400,
            'soup': 100-300, 'dessert': 200-600, 'breakfast': 300-700
        }
        
        # Calorie database for common foods
        self.calorie_database = self._load_calorie_database()
    
    def _load_model(self, model_path: Optional[str]) -> nn.Module:
        """Load or create calorie estimation model"""
        if model_path and Path(model_path).exists():
            model = torch.load(model_path, map_location=self.device)
            return model
        
        # Create CNN for calorie estimation
        model = nn.Sequential(
            # Convolutional layers for feature extraction
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Single output: calorie estimate
        )
        
        return model.to(self.device)
    
    def _load_calorie_database(self) -> Dict:
        """Load comprehensive calorie database"""
        return {
            'fruits': self.fruit_calories,
            'vegetables': {
                'carrot': 41, 'broccoli': 34, 'potato': 77, 'tomato': 18,
                'lettuce': 15, 'cucumber': 16, 'bell_pepper': 31, 'onion': 40,
                'spinach': 23, 'corn': 86, 'peas': 81, 'cauliflower': 25
            },
            'proteins': {
                'chicken_breast': 165, 'beef': 250, 'salmon': 208, 'egg': 155,
                'pork': 242, 'turkey': 135, 'tuna': 144, 'shrimp': 99
            },
            'grains': {
                'rice': 130, 'bread': 265, 'pasta': 131, 'oatmeal': 68,
                'quinoa': 120, 'couscous': 112, 'barley': 354
            }
        }
    
    def estimate_calories(self, image: np.ndarray, food_type: str = None) -> Dict:
        """
        Estimate calories from image
        
        Args:
            image: Food image
            food_type: Type of food (optional)
            
        Returns:
            Calorie estimation analysis
        """
        # Extract visual features
        visual_features = self._extract_visual_features(image)
        
        # Detect food category
        detected_category = self._detect_food_category(image, food_type)
        
        # Estimate portion size
        portion_size = self._estimate_portion_size(image, visual_features)
        
        # Get calorie estimate
        if detected_category == 'fruit':
            calorie_estimate = self._estimate_fruit_calories(image, portion_size, detected_category)
        elif detected_category == 'restaurant_meal':
            calorie_estimate = self._estimate_restaurant_calories(image, portion_size, detected_category)
        else:
            calorie_estimate = self._estimate_general_calories(image, portion_size, detected_category)
        
        # Calculate confidence
        confidence = self._calculate_confidence(visual_features, detected_category)
        
        return {
            'calorie_estimate': calorie_estimate,
            'confidence': confidence,
            'food_category': detected_category,
            'portion_size': portion_size,
            'visual_features': visual_features,
            'method': 'visual_estimation',
            'estimation_range': self._get_estimation_range(calorie_estimate, confidence),
            'serving_size_equivalent': self._get_serving_size_equivalent(calorie_estimate, detected_category)
        }
    
    def _extract_visual_features(self, image: np.ndarray) -> Dict:
        """Extract visual features for calorie estimation"""
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate size features
        height, width = image.shape[:2]
        area = height * width
        aspect_ratio = width / height
        
        # Calculate color features
        avg_color = np.mean(image, axis=(0, 1))
        color_variance = np.var(image, axis=(0, 1))
        
        # Calculate texture features
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Calculate shape features
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_count = len(contours)
        largest_contour = max(contours, key=cv2.contourArea) if contours else None
        
        features = {
            'area': area,
            'aspect_ratio': aspect_ratio,
            'avg_color': avg_color.tolist(),
            'color_variance': color_variance.tolist(),
            'edge_density': edge_density,
            'contour_count': contour_count,
            'largest_contour_area': cv2.contourArea(largest_contour) if largest_contour is not None else 0,
            'shape_complexity': self._calculate_shape_complexity(contours)
        }
        
        return features
    
    def _detect_food_category(self, image: np.ndarray, food_type: str = None) -> str:
        """Detect food category from image"""
        if food_type:
            food_type_lower = food_type.lower()
            
            # Check for fruits
            if any(fruit in food_type_lower for fruit in ['apple', 'banana', 'orange', 'strawberry', 'grape']):
                return 'fruit'
            
            # Check for restaurant meals
            if any(meal in food_type_lower for meal in ['burger', 'pizza', 'pasta', 'salad', 'sandwich']):
                return 'restaurant_meal'
            
            # Check for proteins
            if any(protein in food_type_lower for protein in ['chicken', 'beef', 'fish', 'meat']):
                return 'protein'
        
        # Use visual analysis for category detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Fruit detection (bright colors, round shapes)
        lower_fruit = np.array([0, 50, 50])
        upper_fruit = np.array([180, 255, 255])
        fruit_mask = cv2.inRange(hsv, lower_fruit, upper_fruit)
        fruit_ratio = np.sum(fruit_mask > 0) / fruit_mask.size
        
        # Cooked meal detection (browner colors, varied textures)
        lower_meal = np.array([10, 50, 50])
        upper_meal = np.array([30, 255, 255])
        meal_mask = cv2.inRange(hsv, lower_meal, upper_meal)
        meal_ratio = np.sum(meal_mask > 0) / meal_mask.size
        
        if fruit_ratio > 0.3:
            return 'fruit'
        elif meal_ratio > 0.3:
            return 'restaurant_meal'
        else:
            return 'general'
    
    def _estimate_portion_size(self, image: np.ndarray, visual_features: Dict) -> str:
        """Estimate portion size from visual features"""
        area = visual_features['area']
        contour_area = visual_features['largest_contour_area']
        
        # Normalize area (assuming 224x224 input)
        normalized_area = area / (224 * 224)
        
        if normalized_area < 0.1:
            return 'small'
        elif normalized_area < 0.3:
            return 'medium'
        elif normalized_area < 0.6:
            return 'large'
        else:
            return 'very_large'
    
    def _estimate_fruit_calories(self, image: np.ndarray, portion_size: str, detected_category: str) -> Dict:
        """Estimate calories for fruits"""
        # Detect fruit type
        fruit_type = self._detect_fruit_type(image)
        
        # Get base calories per 100g
        base_calories = self.fruit_calories.get(fruit_type, 50)
        
        # Estimate weight based on portion size
        weight_multipliers = {
            'small': 0.8,    # ~80g
            'medium': 1.5,   # ~150g
            'large': 2.5,     # ~250g
            'very_large': 4.0  # ~400g
        }
        
        multiplier = weight_multipliers.get(portion_size, 1.0)
        estimated_calories = base_calories * multiplier
        
        return {
            'base_calories_per_100g': base_calories,
            'estimated_weight_grams': multiplier * 100,
            'estimated_calories': round(estimated_calories, 0),
            'fruit_type': fruit_type,
            'method': 'fruit_database'
        }
    
    def _estimate_restaurant_calories(self, image: np.ndarray, portion_size: str, detected_category: str) -> Dict:
        """Estimate calories for restaurant meals"""
        # Detect meal type
        meal_type = self._detect_meal_type(image)
        
        # Get calorie range
        calorie_range = self.restaurant_calories.get(meal_type, (300, 600))
        
        # Adjust based on portion size
        portion_multipliers = {
            'small': 0.7,
            'medium': 1.0,
            'large': 1.3,
            'very_large': 1.6
        }
        
        multiplier = portion_multipliers.get(portion_size, 1.0)
        estimated_calories = (calorie_range[0] + calorie_range[1]) / 2 * multiplier
        
        return {
            'meal_type': meal_type,
            'calorie_range': calorie_range,
            'estimated_calories': round(estimated_calories, 0),
            'portion_multiplier': multiplier,
            'method': 'restaurant_database'
        }
    
    def _estimate_general_calories(self, image: np.ndarray, portion_size: str, detected_category: str) -> Dict:
        """Estimate calories for general foods using ML model"""
        # Preprocess image for model
        input_tensor = self._preprocess_image(image)
        
        # Get model prediction
        with torch.no_grad():
            self.model.eval()
            prediction = self.model(input_tensor)
            estimated_calories = prediction.item()
        
        # Ensure reasonable range
        estimated_calories = max(10, min(2000, estimated_calories))
        
        return {
            'estimated_calories': round(estimated_calories, 0),
            'confidence': 0.75,
            'method': 'ml_model'
        }
    
    def _detect_fruit_type(self, image: np.ndarray) -> str:
        """Detect specific fruit type from image"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Color-based fruit detection
        fruit_colors = {
            'apple': {'lower': [0, 50, 50], 'upper': [10, 255, 255]},
            'banana': {'lower': [20, 50, 50], 'upper': [30, 255, 255]},
            'orange': {'lower': [10, 50, 50], 'upper': [25, 255, 255]},
            'strawberry': {'lower': [170, 50, 50], 'upper': [180, 255, 255]},
            'grape': {'lower': [140, 50, 50], 'upper': [170, 255, 255]}
        }
        
        best_match = 'unknown'
        best_ratio = 0
        
        for fruit, color_range in fruit_colors.items():
            mask = cv2.inRange(hsv, np.array(color_range['lower']), np.array(color_range['upper']))
            ratio = np.sum(mask > 0) / mask.size
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = fruit
        
        return best_match if best_ratio > 0.1 else 'unknown'
    
    def _detect_meal_type(self, image: np.ndarray) -> str:
        """Detect restaurant meal type from image"""
        # This would use a trained classifier in production
        # For now, use simple heuristics
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Brown/orange colors for cooked foods
        lower_cooked = np.array([10, 50, 50])
        upper_cooked = np.array([30, 255, 255])
        cooked_mask = cv2.inRange(hsv, lower_cooked, upper_cooked)
        
        # Shape analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Simple heuristics for meal type
        if len(contours) > 5:
            return 'pizza'  # Multiple toppings
        elif len(contours) < 3:
            return 'soup'  # Liquid/semi-liquid
        else:
            return 'burger'  # Default
    
    def _calculate_confidence(self, visual_features: Dict, detected_category: str) -> float:
        """Calculate confidence in calorie estimation"""
        base_confidence = 0.8
        
        # Adjust confidence based on feature quality
        if visual_features['largest_contour_area'] > 1000:
            base_confidence += 0.1
        
        if visual_features['edge_density'] > 0.1:
            base_confidence += 0.05
        
        # Adjust based on category detection confidence
        if detected_category in ['fruit', 'restaurant_meal']:
            base_confidence += 0.1
        
        return min(0.95, base_confidence)
    
    def _get_estimation_range(self, calorie_estimate: float, confidence: float) -> Tuple[float, float]:
        """Get estimation range based on confidence"""
        range_multiplier = (1.0 - confidence) * 0.3
        
        lower_bound = calorie_estimate * (1 - range_multiplier)
        upper_bound = calorie_estimate * (1 + range_multiplier)
        
        return (round(lower_bound, 0), round(upper_bound, 0))
    
    def _get_serving_size_equivalent(self, calories: float, food_category: str) -> str:
        """Get serving size equivalent"""
        if food_category == 'fruit':
            if calories < 50:
                return "Small fruit (snack size)"
            elif calories < 100:
                return "Medium fruit (standard serving)"
            else:
                return "Large fruit (meal portion)"
        
        elif food_category == 'restaurant_meal':
            if calories < 300:
                return "Light meal (appetizer size)"
            elif calories < 600:
                return "Standard meal (entree size)"
            else:
                return "Large meal (family size)"
        
        else:
            return f"{int(calories)} calories"
    
    def _calculate_shape_complexity(self, contours: List) -> float:
        """Calculate shape complexity metric"""
        if not contours:
            return 0.0
        
        total_perimeter = sum(cv2.arcLength(contour, True) for contour in contours)
        total_area = sum(cv2.contourArea(contour) for contour in contours)
        
        if total_area == 0:
            return 0.0
        
        complexity = total_perimeter / (2 * np.sqrt(np.pi * total_area))
        return complexity
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input"""
        # Resize to standard input size
        resized = cv2.resize(image, (224, 224))
        
        # Normalize
        normalized = resized.astype(np.float32) / 255.0
        
        # Convert to tensor
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
        
        return tensor.to(self.device)
    
    def analyze_complete_visual_calories(self, image: np.ndarray, food_type: str = None) -> Dict:
        """
        Complete visual calorie analysis
        
        Args:
            image: Food image
            food_type: Type of food (optional)
            
        Returns:
            Comprehensive calorie analysis
        """
        # Main calorie estimation
        calorie_analysis = self.estimate_calories(image, food_type)
        
        # Add nutritional context
        nutritional_context = self._get_nutritional_context(calorie_analysis)
        
        # Add recommendations
        recommendations = self._generate_calorie_recommendations(calorie_analysis)
        
        return {
            'calorie_analysis': calorie_analysis,
            'nutritional_context': nutritional_context,
            'recommendations': recommendations,
            'analysis_metadata': {
                'food_type': food_type,
                'method': 'visual_estimation',
                'confidence_level': 'high' if calorie_analysis['confidence'] > 0.8 else 'medium'
            }
        }
    
    def _get_nutritional_context(self, calorie_analysis: Dict) -> Dict:
        """Get nutritional context for calorie estimate"""
        calories = calorie_analysis['calorie_estimate']
        category = calorie_analysis['food_category']
        
        # Daily value context (2000 calorie diet)
        daily_percentage = (calories / 2000) * 100
        
        # Meal context
        if category == 'fruit':
            meal_context = "Healthy snack option"
        elif category == 'restaurant_meal':
            meal_context = "Complete meal"
        else:
            meal_context = "Food item"
        
        return {
            'daily_calorie_percentage': round(daily_percentage, 1),
            'meal_context': meal_context,
            'calorie_density': 'low' if calories < 100 else 'medium' if calories < 400 else 'high'
        }
    
    def _generate_calorie_recommendations(self, calorie_analysis: Dict) -> List[str]:
        """Generate recommendations based on calorie analysis"""
        calories = calorie_analysis['calorie_estimate']
        category = calorie_analysis['food_category']
        recommendations = []
        
        # General recommendations
        if calories < 50:
            recommendations.append("Light snack - suitable for weight management")
        elif calories > 800:
            recommendations.append("High calorie meal - consider portion control")
        
        # Category-specific recommendations
        if category == 'fruit':
            recommendations.append("Fresh fruit provides vitamins and fiber")
        elif category == 'restaurant_meal':
            recommendations.append("Balance with vegetables and lean proteins")
        
        # Portion recommendations
        if calorie_analysis['portion_size'] == 'very_large':
            recommendations.append("Consider sharing or saving for later")
        
        return recommendations
