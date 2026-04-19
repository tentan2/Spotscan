"""
Nutrition Analyzer Module
Estimates comprehensive nutritional information from food images and data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path
import cv2
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle

class NutritionAnalyzer:
    """Analyzes and estimates nutritional content of food items"""
    
    def __init__(self, nutrition_data_path: Optional[str] = None):
        """
        Initialize nutrition analyzer
        
        Args:
            nutrition_data_path: Path to nutrition dataset
        """
        self.nutrition_data = self._load_nutrition_data(nutrition_data_path)
        self.portion_estimator = self._init_portion_estimator()
        self.scaler = StandardScaler()
        
        # Standard nutrition components to analyze
        self.nutrition_components = [
            'calories', 'protein', 'carbohydrates', 'fat', 'fiber', 'sugar',
            'sodium', 'cholesterol', 'calcium', 'iron', 'potassium',
            'vitamin_a', 'vitamin_c', 'vitamin_d', 'omega_3', 'caffeine'
        ]
    
    def _load_nutrition_data(self, data_path: Optional[str]) -> pd.DataFrame:
        """Load nutrition dataset from file"""
        if data_path and Path(data_path).exists():
            return pd.read_csv(data_path)
        
        # Create default nutrition database with common foods
        default_data = {
            'food_name': [
                'apple', 'banana', 'orange', 'strawberry', 'grape',
                'broccoli', 'carrot', 'spinach', 'potato', 'tomato',
                'chicken_breast', 'beef', 'salmon', 'egg', 'cheese',
                'rice', 'bread', 'pasta', 'oatmeal', 'yogurt'
            ],
            'calories_per_100g': [
                52, 89, 47, 32, 69, 34, 41, 23, 77, 18,
                165, 250, 208, 155, 402, 130, 265, 131, 68, 59
            ],
            'protein_per_100g': [
                0.3, 1.1, 0.9, 0.7, 0.7, 2.8, 0.9, 2.9, 2.0, 0.9,
                31.0, 26.0, 20.0, 13.0, 25.0, 2.7, 9.0, 5.0, 2.4, 10.0
            ],
            'carbs_per_100g': [
                14.0, 23.0, 12.0, 8.0, 18.0, 7.0, 10.0, 3.6, 17.0, 3.9,
                0.0, 0.0, 0.0, 1.1, 1.3, 28.0, 49.0, 25.0, 12.0, 3.6
            ],
            'fat_per_100g': [
                0.2, 0.3, 0.1, 0.3, 0.2, 0.4, 0.2, 0.4, 0.1, 0.2,
                3.6, 15.0, 13.0, 11.0, 33.0, 0.3, 3.2, 1.1, 1.4, 3.3
            ],
            'fiber_per_100g': [
                2.4, 2.6, 2.4, 2.0, 0.9, 2.6, 2.8, 2.2, 2.2, 1.2,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 2.7, 1.8, 1.7, 0.0
            ],
            'sugar_per_100g': [
                10.4, 12.2, 9.4, 4.9, 15.5, 1.7, 4.7, 0.4, 0.8, 2.6,
                0.0, 0.0, 0.0, 1.1, 0.5, 0.1, 5.0, 0.6, 0.6, 3.2
            ]
        }
        
        return pd.DataFrame(default_data)
    
    def _init_portion_estimator(self) -> RandomForestRegressor:
        """Initialize ML model for portion size estimation"""
        # This would be trained on actual data in a real implementation
        return RandomForestRegressor(n_estimators=100, random_state=42)
    
    def estimate_portion_size(self, image: np.ndarray, food_class: str) -> float:
        """
        Estimate portion size in grams from image
        
        Args:
            image: Food image
            food_class: Detected food class
            
        Returns:
            Estimated weight in grams
        """
        # Extract features from image for portion estimation
        features = self._extract_portion_features(image)
        
        # For now, use heuristic-based estimation
        # In a real implementation, this would use the trained ML model
        height, width = image.shape[:2]
        pixel_count = height * width
        
        # Base estimation on image size and food type
        base_weights = {
            'fruit': 150,    # Average fruit weight
            'vegetable': 100,  # Average vegetable weight
            'meat': 200,     # Average meat portion
            'grain': 180,    # Average grain portion
            'dairy': 120,    # Average dairy portion
            'default': 150
        }
        
        # Determine food category
        category = self._classify_food_category(food_class)
        base_weight = base_weights.get(category, base_weights['default'])
        
        # Adjust based on image size (simple heuristic)
        size_factor = np.log1p(pixel_count / 10000)  # Normalize pixel count
        estimated_weight = base_weight * size_factor
        
        return max(50, min(500, estimated_weight))  # Clamp to reasonable range
    
    def _extract_portion_features(self, image: np.ndarray) -> np.ndarray:
        """Extract features for portion size estimation"""
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Calculate various features
        features = [
            image.shape[0],  # height
            image.shape[1],  # width
            image.shape[0] * image.shape[1],  # total pixels
            np.mean(image),  # mean intensity
            np.std(image),   # std intensity
            np.mean(hsv[:, :, 0]),  # mean hue
            np.mean(lab[:, :, 0]),  # mean lightness
        ]
        
        return np.array(features)
    
    def _classify_food_category(self, food_class: str) -> str:
        """Classify food into broad categories"""
        food_class_lower = food_class.lower()
        
        if any(fruit in food_class_lower for fruit in ['apple', 'banana', 'orange', 'strawberry', 'grape', 'fruit']):
            return 'fruit'
        elif any(veg in food_class_lower for veg in ['broccoli', 'carrot', 'spinach', 'vegetable', 'salad']):
            return 'vegetable'
        elif any(meat in food_class_lower for meat in ['chicken', 'beef', 'pork', 'meat', 'fish']):
            return 'meat'
        elif any(grain in food_class_lower for grain in ['rice', 'bread', 'pasta', 'grain', 'cereal']):
            return 'grain'
        elif any(dairy in food_class_lower for dairy in ['cheese', 'milk', 'yogurt', 'dairy']):
            return 'dairy'
        else:
            return 'default'
    
    def get_nutrition_for_food(self, food_name: str, weight_grams: float) -> Dict:
        """
        Get nutrition information for a specific food and weight
        
        Args:
            food_name: Name of the food
            weight_grams: Weight in grams
            
        Returns:
            Nutrition information dictionary
        """
        # Find food in database
        food_data = self.nutrition_data[
            self.nutrition_data['food_name'].str.contains(food_name, case=False, na=False)
        ]
        
        if food_data.empty:
            # Try to find similar food
            food_data = self._find_similar_food(food_name)
        
        if food_data.empty:
            # Return default values if not found
            return self._get_default_nutrition(weight_grams)
        
        # Get the first match
        food_info = food_data.iloc[0]
        
        # Calculate nutrition based on weight
        factor = weight_grams / 100.0  # Convert from per-100g to actual weight
        
        nutrition = {
            'food_name': food_name,
            'weight_grams': weight_grams,
            'calories': round(food_info['calories_per_100g'] * factor, 1),
            'protein': round(food_info['protein_per_100g'] * factor, 1),
            'carbohydrates': round(food_info['carbs_per_100g'] * factor, 1),
            'fat': round(food_info['fat_per_100g'] * factor, 1),
            'fiber': round(food_info['fiber_per_100g'] * factor, 1),
            'sugar': round(food_info['sugar_per_100g'] * factor, 1),
            'sodium': self._estimate_sodium(food_name, weight_grams),
            'cholesterol': self._estimate_cholesterol(food_name, weight_grams),
            'calcium': self._estimate_mineral('calcium', food_name, weight_grams),
            'iron': self._estimate_mineral('iron', food_name, weight_grams),
            'potassium': self._estimate_mineral('potassium', food_name, weight_grams),
            'vitamin_a': self._estimate_vitamin('a', food_name, weight_grams),
            'vitamin_c': self._estimate_vitamin('c', food_name, weight_grams),
            'vitamin_d': self._estimate_vitamin('d', food_name, weight_grams),
            'omega_3': self._estimate_omega3(food_name, weight_grams),
            'caffeine': self._estimate_caffeine(food_name, weight_grams)
        }
        
        return nutrition
    
    def _find_similar_food(self, food_name: str) -> pd.DataFrame:
        """Find similar food in database"""
        food_name_lower = food_name.lower()
        
        # Simple keyword matching
        for _, row in self.nutrition_data.iterrows():
            db_food = row['food_name'].lower()
            if any(keyword in db_food for keyword in food_name_lower.split('_')):
                return pd.DataFrame([row])
        
        return pd.DataFrame()
    
    def _get_default_nutrition(self, weight_grams: float) -> Dict:
        """Get default nutrition values"""
        factor = weight_grams / 100.0
        
        return {
            'food_name': 'unknown',
            'weight_grams': weight_grams,
            'calories': round(150 * factor, 1),
            'protein': round(10 * factor, 1),
            'carbohydrates': round(20 * factor, 1),
            'fat': round(5 * factor, 1),
            'fiber': round(2 * factor, 1),
            'sugar': round(8 * factor, 1),
            'sodium': round(50 * factor, 1),
            'cholesterol': round(20 * factor, 1),
            'calcium': round(100 * factor, 1),
            'iron': round(1.5 * factor, 1),
            'potassium': round(200 * factor, 1),
            'vitamin_a': round(500 * factor, 1),
            'vitamin_c': round(30 * factor, 1),
            'vitamin_d': round(2 * factor, 1),
            'omega_3': round(0.1 * factor, 1),
            'caffeine': 0.0
        }
    
    def _estimate_sodium(self, food_name: str, weight_grams: float) -> float:
        """Estimate sodium content"""
        # Sodium estimates based on food type
        sodium_factors = {
            'processed': 400,  # mg per 100g
            'meat': 70,
            'vegetable': 20,
            'fruit': 1,
            'dairy': 50,
            'grain': 5,
            'default': 50
        }
        
        category = self._classify_food_category(food_name)
        factor = sodium_factors.get(category, sodium_factors['default'])
        
        return round(factor * weight_grams / 100.0, 1)
    
    def _estimate_cholesterol(self, food_name: str, weight_grams: float) -> float:
        """Estimate cholesterol content"""
        cholesterol_factors = {
            'meat': 80,   # mg per 100g
            'dairy': 30,
            'egg': 370,
            'fish': 50,
            'default': 0
        }
        
        food_name_lower = food_name.lower()
        
        if 'egg' in food_name_lower:
            factor = cholesterol_factors['egg']
        elif category := self._classify_food_category(food_name):
            factor = cholesterol_factors.get(category, cholesterol_factors['default'])
        else:
            factor = cholesterol_factors['default']
        
        return round(factor * weight_grams / 100.0, 1)
    
    def _estimate_mineral(self, mineral: str, food_name: str, weight_grams: float) -> float:
        """Estimate mineral content"""
        # Base values for different minerals (per 100g)
        mineral_bases = {
            'calcium': {'dairy': 120, 'vegetable': 50, 'fruit': 10, 'default': 30},
            'iron': {'meat': 2.5, 'vegetable': 1.5, 'grain': 1.0, 'default': 1.0},
            'potassium': {'fruit': 200, 'vegetable': 300, 'meat': 250, 'default': 150}
        }
        
        category = self._classify_food_category(food_name)
        base_value = mineral_bases[mineral].get(category, mineral_bases[mineral]['default'])
        
        return round(base_value * weight_grams / 100.0, 1)
    
    def _estimate_vitamin(self, vitamin: str, food_name: str, weight_grams: float) -> float:
        """Estimate vitamin content"""
        # Base values for vitamins (per 100g)
        vitamin_bases = {
            'a': {'vegetable': 500, 'fruit': 100, 'dairy': 150, 'default': 200},
            'c': {'fruit': 50, 'vegetable': 30, 'default': 10},
            'd': {'dairy': 2, 'fish': 10, 'default': 1}
        }
        
        category = self._classify_food_category(food_name)
        base_value = vitamin_bases[vitamin].get(category, vitamin_bases[vitamin]['default'])
        
        return round(base_value * weight_grams / 100.0, 1)
    
    def _estimate_omega3(self, food_name: str, weight_grams: float) -> float:
        """Estimate omega-3 content"""
        food_name_lower = food_name.lower()
        
        if 'salmon' in food_name_lower or 'fish' in food_name_lower:
            base = 2.5  # g per 100g
        elif 'walnut' in food_name_lower or 'nut' in food_name_lower:
            base = 9.0  # g per 100g
        else:
            base = 0.1  # g per 100g
        
        return round(base * weight_grams / 100.0, 2)
    
    def _estimate_caffeine(self, food_name: str, weight_grams: float) -> float:
        """Estimate caffeine content"""
        food_name_lower = food_name.lower()
        
        if 'coffee' in food_name_lower:
            base = 40  # mg per 100g
        elif 'tea' in food_name_lower:
            base = 20  # mg per 100g
        elif 'chocolate' in food_name_lower:
            base = 70  # mg per 100g
        else:
            base = 0
        
        return round(base * weight_grams / 100.0, 1)
    
    def analyze_complete_nutrition(self, image: np.ndarray, food_class: str, 
                                 food_name: Optional[str] = None) -> Dict:
        """
        Complete nutrition analysis from image
        
        Args:
            image: Food image
            food_class: Detected food class
            food_name: Specific food name (optional)
            
        Returns:
            Complete nutrition analysis
        """
        # Estimate portion size
        estimated_weight = self.estimate_portion_size(image, food_class)
        
        # Use provided food name or derive from class
        if food_name is None:
            food_name = food_class.replace('_', ' ')
        
        # Get nutrition information
        nutrition = self.get_nutrition_for_food(food_name, estimated_weight)
        
        # Add analysis metadata
        nutrition['analysis_metadata'] = {
            'food_class': food_class,
            'confidence': 'high',  # Would come from detection model
            'portion_estimation_method': 'ml_heuristic',
            'data_source': 'nutrition_database'
        }
        
        return nutrition
    
    def calculate_daily_values_percentage(self, nutrition: Dict) -> Dict:
        """
        Calculate percentage of daily values for each nutrient
        
        Args:
            nutrition: Nutrition information
            
        Returns:
            Daily value percentages
        """
        # Daily value references (based on 2000 calorie diet)
        daily_values = {
            'calories': 2000,
            'protein': 50,
            'carbohydrates': 300,
            'fat': 65,
            'fiber': 25,
            'sodium': 2300,
            'cholesterol': 300,
            'calcium': 1300,
            'iron': 18,
            'potassium': 4700,
            'vitamin_a': 900,
            'vitamin_c': 90,
            'vitamin_d': 20
        }
        
        dv_percentages = {}
        
        for nutrient, dv in daily_values.items():
            if nutrient in nutrition:
                percentage = (nutrition[nutrient] / dv) * 100
                dv_percentages[f'{nutrient}_dv_percent'] = round(percentage, 1)
        
        return dv_percentages
