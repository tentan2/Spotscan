"""
Processed Food Classifier Module
Classifies foods on a 5-level processing scale (1-5)
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path
import re

class ProcessedFoodClassifier:
    """Classifies foods on the NOVA processing scale (1-5)"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize processed food classifier
        
        Args:
            model_path: Path to pre-trained model
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.scaler = StandardScaler()
        
        # NOVA processing level definitions
        self.processing_levels = {
            1: {
                'name': 'Unprocessed or Minimally Processed Foods',
                'description': 'Whole, unaltered foods with minimal processing',
                'characteristics': ['fresh', 'whole', 'natural', 'unrefined'],
                'examples': ['fresh fruits', 'vegetables', 'meat', 'fish', 'eggs', 'milk', 'grains'],
                'indicators': ['no additives', 'minimal processing', 'whole form']
            },
            2: {
                'name': 'Processed Culinary Ingredients',
                'description': 'Ingredients derived from foods or nature by pressing, refining, grinding, milling',
                'characteristics': ['refined', 'pressed', 'milled', 'ground'],
                'examples': ['oils', 'fats', 'sugar', 'salt', 'vinegar', 'honey', 'maple syrup'],
                'indicators': ['single ingredient', 'derived from whole food', 'used in cooking']
            },
            3: {
                'name': 'Processed Foods',
                'description': 'Foods manufactured by adding salt, sugar, oil, or other Level 2 ingredients',
                'characteristics': ['preserved', 'canned', 'bottled', 'fermented'],
                'examples': ['canned vegetables', 'canned fish', 'cheese', 'fresh bread', 'beer', 'wine'],
                'indicators': ['preserved', 'fermented', 'minimal additives', 'recognizable ingredients']
            },
            4: {
                'name': 'Ultra-processed Foods (UPF)',
                'description': 'Formulations of several ingredients which, besides salt, sugars, fats and oils',
                'characteristics': ['industrial', 'formulated', 'multiple ingredients', 'additives'],
                'examples': ['chips', 'candy', 'ice cream', 'cookies', 'frozen meals', 'soft drinks'],
                'indicators': ['artificial ingredients', 'preservatives', 'flavors', 'colors']
            },
            5: {
                'name': 'Highly Ultra-processed Foods',
                'description': 'Most processed foods with extensive artificial ingredients and processing',
                'characteristics': ['highly artificial', 'extensive additives', 'synthetic'],
                'examples': ['energy drinks', 'instant meals', 'synthetic foods', 'heavily processed snacks'],
                'indicators': ['synthetic', 'extensive processing', 'artificial everything']
            }
        }
        
        # Processing indicators
        self.processing_indicators = {
            'level_1': ['fresh', 'raw', 'whole', 'natural', 'unprocessed'],
            'level_2': ['oil', 'sugar', 'salt', 'flour', 'milled', 'ground', 'pressed', 'refined'],
            'level_3': ['canned', 'bottled', 'fermented', 'preserved', 'cured', 'smoked', 'dried'],
            'level_4': ['artificial', 'flavor', 'preservative', 'color', 'emulsifier', 'stabilizer'],
            'level_5': ['synthetic', 'instant', 'hydrogenated', 'modified', 'concentrate', 'isolate']
        }
    
    def _load_model(self, model_path: Optional[str]) -> nn.Module:
        """Load or create processing level classification model"""
        if model_path and Path(model_path).exists():
            model = torch.load(model_path, map_location=self.device)
            return model
        
        # Create CNN for processing level classification
        model = nn.Sequential(
            # Feature extraction
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Classification layers
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 5),  # 5 processing levels
            nn.Softmax(dim=1)
        )
        
        return model.to(self.device)
    
    def classify_processing_level(self, image: np.ndarray, food_name: str, 
                                 ingredients_list: Optional[List[str]] = None) -> Dict:
        """
        Classify food processing level
        
        Args:
            image: Food image
            food_name: Name of the food
            ingredients_list: List of ingredients (optional)
            
        Returns:
            Processing level classification
        """
        # Visual analysis
        visual_analysis = self._analyze_visual_processing(image)
        
        # Ingredient-based analysis
        ingredient_analysis = self._analyze_ingredients_processing(ingredients_list, food_name)
        
        # Name-based analysis
        name_analysis = self._analyze_name_processing(food_name)
        
        # ML prediction
        ml_prediction = self._predict_processing_level_ml(image)
        
        # Combine all analyses
        combined_level = self._combine_processing_analyses(
            visual_analysis, ingredient_analysis, name_analysis, ml_prediction
        )
        
        # Generate detailed report
        report = self._generate_processing_report(combined_level, food_name, ingredients_list)
        
        return {
            'processing_level': combined_level,
            'level_details': self.processing_levels[combined_level],
            'visual_analysis': visual_analysis,
            'ingredient_analysis': ingredient_analysis,
            'name_analysis': name_analysis,
            'ml_prediction': ml_prediction,
            'confidence_score': self._calculate_confidence(
                visual_analysis, ingredient_analysis, name_analysis, ml_prediction
            ),
            'report': report
        }
    
    def _analyze_visual_processing(self, image: np.ndarray) -> Dict:
        """Analyze visual indicators of processing"""
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Color uniformity (processed foods often have uniform colors)
        color_uniformity = self._calculate_color_uniformity(hsv)
        
        # Texture complexity (natural foods have more texture variation)
        texture_complexity = self._calculate_texture_complexity(gray)
        
        # Shape regularity (processed foods often have regular shapes)
        shape_regularity = self._calculate_shape_regularity(gray)
        
        # Packaging indicators (highly processed foods often show packaging)
        packaging_indicators = self._detect_packaging(image)
        
        # Surface sheen (processed foods often have artificial sheen)
        surface_sheen = self._detect_surface_sheen(image)
        
        # Visual scoring for each level
        level_scores = {}
        
        # Level 1: High texture complexity, low uniformity, irregular shapes
        level_scores[1] = (texture_complexity * 0.4 + 
                          (1 - color_uniformity) * 0.3 + 
                          (1 - shape_regularity) * 0.3)
        
        # Level 2: Moderate processing indicators
        level_scores[2] = ((1 - texture_complexity) * 0.2 + 
                          color_uniformity * 0.3 + 
                          shape_regularity * 0.2 + 
                          0.3)  # Base score
        
        # Level 3: Some processing visible
        level_scores[3] = (color_uniformity * 0.4 + 
                          shape_regularity * 0.3 + 
                          (1 - texture_complexity) * 0.3)
        
        # Level 4: High uniformity, regular shapes, possible packaging
        level_scores[4] = (color_uniformity * 0.3 + 
                          shape_regularity * 0.3 + 
                          packaging_indicators * 0.2 + 
                          surface_sheen * 0.2)
        
        # Level 5: Very high uniformity, strong packaging indicators
        level_scores[5] = (color_uniformity * 0.2 + 
                          shape_regularity * 0.2 + 
                          packaging_indicators * 0.4 + 
                          surface_sheen * 0.2)
        
        return {
            'level_scores': level_scores,
            'predicted_level': max(level_scores, key=level_scores.get),
            'color_uniformity': color_uniformity,
            'texture_complexity': texture_complexity,
            'shape_regularity': shape_regularity,
            'packaging_indicators': packaging_indicators,
            'surface_sheen': surface_sheen
        }
    
    def _analyze_ingredients_processing(self, ingredients_list: Optional[List[str]], 
                                      food_name: str) -> Dict:
        """Analyze ingredients for processing indicators"""
        if not ingredients_list:
            return {'error': 'No ingredients provided'}
        
        level_scores = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        ingredient_analysis = []
        
        for ingredient in ingredients_list:
            ing_lower = ingredient.lower()
            ing_score = self._score_ingredient_processing(ing_lower)
            ingredient_analysis.append({
                'ingredient': ingredient,
                'processing_score': ing_score,
                'level': ing_score['level'],
                'indicators': ing_score['indicators']
            })
            
            # Accumulate level scores
            level_scores[ing_score['level']] += 1
        
        # Normalize scores
        total_ingredients = len(ingredients_list)
        if total_ingredients > 0:
            for level in level_scores:
                level_scores[level] = level_scores[level] / total_ingredients
        
        return {
            'level_scores': level_scores,
            'predicted_level': max(level_scores, key=level_scores.get),
            'ingredient_details': ingredient_analysis,
            'total_ingredients': total_ingredients
        }
    
    def _analyze_name_processing(self, food_name: str) -> Dict:
        """Analyze food name for processing indicators"""
        name_lower = food_name.lower()
        level_scores = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        
        # Check for processing indicators in name
        for level, indicators in self.processing_indicators.items():
            for indicator in indicators:
                if indicator in name_lower:
                    level_scores[level] += 1
        
        # Normalize scores
        total_indicators = sum(level_scores.values())
        if total_indicators > 0:
            for level in level_scores:
                level_scores[level] = level_scores[level] / total_indicators
        else:
            # Default to level 1 if no indicators found
            level_scores[1] = 1.0
        
        return {
            'level_scores': level_scores,
            'predicted_level': max(level_scores, key=level_scores.get),
            'detected_indicators': [ind for level, indicators in self.processing_indicators.items() 
                                   for ind in indicators if ind in name_lower]
        }
    
    def _predict_processing_level_ml(self, image: np.ndarray) -> Dict:
        """Predict processing level using ML model"""
        # Preprocess image
        input_tensor = self._preprocess_for_processing(image)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Perform inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = outputs.cpu().numpy()[0]
        
        # Convert to level scores
        level_scores = {i+1: float(prob) for i, prob in enumerate(probabilities)}
        predicted_level = max(level_scores, key=level_scores.get)
        
        return {
            'level_scores': level_scores,
            'predicted_level': predicted_level,
            'probabilities': probabilities.tolist()
        }
    
    def _preprocess_for_processing(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for processing level prediction"""
        # Resize to model input size
        resized = cv2.resize(image, (224, 224))
        
        # Convert to tensor
        tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float()
        tensor = tensor / 255.0  # Normalize to [0, 1]
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)
    
    def _combine_processing_analyses(self, visual: Dict, ingredient: Dict, 
                                    name: Dict, ml: Dict) -> int:
        """Combine all analyses to determine final processing level"""
        # Weight factors for different analyses
        weights = {
            'visual': 0.25,
            'ingredient': 0.35,
            'name': 0.15,
            'ml': 0.25
        }
        
        # Combine level scores
        combined_scores = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        
        # Visual analysis
        if 'level_scores' in visual:
            for level, score in visual['level_scores'].items():
                combined_scores[level] += score * weights['visual']
        
        # Ingredient analysis
        if 'level_scores' in ingredient:
            for level, score in ingredient['level_scores'].items():
                combined_scores[level] += score * weights['ingredient']
        
        # Name analysis
        if 'level_scores' in name:
            for level, score in name['level_scores'].items():
                combined_scores[level] += score * weights['name']
        
        # ML prediction
        if 'level_scores' in ml:
            for level, score in ml['level_scores'].items():
                combined_scores[level] += score * weights['ml']
        
        # Return level with highest score
        return max(combined_scores, key=combined_scores.get)
    
    def _calculate_color_uniformity(self, hsv: np.ndarray) -> float:
        """Calculate color uniformity"""
        h_std = np.std(hsv[:, :, 0])
        s_std = np.std(hsv[:, :, 1])
        v_std = np.std(hsv[:, :, 2])
        
        uniformity = 1.0 / (1.0 + (h_std + s_std + v_std) / 3.0)
        return uniformity
    
    def _calculate_texture_complexity(self, gray: np.ndarray) -> float:
        """Calculate texture complexity"""
        # Use entropy as complexity measure
        histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])
        histogram = histogram.flatten()
        histogram = histogram / histogram.sum()
        
        # Remove zero values
        histogram = histogram[histogram > 0]
        
        entropy = -np.sum(histogram * np.log2(histogram))
        return entropy / 8.0  # Normalize to 0-1 range
    
    def _calculate_shape_regularity(self, gray: np.ndarray) -> float:
        """Calculate shape regularity"""
        # Find contours
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.5  # Default
        
        # Analyze main contour
        main_contour = max(contours, key=cv2.contourArea)
        
        # Calculate circularity
        area = cv2.contourArea(main_contour)
        perimeter = cv2.arcLength(main_contour, True)
        
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        else:
            circularity = 0
        
        return circularity
    
    def _detect_packaging(self, image: np.ndarray) -> float:
        """Detect packaging indicators"""
        # Look for straight lines, text, barcodes (packaging indicators)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Line detection
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
        # More lines = more likely packaged
        line_count = len(lines) if lines is not None else 0
        packaging_score = min(line_count / 10.0, 1.0)
        
        return packaging_score
    
    def _detect_surface_sheen(self, image: np.ndarray) -> float:
        """Detect artificial surface sheen"""
        # Look for specular highlights
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # High value areas indicate sheen
        value_channel = hsv[:, :, 2]
        bright_areas = np.sum(value_channel > 200)
        sheen_score = bright_areas / (image.shape[0] * image.shape[1])
        
        return min(sheen_score * 5, 1.0)  # Scale to 0-1
    
    def _score_ingredient_processing(self, ingredient: str) -> Dict:
        """Score individual ingredient for processing level"""
        level_scores = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        detected_indicators = []
        
        # Check against processing indicators
        for level, indicators in self.processing_indicators.items():
            for indicator in indicators:
                if indicator in ingredient:
                    level_scores[level] += 1
                    detected_indicators.append(indicator)
        
        # Determine primary level
        if level_scores[5] > 0:
            primary_level = 5
        elif level_scores[4] > 0:
            primary_level = 4
        elif level_scores[3] > 0:
            primary_level = 3
        elif level_scores[2] > 0:
            primary_level = 2
        else:
            primary_level = 1
        
        return {
            'level': primary_level,
            'level_scores': level_scores,
            'indicators': detected_indicators
        }
    
    def _calculate_confidence(self, visual: Dict, ingredient: Dict, 
                            name: Dict, ml: Dict) -> float:
        """Calculate overall confidence in classification"""
        confidences = []
        
        if 'predicted_level' in visual:
            max_visual_score = max(visual['level_scores'].values())
            confidences.append(max_visual_score)
        
        if 'predicted_level' in ingredient:
            max_ingredient_score = max(ingredient['level_scores'].values())
            confidences.append(max_ingredient_score)
        
        if 'predicted_level' in name:
            max_name_score = max(name['level_scores'].values())
            confidences.append(max_name_score)
        
        if 'predicted_level' in ml:
            max_ml_score = max(ml['level_scores'].values())
            confidences.append(max_ml_score)
        
        if confidences:
            return np.mean(confidences)
        else:
            return 0.5  # Default confidence
    
    def _generate_processing_report(self, level: int, food_name: str, 
                                 ingredients_list: Optional[List[str]]) -> Dict:
        """Generate detailed processing report"""
        level_info = self.processing_levels[level]
        
        report = {
            'food_name': food_name,
            'processing_level': level,
            'level_name': level_info['name'],
            'description': level_info['description'],
            'characteristics': level_info['characteristics'],
            'examples': level_info['examples'],
            'health_implications': self._get_health_implications(level),
            'recommendations': self._get_processing_recommendations(level),
            'alternatives': self._get_healthier_alternatives(level, food_name)
        }
        
        return report
    
    def _get_health_implications(self, level: int) -> List[str]:
        """Get health implications for processing level"""
        implications = {
            1: [
                "Generally considered healthy",
                "High in nutrients and fiber",
                "Minimal additives and preservatives",
                "Closer to natural state"
            ],
            2: [
                "Generally healthy in moderation",
                "May be refined (loss of some nutrients)",
                "Single ingredient foods",
                "Often used in cooking"
            ],
            3: [
                "Moderately healthy",
                "May contain added salt, sugar, or oil",
                "Some nutrient loss from processing",
                "Still recognizable food form"
            ],
            4: [
                "Often less healthy",
                "High in additives, preservatives, artificial ingredients",
                "May contain excessive sugar, salt, unhealthy fats",
                "Linked to overeating and health issues"
            ],
            5: [
                "Generally unhealthy",
                "Extremely processed with synthetic ingredients",
                "Highly palatable but low nutritional value",
                "Strongly associated with chronic diseases"
            ]
        }
        
        return implications.get(level, [])
    
    def _get_processing_recommendations(self, level: int) -> List[str]:
        """Get recommendations based on processing level"""
        recommendations = {
            1: [
                "Continue choosing whole, unprocessed foods",
                "Eat a variety of fresh foods",
                "Minimal cooking to preserve nutrients"
            ],
            2: [
                "Use in moderation as part of balanced diet",
                "Choose unrefined versions when possible",
                "Combine with whole foods"
            ],
            3: [
                "Limit consumption",
                "Check for added sugars and sodium",
                "Choose versions with fewer additives"
            ],
            4: [
                "Significantly limit consumption",
                "Read labels carefully",
                "Choose whole food alternatives",
                "Be aware of addictive properties"
            ],
            5: [
                "Avoid or severely limit consumption",
                "These foods offer minimal nutritional value",
                "Strongly associated with health problems",
                "Replace with whole food alternatives"
            ]
        }
        
        return recommendations.get(level, [])
    
    def _get_healthier_alternatives(self, level: int, food_name: str) -> List[str]:
        """Get healthier alternatives for processed foods"""
        if level <= 2:
            return ["This food is already relatively healthy"]
        
        alternatives = {
            3: [
                "Fresh or frozen versions without additives",
                "Homemade versions with controlled ingredients"
            ],
            4: [
                "Whole food alternatives",
                "Homemade versions with natural ingredients",
                "Less processed brands with fewer additives"
            ],
            5: [
                "Completely avoid this food",
                "Whole food alternatives",
                "Fresh, unprocessed options"
            ]
        }
        
        return alternatives.get(level, [])
