"""
Enhanced Visual Estimator Module
Advanced visual analysis for ingredients, nutrition facts, portion sizes, and drink volumes
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path
import json
import math

class EnhancedVisualEstimator:
    """Advanced visual estimation for comprehensive food analysis"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize enhanced visual estimator
        
        Args:
            model_path: Path to pre-trained models
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = self._load_models(model_path)
        self.scalers = {}
        
        # Visual databases
        self.ingredient_visual_database = self._load_ingredient_visual_database()
        self.plate_size_database = self._load_plate_size_database()
        self.drink_volume_database = self._load_drink_volume_database()
        self.nutrition_facts_database = self._load_nutrition_facts_database()
        
        # Standard measurements
        self.standard_plate_diameter = 25  # cm (10 inches)
        self.standard_glass_diameter = 8   # cm (3.2 inches)
        self.pixel_to_cm_ratio = None
        
    def _load_models(self, model_path: Optional[str]) -> Dict:
        """Load all visual estimation models"""
        models = {}
        
        # Ingredient detection model
        models['ingredient'] = self._create_ingredient_model()
        
        # Portion size estimation model
        models['portion'] = self._create_portion_model()
        
        # Drink volume estimation model
        models['volume'] = self._create_volume_model()
        
        # Nutrition facts recognition model
        models['nutrition_facts'] = self._create_nutrition_facts_model()
        
        return models
    
    def _create_ingredient_model(self) -> nn.Module:
        """Create ingredient detection model"""
        model = nn.Sequential(
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
            nn.Linear(128, 50)  # 50 common ingredients
        )
        return model.to(self.device)
    
    def _create_portion_model(self) -> nn.Module:
        """Create portion size estimation model"""
        model = nn.Sequential(
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
            nn.Linear(64, 1)  # Portion size in grams
        )
        return model.to(self.device)
    
    def _create_volume_model(self) -> nn.Module:
        """Create drink volume estimation model"""
        model = nn.Sequential(
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
            nn.Linear(64, 1)  # Volume in ml
        )
        return model.to(self.device)
    
    def _create_nutrition_facts_model(self) -> nn.Module:
        """Create nutrition facts recognition model"""
        model = nn.Sequential(
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
            nn.Linear(64, 20)  # 20 nutrition values
        )
        return model.to(self.device)
    
    def _load_ingredient_visual_database(self) -> Dict:
        """Load visual ingredient database"""
        return {
            'vegetables': {
                'lettuce': {'color': [120, 200, 120], 'texture': 'leafy', 'density': 0.25},
                'tomato': {'color': [200, 50, 50], 'texture': 'smooth', 'density': 0.95},
                'carrot': {'color': [255, 140, 0], 'texture': 'crisp', 'density': 0.65},
                'broccoli': {'color': [50, 150, 50], 'texture': 'floret', 'density': 0.35},
                'onion': {'color': [255, 255, 200], 'texture': 'layered', 'density': 0.45},
                'pepper': {'color': [255, 100, 0], 'texture': 'crisp', 'density': 0.95},
                'cucumber': {'color': [100, 200, 100], 'texture': 'smooth', 'density': 0.95},
                'spinach': {'color': [50, 150, 50], 'texture': 'leafy', 'density': 0.15}
            },
            'proteins': {
                'chicken': {'color': [200, 150, 100], 'texture': 'fibrous', 'density': 1.0},
                'beef': {'color': [150, 100, 80], 'texture': 'marbled', 'density': 1.05},
                'fish': {'color': [200, 200, 230], 'texture': 'flaky', 'density': 0.9},
                'egg': {'color': [255, 255, 200], 'texture': 'smooth', 'density': 1.0},
                'tofu': {'color': [240, 240, 230], 'texture': 'soft', 'density': 0.95},
                'beans': {'color': [150, 100, 50], 'texture': 'firm', 'density': 1.2}
            },
            'carbs': {
                'rice': {'color': [255, 250, 230], 'texture': 'grainy', 'density': 0.85},
                'pasta': {'color': [255, 240, 200], 'texture': 'smooth', 'density': 1.0},
                'bread': {'color': [240, 200, 150], 'texture': 'spongy', 'density': 0.4},
                'potato': {'color': [240, 220, 180], 'texture': 'starchy', 'density': 1.1},
                'quinoa': {'color': [230, 210, 180], 'texture': 'grainy', 'density': 0.85}
            },
            'fruits': {
                'apple': {'color': [255, 100, 100], 'texture': 'crisp', 'density': 0.9},
                'banana': {'color': [255, 220, 100], 'texture': 'soft', 'density': 0.95},
                'orange': {'color': [255, 150, 0], 'texture': 'juicy', 'density': 0.95},
                'strawberry': {'color': [255, 50, 100], 'texture': 'seedy', 'density': 0.6},
                'grape': {'color': [150, 50, 150], 'texture': 'juicy', 'density': 1.0}
            }
        }
    
    def _load_plate_size_database(self) -> Dict:
        """Load plate size reference database"""
        return {
            'standard_dinner_plate': {'diameter_cm': 25, 'area_cm2': 490},
            'salad_plate': {'diameter_cm': 20, 'area_cm2': 314},
            'dessert_plate': {'diameter_cm': 15, 'area_cm2': 177},
            'bread_plate': {'diameter_cm': 12, 'area_cm2': 113},
            'sauce_plate': {'diameter_cm': 10, 'area_cm2': 79},
            'standard_bowl': {'diameter_cm': 18, 'depth_cm': 5, 'volume_cm3': 1272},
            'soup_bowl': {'diameter_cm': 20, 'depth_cm': 7, 'volume_cm3': 2199},
            'cereal_bowl': {'diameter_cm': 15, 'depth_cm': 6, 'volume_cm3': 1060}
        }
    
    def _load_drink_volume_database(self) -> Dict:
        """Load drink volume reference database"""
        return {
            'standard_glass': {'diameter_cm': 8, 'height_cm': 15, 'volume_ml': 250},
            'wine_glass': {'diameter_cm': 7, 'height_cm': 18, 'volume_ml': 200},
            'coffee_mug': {'diameter_cm': 8, 'height_cm': 10, 'volume_ml': 350},
            'water_glass': {'diameter_cm': 6, 'height_cm': 20, 'volume_ml': 300},
            'juice_glass': {'diameter_cm': 7, 'height_cm': 12, 'volume_ml': 200},
            'soda_can': {'diameter_cm': 6.5, 'height_cm': 12, 'volume_ml': 355},
            'water_bottle': {'diameter_cm': 7, 'height_cm': 22, 'volume_ml': 500},
            'sports_bottle': {'diameter_cm': 8, 'height_cm': 25, 'volume_ml': 750}
        }
    
    def _load_nutrition_facts_database(self) -> Dict:
        """Load nutrition facts reference database"""
        return {
            'serving_sizes': {
                'per_100g': 'standard',
                'per_cup': '240ml',
                'per_piece': 'individual',
                'per_package': 'full_package'
            },
            'daily_values': {
                'calories': 2000,
                'fat': 65, 'saturated_fat': 20,
                'cholesterol': 300, 'sodium': 2300,
                'carbohydrates': 300, 'fiber': 25, 'sugars': 50,
                'protein': 50
            }
        }
    
    def analyze_enhanced_visual(self, image: np.ndarray, food_type: str = None) -> Dict:
        """
        Perform comprehensive enhanced visual analysis
        
        Args:
            image: Food image
            food_type: Type of food (optional)
            
        Returns:
            Comprehensive visual analysis results
        """
        # Detect plate/glass boundaries
        plate_info = self._detect_plate_boundaries(image)
        drink_info = self._detect_drink_boundaries(image)
        
        # Estimate pixel to cm ratio
        self.pixel_to_cm_ratio = self._estimate_pixel_to_cm_ratio(image, plate_info, drink_info)
        
        # Analyze ingredients visually
        ingredient_analysis = self._analyze_visual_ingredients(image)
        
        # Estimate portion sizes
        portion_analysis = self._estimate_visual_portions(image, plate_info)
        
        # Estimate drink volumes
        volume_analysis = self._estimate_visual_volumes(image, drink_info)
        
        # Recognize nutrition facts
        nutrition_facts_analysis = self._recognize_nutrition_facts(image)
        
        # Combine all analyses
        combined_analysis = self._combine_visual_analyses(
            ingredient_analysis, portion_analysis, 
            volume_analysis, nutrition_facts_analysis
        )
        
        return {
            'ingredient_analysis': ingredient_analysis,
            'portion_analysis': portion_analysis,
            'volume_analysis': volume_analysis,
            'nutrition_facts_analysis': nutrition_facts_analysis,
            'combined_analysis': combined_analysis,
            'visual_metadata': {
                'pixel_to_cm_ratio': self.pixel_to_cm_ratio,
                'plate_info': plate_info,
                'drink_info': drink_info,
                'confidence': self._calculate_overall_confidence(combined_analysis)
            }
        }
    
    def _detect_plate_boundaries(self, image: np.ndarray) -> Dict:
        """Detect plate boundaries and size"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter for circular/oval shapes (plates)
        plate_candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Minimum size
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * math.pi * area / (perimeter * perimeter)
                    if circularity > 0.5:  # Reasonably circular
                        plate_candidates.append({
                            'contour': contour,
                            'area': area,
                            'circularity': circularity,
                            'center': self._get_contour_center(contour)
                        })
        
        # Select best plate candidate
        if plate_candidates:
            best_plate = max(plate_candidates, key=lambda x: x['area'])
            return {
                'detected': True,
                'center': best_plate['center'],
                'area_pixels': best_plate['area'],
                'circularity': best_plate['circularity'],
                'estimated_diameter_pixels': self._estimate_diameter_from_area(best_plate['area'])
            }
        
        return {'detected': False}
    
    def _detect_drink_boundaries(self, image: np.ndarray) -> Dict:
        """Detect drink/glass boundaries"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter for rectangular/oval shapes (glasses)
        drink_candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Minimum size
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    # Check for glass-like shapes
                    rect = cv2.minAreaRect(contour)
                    aspect_ratio = max(rect[1]) / min(rect[1]) if min(rect[1]) > 0 else 1
                    
                    # Glasses typically have aspect ratio > 1.5
                    if 1.5 < aspect_ratio < 5:
                        drink_candidates.append({
                            'contour': contour,
                            'area': area,
                            'aspect_ratio': aspect_ratio,
                            'center': self._get_contour_center(contour),
                            'rect': rect
                        })
        
        # Select best drink candidate
        if drink_candidates:
            best_drink = max(drink_candidates, key=lambda x: x['area'])
            return {
                'detected': True,
                'center': best_drink['center'],
                'area_pixels': best_drink['area'],
                'aspect_ratio': best_drink['aspect_ratio'],
                'rect': best_drink['rect']
            }
        
        return {'detected': False}
    
    def _estimate_pixel_to_cm_ratio(self, image: np.ndarray, plate_info: Dict, drink_info: Dict) -> float:
        """Estimate pixel to cm conversion ratio"""
        # Try to use plate as reference
        if plate_info['detected']:
            # Assume standard dinner plate (25cm diameter)
            estimated_diameter_pixels = plate_info['estimated_diameter_pixels']
            if estimated_diameter_pixels > 0:
                return self.standard_plate_diameter / estimated_diameter_pixels
        
        # Try to use glass as reference
        if drink_info['detected']:
            # Assume standard glass (8cm diameter)
            rect = drink_info['rect']
            width_pixels = max(rect[1])
            if width_pixels > 0:
                return self.standard_glass_diameter / width_pixels
        
        # Default estimation based on image size
        image_height, image_width = image.shape[:2]
        # Assume average plate takes up 60% of image width
        estimated_plate_pixels = image_width * 0.6
        return self.standard_plate_diameter / estimated_plate_pixels
    
    def _analyze_visual_ingredients(self, image: np.ndarray) -> Dict:
        """Analyze ingredients visually"""
        # Segment image into regions
        segments = self._segment_image_regions(image)
        
        detected_ingredients = []
        total_coverage = 0
        
        for segment in segments:
            # Extract features from segment
            features = self._extract_segment_features(segment)
            
            # Match against ingredient database
            ingredient_match = self._match_ingredient_to_database(features)
            
            if ingredient_match:
                # Calculate coverage percentage
                segment_area = cv2.countNonZero(segment['mask'])
                image_area = image.shape[0] * image.shape[1]
                coverage_percentage = (segment_area / image_area) * 100
                
                detected_ingredients.append({
                    'name': ingredient_match['name'],
                    'category': ingredient_match['category'],
                    'confidence': ingredient_match['confidence'],
                    'coverage_percentage': coverage_percentage,
                    'estimated_weight_grams': self._estimate_ingredient_weight(
                        segment_area, self.pixel_to_cm_ratio, ingredient_match['density']
                    ),
                    'visual_features': features
                })
                
                total_coverage += coverage_percentage
        
        return {
            'detected_ingredients': detected_ingredients,
            'total_coverage_percentage': total_coverage,
            'ingredient_count': len(detected_ingredients),
            'dominant_ingredient': max(detected_ingredients, key=lambda x: x['coverage_percentage']) if detected_ingredients else None
        }
    
    def _segment_image_regions(self, image: np.ndarray) -> List[Dict]:
        """Segment image into distinct regions"""
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Use color-based segmentation
        segments = []
        
        # Define color ranges for common food colors
        color_ranges = [
            ([0, 50, 50], [10, 255, 255]),    # Red/orange
            ([10, 50, 50], [25, 255, 255]),   # Yellow/orange
            ([35, 50, 50], [85, 255, 255]),   # Green
            ([100, 50, 50], [130, 255, 255]),  # Blue
            ([140, 50, 50], [170, 255, 255]),  # Purple
            ([0, 0, 200], [180, 30, 255]),     # White/light
            ([0, 0, 0], [180, 255, 30])        # Black/dark
        ]
        
        for i, (lower, upper) in enumerate(color_ranges):
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            
            # Clean up mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours in mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Minimum area threshold
                    segment_mask = np.zeros(mask.shape, dtype=np.uint8)
                    cv2.drawContours(segment_mask, [contour], -1, 255, -1)
                    
                    # Extract segment from original image
                    segment_image = cv2.bitwise_and(image, image, mask=segment_mask)
                    
                    segments.append({
                        'mask': segment_mask,
                        'image': segment_image,
                        'contour': contour,
                        'area': area,
                        'color_range_index': i
                    })
        
        return segments
    
    def _extract_segment_features(self, segment: Dict) -> Dict:
        """Extract visual features from image segment"""
        segment_image = segment['image']
        mask = segment['mask']
        
        # Only consider pixels within the segment
        segment_pixels = segment_image[mask > 0]
        
        if len(segment_pixels) == 0:
            return {}
        
        # Color features
        avg_color = np.mean(segment_pixels, axis=0)
        color_std = np.std(segment_pixels, axis=0)
        
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(segment_image, cv2.COLOR_BGR2HSV)
        hsv_pixels = hsv[mask > 0]
        avg_hsv = np.mean(hsv_pixels, axis=0)
        
        # Texture features
        gray = cv2.cvtColor(segment_image, cv2.COLOR_BGR2GRAY)
        gray_pixels = gray[mask > 0]
        
        # Calculate texture metrics
        texture_variance = np.var(gray_pixels)
        texture_entropy = self._calculate_entropy(gray_pixels)
        
        # Shape features
        contour = segment['contour']
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter > 0:
            circularity = 4 * math.pi * area / (perimeter * perimeter)
        else:
            circularity = 0
        
        # Aspect ratio
        rect = cv2.minAreaRect(contour)
        if min(rect[1]) > 0:
            aspect_ratio = max(rect[1]) / min(rect[1])
        else:
            aspect_ratio = 1
        
        return {
            'avg_color': avg_color.tolist(),
            'color_std': color_std.tolist(),
            'avg_hsv': avg_hsv.tolist(),
            'texture_variance': texture_variance,
            'texture_entropy': texture_entropy,
            'area': area,
            'circularity': circularity,
            'aspect_ratio': aspect_ratio
        }
    
    def _calculate_entropy(self, pixels: np.ndarray) -> float:
        """Calculate entropy of pixel values"""
        hist, _ = np.histogram(pixels, bins=256, range=(0, 256))
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))
    
    def _match_ingredient_to_database(self, features: Dict) -> Optional[Dict]:
        """Match segment features to ingredient database"""
        if not features:
            return None
        
        best_match = None
        best_score = 0
        
        for category, ingredients in self.ingredient_visual_database.items():
            for ingredient, properties in ingredients.items():
                score = 0
                
                # Color matching
                if 'avg_color' in features:
                    db_color = np.array(properties['color'])
                    img_color = np.array(features['avg_color'])
                    color_distance = np.linalg.norm(db_color - img_color)
                    color_score = max(0, 1 - color_distance / 441)  # Normalize to 0-1
                    score += color_score * 0.4
                
                # Texture matching
                if 'texture_variance' in features:
                    texture_map = {
                        'leafy': (0.8, 1.2),
                        'smooth': (0.1, 0.5),
                        'crisp': (0.3, 0.8),
                        'fibrous': (0.5, 1.0),
                        'grainy': (0.6, 1.5),
                        'spongy': (0.2, 0.6),
                        'starchy': (0.4, 0.9),
                        'flaky': (0.7, 1.3),
                        'soft': (0.1, 0.4),
                        'firm': (0.3, 0.7),
                        'juicy': (0.2, 0.6),
                        'seedy': (0.9, 1.5),
                        'layered': (0.5, 1.0),
                        'marbled': (0.4, 0.8)
                    }
                    
                    texture_range = texture_map.get(properties['texture'], (0, 2))
                    if texture_range[0] <= features['texture_variance'] <= texture_range[1]:
                        score += 0.3
                
                # Shape matching
                if 'circularity' in features:
                    if properties['texture'] in ['leafy', 'crisp', 'juicy']:
                        # Less circular
                        if features['circularity'] < 0.7:
                            score += 0.2
                    else:
                        # More circular
                        if features['circularity'] > 0.5:
                            score += 0.2
                
                if score > best_score:
                    best_score = score
                    best_match = {
                        'name': ingredient,
                        'category': category,
                        'confidence': score,
                        'density': properties['density']
                    }
        
        return best_match if best_score > 0.3 else None
    
    def _estimate_ingredient_weight(self, area_pixels: int, pixel_to_cm_ratio: float, density: float) -> float:
        """Estimate ingredient weight from visual area"""
        if pixel_to_cm_ratio is None:
            return 0
        
        # Convert pixel area to cm2
        area_cm2 = area_pixels * (pixel_to_cm_ratio ** 2)
        
        # Estimate thickness (assume average thickness of 1cm for most foods)
        thickness_cm = 1.0
        
        # Calculate volume in cm3
        volume_cm3 = area_cm2 * thickness_cm
        
        # Convert to grams using density (g/cm3)
        weight_grams = volume_cm3 * density
        
        return weight_grams
    
    def _estimate_visual_portions(self, image: np.ndarray, plate_info: Dict) -> Dict:
        """Estimate portion sizes visually"""
        if not plate_info['detected']:
            return {'method': 'heuristic', 'portions': []}
        
        # Calculate actual plate size
        pixel_to_cm_ratio = self.pixel_to_cm_ratio
        if pixel_to_cm_ratio is None:
            return {'method': 'heuristic', 'portions': []}
        
        plate_diameter_cm = plate_info['estimated_diameter_pixels'] * pixel_to_cm_ratio
        plate_area_cm2 = math.pi * (plate_diameter_cm / 2) ** 2
        
        # Segment food on plate
        food_segments = self._segment_food_on_plate(image, plate_info)
        
        portions = []
        for segment in food_segments:
            # Calculate actual area
            segment_area_pixels = cv2.countNonZero(segment['mask'])
            segment_area_cm2 = segment_area_pixels * (pixel_to_cm_ratio ** 2)
            
            # Calculate percentage of plate
            plate_coverage = (segment_area_cm2 / plate_area_cm2) * 100
            
            # Estimate weight
            estimated_weight = self._estimate_portion_weight(segment_area_cm2, segment['features'])
            
            portions.append({
                'name': segment['name'],
                'area_cm2': segment_area_cm2,
                'plate_coverage_percentage': plate_coverage,
                'estimated_weight_grams': estimated_weight,
                'portion_size': self._classify_portion_size(plate_coverage),
                'confidence': segment['confidence']
            })
        
        return {
            'method': 'plate_based',
            'plate_diameter_cm': plate_diameter_cm,
            'plate_area_cm2': plate_area_cm2,
            'portions': portions,
            'total_estimated_weight_grams': sum(p['estimated_weight_grams'] for p in portions)
        }
    
    def _segment_food_on_plate(self, image: np.ndarray, plate_info: Dict) -> List[Dict]:
        """Segment individual food items on plate"""
        # Create mask for plate area
        plate_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(plate_mask, [plate_info['contour']], -1, 255, -1)
        
        # Extract plate region
        plate_region = cv2.bitwise_and(image, image, mask=plate_mask)
        
        # Use the same segmentation as ingredient analysis
        segments = self._segment_image_regions(plate_region)
        
        # Filter segments that are on the plate
        food_segments = []
        for segment in segments:
            # Check if segment overlaps with plate
            overlap = cv2.bitwise_and(segment['mask'], plate_mask)
            if np.sum(overlap) > 0:
                # Extract features and match to food types
                features = self._extract_segment_features(segment)
                food_match = self._match_ingredient_to_database(features)
                
                if food_match:
                    food_segments.append({
                        'name': food_match['name'],
                        'mask': segment['mask'],
                        'features': features,
                        'confidence': food_match['confidence'],
                        'density': food_match['density']
                    })
        
        return food_segments
    
    def _estimate_portion_weight(self, area_cm2: float, features: Dict) -> float:
        """Estimate portion weight from area and features"""
        # Base weight calculation (assuming 1cm thickness)
        base_weight = area_cm2 * 1.0  # cm3
        
        # Adjust based on food density
        density = features.get('density', 1.0)
        adjusted_weight = base_weight * density
        
        # Adjust based on texture (fluffy vs dense)
        if 'texture_variance' in features:
            texture_factor = 1.0
            if features['texture_variance'] > 1.0:  # High variance = fluffy
                texture_factor = 0.8
            elif features['texture_variance'] < 0.3:  # Low variance = dense
                texture_factor = 1.2
            
            adjusted_weight *= texture_factor
        
        return adjusted_weight
    
    def _classify_portion_size(self, plate_coverage: float) -> str:
        """Classify portion size based on plate coverage"""
        if plate_coverage < 10:
            return 'very_small'
        elif plate_coverage < 25:
            return 'small'
        elif plate_coverage < 50:
            return 'medium'
        elif plate_coverage < 75:
            return 'large'
        else:
            return 'very_large'
    
    def _estimate_visual_volumes(self, image: np.ndarray, drink_info: Dict) -> Dict:
        """Estimate drink volumes visually"""
        if not drink_info['detected']:
            return {'method': 'heuristic', 'volume_ml': 0}
        
        pixel_to_cm_ratio = self.pixel_to_cm_ratio
        if pixel_to_cm_ratio is None:
            return {'method': 'heuristic', 'volume_ml': 0}
        
        # Calculate actual dimensions
        rect = drink_info['rect']
        width_pixels = max(rect[1])
        height_pixels = min(rect[1])
        
        width_cm = width_pixels * pixel_to_cm_ratio
        height_cm = height_pixels * pixel_to_cm_ratio
        
        # Estimate volume based on glass shape
        aspect_ratio = drink_info['aspect_ratio']
        
        if aspect_ratio > 2.0:
            # Tall glass - cylinder approximation
            radius_cm = width_cm / 2
            volume_ml = math.pi * (radius_cm ** 2) * height_cm
        else:
            # Short glass - cup approximation
            radius_cm = width_cm / 2
            volume_ml = math.pi * (radius_cm ** 2) * height_cm * 0.8
        
        # Adjust for fill level (assume 80% full)
        volume_ml *= 0.8
        
        return {
            'method': 'glass_based',
            'width_cm': width_cm,
            'height_cm': height_cm,
            'aspect_ratio': aspect_ratio,
            'estimated_volume_ml': round(volume_ml, 0),
            'glass_type': self._classify_glass_type(aspect_ratio, width_cm, height_cm),
            'confidence': 0.8
        }
    
    def _classify_glass_type(self, aspect_ratio: float, width_cm: float, height_cm: float) -> str:
        """Classify glass type from dimensions"""
        if aspect_ratio > 2.5 and height_cm > 15:
            return 'wine_glass'
        elif aspect_ratio > 2.0 and width_cm < 7:
            return 'water_glass'
        elif aspect_ratio < 1.5 and height_cm < 12:
            return 'coffee_mug'
        elif 1.5 < aspect_ratio < 2.0 and 6 < width_cm < 8:
            return 'standard_glass'
        else:
            return 'unknown'
    
    def _recognize_nutrition_facts(self, image: np.ndarray) -> Dict:
        """Recognize and parse nutrition facts from image"""
        # This would use OCR in a real implementation
        # For now, provide visual estimation based on common label patterns
        
        # Look for nutrition facts panel (typically white background with black text)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find rectangular regions with high contrast
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        nutrition_candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Minimum size
                rect = cv2.minAreaRect(contour)
                width, height = rect[1]
                
                # Nutrition panels are typically wider than tall
                if width > height * 1.5:
                    # Check for high contrast (white background)
                    mask = np.zeros(gray.shape, dtype=np.uint8)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(mask, [box], 0, 255, -1)
                    
                    region = gray[mask > 0]
                    if len(region) > 0:
                        brightness = np.mean(region)
                        if brightness > 200:  # Bright background
                            nutrition_candidates.append({
                                'rect': rect,
                                'area': area,
                                'brightness': brightness,
                                'box': box
                            })
        
        if nutrition_candidates:
            best_candidate = max(nutrition_candidates, key=lambda x: x['area'])
            
            # Estimate nutrition values based on panel size and position
            panel_width = max(best_candidate['rect'][1])
            panel_height = min(best_candidate['rect'][1])
            
            # Heuristic estimation based on panel characteristics
            estimated_nutrition = self._estimate_nutrition_from_panel(panel_width, panel_height)
            
            return {
                'detected': True,
                'panel_info': {
                    'width_pixels': panel_width,
                    'height_pixels': panel_height,
                    'area_pixels': best_candidate['area']
                },
                'estimated_nutrition': estimated_nutrition,
                'confidence': 0.6,
                'method': 'visual_estimation'
            }
        
        return {
            'detected': False,
            'method': 'visual_estimation',
            'confidence': 0.0
        }
    
    def _estimate_nutrition_from_panel(self, width_pixels: int, height_pixels: int) -> Dict:
        """Estimate nutrition values from panel characteristics"""
        # Heuristic estimation based on panel size
        panel_area = width_pixels * height_pixels
        
        # Larger panels typically indicate more detailed nutrition information
        if panel_area > 50000:
            # Detailed nutrition panel
            return {
                'calories': 250,
                'fat': 12,
                'carbohydrates': 31,
                'protein': 8,
                'fiber': 3,
                'sugar': 15,
                'sodium': 600,
                'cholesterol': 30
            }
        elif panel_area > 20000:
            # Standard nutrition panel
            return {
                'calories': 180,
                'fat': 8,
                'carbohydrates': 24,
                'protein': 6,
                'fiber': 2,
                'sugar': 12,
                'sodium': 400,
                'cholesterol': 20
            }
        else:
            # Basic nutrition panel
            return {
                'calories': 120,
                'fat': 4,
                'carbohydrates': 18,
                'protein': 4,
                'fiber': 1,
                'sugar': 8,
                'sodium': 200,
                'cholesterol': 10
            }
    
    def _combine_visual_analyses(self, ingredient_analysis: Dict, portion_analysis: Dict, 
                                volume_analysis: Dict, nutrition_facts_analysis: Dict) -> Dict:
        """Combine all visual analyses into comprehensive results"""
        combined = {
            'total_estimated_weight_grams': 0,
            'total_estimated_volume_ml': 0,
            'detected_ingredients': [],
            'nutrition_summary': {},
            'confidence_scores': []
        }
        
        # Combine ingredient and portion analysis
        if ingredient_analysis['detected_ingredients']:
            combined['detected_ingredients'] = ingredient_analysis['detected_ingredients']
            combined['total_estimated_weight_grams'] = sum(
                ing['estimated_weight_grams'] for ing in ingredient_analysis['detected_ingredients']
            )
            combined['confidence_scores'].append(ingredient_analysis.get('confidence', 0.7))
        
        # Add portion analysis if available
        if portion_analysis.get('method') == 'plate_based':
            combined['plate_based_weight'] = portion_analysis['total_estimated_weight_grams']
            combined['confidence_scores'].append(0.8)
        
        # Add volume analysis
        if volume_analysis.get('detected'):
            combined['total_estimated_volume_ml'] = volume_analysis['estimated_volume_ml']
            combined['confidence_scores'].append(volume_analysis['confidence'])
        
        # Add nutrition facts analysis
        if nutrition_facts_analysis.get('detected'):
            combined['nutrition_summary'] = nutrition_facts_analysis['estimated_nutrition']
            combined['confidence_scores'].append(nutrition_facts_analysis['confidence'])
        
        # Calculate overall confidence
        if combined['confidence_scores']:
            combined['overall_confidence'] = np.mean(combined['confidence_scores'])
        else:
            combined['overall_confidence'] = 0.5
        
        return combined
    
    def _calculate_overall_confidence(self, combined_analysis: Dict) -> float:
        """Calculate overall confidence in visual analysis"""
        return combined_analysis.get('overall_confidence', 0.5)
    
    def _get_contour_center(self, contour) -> Tuple[int, int]:
        """Get center point of contour"""
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
        return (0, 0)
    
    def _estimate_diameter_from_area(self, area: float) -> float:
        """Estimate diameter from circular area"""
        return 2 * math.sqrt(area / math.pi)
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input"""
        # Resize to standard input size
        resized = cv2.resize(image, (224, 224))
        
        # Normalize
        normalized = resized.astype(np.float32) / 255.0
        
        # Convert to tensor
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
        
        return tensor.to(self.device)
