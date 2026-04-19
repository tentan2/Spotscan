"""
Portion Analyzer Module
Estimates size, volume, and portion sizes from food images
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
import math

class PortionAnalyzer:
    """Analyzes food portions to estimate size, volume, and weight"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize portion analyzer
        
        Args:
            model_path: Path to pre-trained model
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.size_estimator = self._load_size_model(model_path)
        self.volume_estimator = self._load_volume_model(model_path)
        self.scaler = StandardScaler()
        
        # Density values for common foods (g/cm3)
        self.food_densities = {
            'fruits': {
                'apple': 0.6, 'banana': 0.95, 'orange': 0.48, 'strawberry': 0.6,
                'grape': 0.98, 'watermelon': 0.93, 'lemon': 0.58, 'peach': 0.58
            },
            'vegetables': {
                'carrot': 0.64, 'broccoli': 0.34, 'tomato': 0.95, 'potato': 1.09,
                'onion': 0.91, 'lettuce': 0.24, 'cucumber': 0.96, 'pepper': 0.34
            },
            'grains': {
                'rice': 1.45, 'bread': 0.25, 'pasta': 1.5, 'oatmeal': 0.78,
                'cereal': 0.35, 'quinoa': 1.4
            },
            'proteins': {
                'chicken': 1.03, 'beef': 1.06, 'pork': 1.09, 'fish': 1.05,
                'egg': 1.03, 'cheese': 1.13, 'yogurt': 1.04, 'tofu': 0.55
            },
            'liquids': {
                'water': 1.0, 'milk': 1.03, 'juice': 1.05, 'oil': 0.92,
                'sauce': 1.2, 'soup': 1.1
            }
        }
        
        # Reference object sizes for calibration (in pixels)
        self.reference_objects = {
            'credit_card': {'width': 85.6, 'height': 53.98},  # mm
            'coin_quarter': {'diameter': 24.26},  # mm
            'coin_penny': {'diameter': 19.05},  # mm
            'paper_dollar': {'width': 155.96, 'height': 66.29},  # mm
            'smartphone': {'average_width': 75, 'average_height': 150}  # mm
        }
        
        # Standard portion sizes (in grams)
        self.standard_portions = {
            'fruits': {'small': 80, 'medium': 150, 'large': 250},
            'vegetables': {'small': 50, 'medium': 100, 'large': 200},
            'grains': {'small': 30, 'medium': 60, 'large': 120},
            'proteins': {'small': 50, 'medium': 100, 'large': 200},
            'liquids': {'small': 100, 'medium': 250, 'large': 500}  # mL
        }
    
    def _load_size_model(self, model_path: Optional[str]) -> nn.Module:
        """Load or create size estimation model"""
        if model_path and Path(model_path).exists():
            model = torch.load(model_path, map_location=self.device)
            return model
        
        # Create CNN for size estimation
        model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3)  # width, height, depth in mm
        )
        
        return model.to(self.device)
    
    def _load_volume_model(self, model_path: Optional[str]) -> nn.Module:
        """Load or create volume estimation model"""
        # Similar structure to size model but outputs volume
        model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)  # volume in cm3
        )
        
        return model.to(self.device)
    
    def analyze_portion(self, image: np.ndarray, food_type: str, 
                       reference_object: Optional[str] = None) -> Dict:
        """
        Analyze food portion to estimate size, volume, and weight
        
        Args:
            image: Food image
            food_type: Type of food
            reference_object: Reference object for calibration (optional)
            
        Returns:
            Comprehensive portion analysis
        """
        # Detect food region
        food_region = self._detect_food_region(image)
        
        # Estimate physical dimensions
        size_analysis = self._estimate_size(image, food_region, reference_object)
        
        # Estimate volume
        volume_analysis = self._estimate_volume(image, food_region, food_type)
        
        # Calculate weight using density
        weight_analysis = self._calculate_weight(volume_analysis, food_type)
        
        # Classify portion size
        portion_classification = self._classify_portion_size(
            weight_analysis['weight'], food_type
        )
        
        # Analyze shape for volume calculation
        shape_analysis = self._analyze_shape_for_volume(food_region)
        
        # Generate nutritional scaling
        nutritional_scaling = self._calculate_nutritional_scaling(
            weight_analysis['weight'], food_type
        )
        
        return {
            'food_type': food_type,
            'size_analysis': size_analysis,
            'volume_analysis': volume_analysis,
            'weight_analysis': weight_analysis,
            'portion_classification': portion_classification,
            'shape_analysis': shape_analysis,
            'nutritional_scaling': nutritional_scaling,
            'reference_object': reference_object,
            'confidence_score': self._calculate_confidence(
                size_analysis, volume_analysis, weight_analysis
            )
        }
    
    def _detect_food_region(self, image: np.ndarray) -> Dict:
        """Detect the main food region in the image"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to separate food from background
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Clean up with morphological operations
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {'error': 'No food region detected'}
        
        # Find the largest contour (main food)
        main_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(main_contour)
        
        # Calculate area
        area = cv2.contourArea(main_contour)
        
        return {
            'contour': main_contour,
            'bounding_box': [x, y, w, h],
            'area': area,
            'center': [x + w//2, y + h//2],
            'mask': thresh
        }
    
    def _estimate_size(self, image: np.ndarray, food_region: Dict, 
                      reference_object: Optional[str]) -> Dict:
        """Estimate physical dimensions of food"""
        # Get pixel dimensions
        bbox = food_region['bounding_box']
        pixel_width = bbox[2]
        pixel_height = bbox[3]
        
        # Calculate scale factor
        if reference_object:
            scale_factor = self._calculate_scale_factor(image, reference_object)
        else:
            # Use default scale (assume average phone camera)
            scale_factor = self._estimate_default_scale(image, food_region)
        
        # Convert to physical dimensions
        physical_width = pixel_width / scale_factor
        physical_height = pixel_height / scale_factor
        
        # Estimate depth using ML model
        depth_estimate = self._estimate_depth_ml(image, food_region)
        
        # Use ML model for better size estimation
        ml_size = self._predict_size_ml(image, food_region)
        
        return {
            'pixel_dimensions': {'width': pixel_width, 'height': pixel_height},
            'physical_dimensions': {
                'width_mm': physical_width,
                'height_mm': physical_height,
                'depth_mm': depth_estimate
            },
            'ml_dimensions': {
                'width_mm': ml_size[0],
                'height_mm': ml_size[1],
                'depth_mm': ml_size[2]
            },
            'scale_factor': scale_factor,
            'reference_object': reference_object
        }
    
    def _calculate_scale_factor(self, image: np.ndarray, reference_object: str) -> float:
        """Calculate scale factor using reference object"""
        # This is a simplified implementation
        # In practice, would detect the reference object in the image
        
        # For now, use assumed reference object size
        if reference_object == 'credit_card':
            # Assume credit card takes up certain portion of image
            image_width = image.shape[1]
            assumed_card_pixels = image_width * 0.3  # Assume card is 30% of image width
            actual_card_width = self.reference_objects['credit_card']['width']
            scale_factor = assumed_card_pixels / actual_card_width
        elif reference_object == 'coin_quarter':
            image_width = image.shape[1]
            assumed_coin_pixels = image_width * 0.05  # Assume coin is 5% of image width
            actual_coin_diameter = self.reference_objects['coin_quarter']['diameter']
            scale_factor = assumed_coin_pixels / actual_coin_diameter
        else:
            # Default scale
            scale_factor = 10.0  # 10 pixels per mm
        
        return scale_factor
    
    def _estimate_default_scale(self, image: np.ndarray, food_region: Dict) -> float:
        """Estimate default scale factor without reference object"""
        # Use heuristics based on image properties
        image_height = image.shape[0]
        food_height_pixels = food_region['bounding_box'][3]
        
        # Assume food takes up certain portion of image
        food_height_ratio = food_height_pixels / image_height
        
        # Estimate actual food size based on typical ranges
        if food_height_ratio > 0.5:
            # Large food item
            estimated_height_mm = 100
        elif food_height_ratio > 0.2:
            # Medium food item
            estimated_height_mm = 50
        else:
            # Small food item
            estimated_height_mm = 25
        
        scale_factor = food_height_pixels / estimated_height_mm
        
        return scale_factor
    
    def _estimate_depth_ml(self, image: np.ndarray, food_region: Dict) -> float:
        """Estimate depth using ML model"""
        # For now, use heuristic based on shape
        bbox = food_region['bounding_box']
        width = bbox[2]
        height = bbox[3]
        
        # Depth is typically smaller than width and height
        # Use aspect ratio as heuristic
        aspect_ratio = width / height if height > 0 else 1
        
        if aspect_ratio > 2:  # Long, thin object
            depth = min(width, height) * 0.3
        elif aspect_ratio < 0.5:  # Tall, thin object
            depth = min(width, height) * 0.5
        else:  # Roughly square/circular
            depth = min(width, height) * 0.7
        
        # Convert to mm using scale factor
        scale_factor = self._estimate_default_scale(image, food_region)
        depth_mm = depth / scale_factor
        
        return depth_mm
    
    def _predict_size_ml(self, image: np.ndarray, food_region: Dict) -> List[float]:
        """Predict size using ML model"""
        # Extract food region
        bbox = food_region['bounding_box']
        x, y, w, h = bbox
        
        # Crop food region
        food_image = image[y:y+h, x:x+w]
        
        if food_image.size == 0:
            return [50.0, 50.0, 25.0]  # Default size
        
        # Resize to model input size
        resized = cv2.resize(food_image, (224, 224))
        
        # Convert to tensor
        tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float()
        tensor = tensor / 255.0
        tensor = tensor.unsqueeze(0)
        
        # Predict
        self.size_estimator.eval()
        with torch.no_grad():
            size_tensor = self.size_estimator(tensor.to(self.device))
            size_pred = size_tensor.cpu().numpy()[0]
        
        return size_pred.tolist()
    
    def _estimate_volume(self, image: np.ndarray, food_region: Dict, food_type: str) -> Dict:
        """Estimate volume of food"""
        # Get dimensions
        bbox = food_region['bounding_box']
        pixel_width = bbox[2]
        pixel_height = bbox[3]
        scale_factor = self._estimate_default_scale(image, food_region)
        
        # Physical dimensions
        width_mm = pixel_width / scale_factor
        height_mm = pixel_height / scale_factor
        depth_mm = self._estimate_depth_ml(image, food_region)
        
        # Convert to cm
        width_cm = width_mm / 10
        height_cm = height_mm / 10
        depth_cm = depth_mm / 10
        
        # Estimate volume based on shape
        shape_volume = self._calculate_shape_volume(width_cm, height_cm, depth_cm, food_type)
        
        # Use ML model for volume estimation
        ml_volume = self._predict_volume_ml(image, food_region)
        
        # Average the estimates
        final_volume = (shape_volume + ml_volume) / 2
        
        return {
            'dimensions_cm': {
                'width': width_cm,
                'height': height_cm,
                'depth': depth_cm
            },
            'shape_volume': shape_volume,
            'ml_volume': ml_volume,
            'estimated_volume': final_volume,
            'volume_unit': 'cm3'
        }
    
    def _calculate_shape_volume(self, width: float, height: float, depth: float, food_type: str) -> float:
        """Calculate volume based on assumed shape"""
        food_type_lower = food_type.lower()
        
        # Determine shape based on food type
        if 'apple' in food_type_lower or 'orange' in food_type_lower:
            # Sphere
            radius = min(width, height) / 2
            volume = (4/3) * math.pi * (radius ** 3)
        elif 'banana' in food_type_lower:
            # Ellipsoid
            volume = (4/3) * math.pi * (width/2) * (height/2) * (depth/2)
        elif 'carrot' in food_type_lower:
            # Cylinder
            radius = min(width, depth) / 2
            volume = math.pi * (radius ** 2) * height
        elif 'bread' in food_type_lower or 'cake' in food_type_lower:
            # Rectangular prism
            volume = width * height * depth
        else:
            # Default to ellipsoid
            volume = (4/3) * math.pi * (width/2) * (height/2) * (depth/2)
        
        return volume
    
    def _predict_volume_ml(self, image: np.ndarray, food_region: Dict) -> float:
        """Predict volume using ML model"""
        # Extract food region
        bbox = food_region['bounding_box']
        x, y, w, h = bbox
        
        # Crop food region
        food_image = image[y:y+h, x:x+w]
        
        if food_image.size == 0:
            return 100.0  # Default volume
        
        # Resize to model input size
        resized = cv2.resize(food_image, (224, 224))
        
        # Convert to tensor
        tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float()
        tensor = tensor / 255.0
        tensor = tensor.unsqueeze(0)
        
        # Predict
        self.volume_estimator.eval()
        with torch.no_grad():
            volume_tensor = self.volume_estimator(tensor.to(self.device))
            volume_pred = volume_tensor.cpu().numpy()[0][0]
        
        return float(volume_pred)
    
    def _calculate_weight(self, volume_analysis: Dict, food_type: str) -> Dict:
        """Calculate weight using volume and density"""
        volume = volume_analysis['estimated_volume']
        
        # Get density for food type
        density = self._get_food_density(food_type)
        
        # Calculate weight
        weight = volume * density
        
        return {
            'weight': weight,
            'weight_unit': 'g',
            'density_used': density,
            'volume_used': volume
        }
    
    def _get_food_density(self, food_type: str) -> float:
        """Get density for specific food type"""
        food_type_lower = food_type.lower()
        
        # Search through categories
        for category, foods in self.food_densities.items():
            for food, density in foods.items():
                if food in food_type_lower:
                    return density
        
        # Default density if not found
        return 1.0  # g/cm3 (water density)
    
    def _classify_portion_size(self, weight: float, food_type: str) -> Dict:
        """Classify portion size"""
        food_type_lower = food_type.lower()
        
        # Determine category
        category = self._get_food_category(food_type_lower)
        
        if category in self.standard_portions:
            portions = self.standard_portions[category]
            
            if weight < portions['small']:
                size = 'very_small'
                percentage = weight / portions['small'] * 100
            elif weight < portions['medium']:
                size = 'small'
                percentage = weight / portions['medium'] * 100
            elif weight < portions['large']:
                size = 'medium'
                percentage = weight / portions['large'] * 100
            else:
                size = 'large'
                percentage = weight / portions['large'] * 100
        else:
            # Default classification
            if weight < 50:
                size = 'small'
                percentage = 50
            elif weight < 150:
                size = 'medium'
                percentage = 100
            else:
                size = 'large'
                percentage = 150
        
        return {
            'size_category': size,
            'weight_grams': weight,
            'percentage_of_standard': percentage,
            'category': category
        }
    
    def _get_food_category(self, food_type: str) -> str:
        """Get food category"""
        if any(fruit in food_type for fruit in ['apple', 'banana', 'orange', 'strawberry', 'grape']):
            return 'fruits'
        elif any(veg in food_type for veg in ['carrot', 'broccoli', 'tomato', 'potato', 'lettuce']):
            return 'vegetables'
        elif any(grain in food_type for grain in ['rice', 'bread', 'pasta', 'oatmeal']):
            return 'grains'
        elif any(protein in food_type for protein in ['chicken', 'beef', 'pork', 'fish', 'egg']):
            return 'proteins'
        elif any(liquid in food_type for liquid in ['water', 'milk', 'juice', 'soup']):
            return 'liquids'
        else:
            return 'other'
    
    def _analyze_shape_for_volume(self, food_region: Dict) -> Dict:
        """Analyze shape characteristics for volume calculation"""
        contour = food_region['contour']
        
        # Calculate shape metrics
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter > 0:
            circularity = 4 * math.pi * area / (perimeter ** 2)
        else:
            circularity = 0
        
        # Get bounding box
        bbox = food_region['bounding_box']
        bbox_area = bbox[2] * bbox[3]
        
        # Extent (how filled the bounding box is)
        extent = area / bbox_area if bbox_area > 0 else 0
        
        # Solidity (area vs convex hull area)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Determine shape type
        if circularity > 0.7:
            shape_type = 'circular'
        elif extent > 0.7:
            shape_type = 'rectangular'
        elif solidity > 0.8:
            shape_type = 'irregular_filled'
        else:
            shape_type = 'irregular'
        
        return {
            'shape_type': shape_type,
            'circularity': circularity,
            'extent': extent,
            'solidity': solidity,
            'area': area,
            'perimeter': perimeter
        }
    
    def _calculate_nutritional_scaling(self, weight: float, food_type: str) -> Dict:
        """Calculate nutritional scaling based on portion size"""
        # Get standard portion weight
        category = self._get_food_category(food_type)
        
        if category in self.standard_portions:
            standard_weight = self.standard_portions[category]['medium']
        else:
            standard_weight = 100  # Default 100g
        
        # Calculate scaling factor
        scaling_factor = weight / standard_weight
        
        return {
            'scaling_factor': scaling_factor,
            'standard_portion_weight': standard_weight,
            'actual_weight': weight,
            'category': category,
            'message': f"This portion is {scaling_factor:.1f}x the standard serving size"
        }
    
    def _calculate_confidence(self, size_analysis: Dict, volume_analysis: Dict, 
                            weight_analysis: Dict) -> float:
        """Calculate overall confidence in measurements"""
        # Base confidence
        confidence = 0.7
        
        # Adjust based on reference object
        if size_analysis.get('reference_object'):
            confidence += 0.2
        
        # Adjust based on consistency between methods
        size_volume = size_analysis['physical_dimensions']['width_mm'] * \
                     size_analysis['physical_dimensions']['height_mm'] * \
                     size_analysis['physical_dimensions']['depth_mm'] / 1000  # Rough volume
        ml_volume = volume_analysis['ml_volume']
        
        if abs(size_volume - ml_volume) / ml_volume < 0.3:  # Within 30%
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def estimate_liquid_volume(self, image: np.ndarray, container_type: str = 'glass') -> Dict:
        """Estimate liquid volume in containers"""
        # Detect container
        container = self._detect_container(image, container_type)
        
        if not container['detected']:
            return {'error': 'Container not detected'}
        
        # Estimate liquid level
        liquid_level = self._detect_liquid_level(image, container)
        
        # Calculate volume based on container shape
        volume = self._calculate_container_volume(container, liquid_level)
        
        return {
            'container': container,
            'liquid_level': liquid_level,
            'estimated_volume': volume,
            'volume_unit': 'mL',
            'container_type': container_type
        }
    
    def _detect_container(self, image: np.ndarray, container_type: str) -> Dict:
        """Detect container in image"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find container-like shapes
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area > 1000:  # Minimum area threshold
                # Check if shape matches container type
                if container_type == 'glass':
                    # Look for circular/oval shapes
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * math.pi * area / (perimeter ** 2)
                        if circularity > 0.6:
                            return {
                                'detected': True,
                                'contour': contour,
                                'area': area,
                                'circularity': circularity
                            }
        
        return {'detected': False}
    
    def _detect_liquid_level(self, image: np.ndarray, container: Dict) -> Dict:
        """Detect liquid level in container"""
        # This is a simplified implementation
        # In practice, would use color and texture analysis
        
        # Get container bounding box
        contour = container['contour']
        x, y, w, h = cv2.boundingRect(contour)
        
        # Assume liquid fills certain percentage
        fill_percentage = 0.7  # Default 70% filled
        
        liquid_height = h * fill_percentage
        
        return {
            'fill_percentage': fill_percentage,
            'liquid_height_pixels': liquid_height,
            'container_height_pixels': h
        }
    
    def _calculate_container_volume(self, container: Dict, liquid_level: Dict) -> float:
        """Calculate volume based on container shape"""
        # Simplified calculation
        # In practice, would use actual container dimensions
        
        container_area = container['area']
        fill_percentage = liquid_level['fill_percentage']
        
        # Estimate volume (simplified)
        volume_ml = container_area * fill_percentage * 0.5  # Rough conversion
        
        return volume_ml
