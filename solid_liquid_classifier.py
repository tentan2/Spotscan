"""
Solid-to-Liquid Scale Classifier Module
Classifies foods on a 1-5 scale from rock hard to fully liquid
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
import math

class SolidLiquidClassifier:
    """Classifies foods on solid-to-liquid scale (1-5)"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize solid-liquid classifier
        
        Args:
            model_path: Path to pre-trained model
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.scaler = StandardScaler()
        
        # Solid-to-liquid scale definitions
        self.solid_liquid_scale = {
            1: {
                'name': 'Rock Hard',
                'description': 'Extremely solid, requires significant force to deform',
                'characteristics': ['rigid', 'brittle', 'hard', 'unyielding'],
                'examples': ['hard candy', 'nuts', 'seeds', 'hard cheese', 'dry crackers'],
                'indicators': ['no_deformation', 'sharp_edges', 'brittle_texture', 'hard_surface']
            },
            2: {
                'name': 'Firm',
                'description': 'Solid but with some give, moderate resistance',
                'characteristics': ['firm', 'resistant', 'dense', 'structured'],
                'examples': ['carrots', 'apples', 'firm cheese', 'bread crust', 'raw vegetables'],
                'indicators': ['slight_deformation', 'firm_texture', 'structured_surface', 'resistant']
            },
            3: {
                'name': 'Middle Ground',
                'description': 'Balanced solid and liquid properties, chewy or dough-like',
                'characteristics': ['chewy', 'malleable', 'flexible', 'viscous'],
                'examples': ['gummy candy', 'dough', 'chewy candy', 'cheese', 'caramel'],
                'indicators': ['moderate_deformation', 'chewy_texture', 'flexible_surface', 'viscous']
            },
            4: {
                'name': 'Soft',
                'description': 'Primarily liquid with some solid structure',
                'characteristics': ['soft', 'yielding', 'moist', 'pliable'],
                'examples': ['pudding', 'soft cheese', 'ice cream', 'jam', 'mashed potatoes'],
                'indicators': ['easy_deformation', 'soft_texture', 'moist_surface', 'yielding']
            },
            5: {
                'name': 'Fully Liquid',
                'description': 'Completely liquid, no solid structure',
                'characteristics': ['liquid', 'fluid', 'free_flowing', 'no_structure'],
                'examples': ['water', 'juice', 'milk', 'soup', 'oil', 'syrup'],
                'indicators': ['no_structure', 'fluid_motion', 'free_flowing', 'liquid_surface']
            }
        }
        
        # Visual indicators for each scale level
        self.visual_indicators = {
            1: {
                'edge_sharpness': (0.8, 1.0),
                'surface_texture': (0.7, 1.0),
                'deformation_resistance': (0.8, 1.0),
                'light_reflection': (0.3, 0.7)
            },
            2: {
                'edge_sharpness': (0.5, 0.8),
                'surface_texture': (0.5, 0.8),
                'deformation_resistance': (0.5, 0.8),
                'light_reflection': (0.4, 0.8)
            },
            3: {
                'edge_sharpness': (0.2, 0.6),
                'surface_texture': (0.3, 0.7),
                'deformation_resistance': (0.2, 0.6),
                'light_reflection': (0.5, 0.9)
            },
            4: {
                'edge_sharpness': (0.0, 0.4),
                'surface_texture': (0.1, 0.5),
                'deformation_resistance': (0.0, 0.4),
                'light_reflection': (0.6, 1.0)
            },
            5: {
                'edge_sharpness': (0.0, 0.2),
                'surface_texture': (0.0, 0.3),
                'deformation_resistance': (0.0, 0.2),
                'light_reflection': (0.7, 1.0)
            }
        }
        
        # Physical property ranges
        self.physical_properties = {
            'hardness': {1: (8, 10), 2: (5, 8), 3: (2, 5), 4: (0, 2), 5: (0, 1)},
            'elasticity': {1: (0, 1), 2: (1, 3), 3: (3, 6), 4: (6, 8), 5: (9, 10)},
            'viscosity': {1: (0, 1), 2: (1, 5), 3: (5, 50), 4: (50, 500), 5: (500, 10000)},
            'moisture': {1: (0, 10), 2: (10, 30), 3: (30, 60), 4: (60, 85), 5: (85, 100)}
        }
    
    def _load_model(self, model_path: Optional[str]) -> nn.Module:
        """Load or create solid-liquid classification model"""
        if model_path and Path(model_path).exists():
            model = torch.load(model_path, map_location=self.device)
            return model
        
        # Create CNN for solid-liquid classification
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
            nn.Linear(128, 5),  # 5 solid-liquid levels
            nn.Softmax(dim=1)
        )
        
        return model.to(self.device)
    
    def classify_solid_liquid_scale(self, image: np.ndarray, food_type: Optional[str] = None) -> Dict:
        """
        Classify food on solid-to-liquid scale
        
        Args:
            image: Food image
            food_type: Type of food (optional)
            
        Returns:
            Solid-liquid classification
        """
        # Analyze visual indicators
        visual_analysis = self._analyze_visual_indicators(image)
        
        # Analyze texture properties
        texture_analysis = self._analyze_texture_properties(image)
        
        # Analyze shape and deformation
        shape_analysis = self._analyze_shape_deformation(image)
        
        # Analyze surface properties
        surface_analysis = self._analyze_surface_properties(image)
        
        # Analyze light interaction
        light_analysis = self._analyze_light_interaction(image)
        
        # Predict using ML model
        ml_prediction = self._predict_solid_liquid_ml(image)
        
        # Combine all analyses
        combined_scale = self._combine_solid_liquid_analyses(
            visual_analysis, texture_analysis, shape_analysis,
            surface_analysis, light_analysis, ml_prediction
        )
        
        # Generate detailed report
        report = self._generate_solid_liquid_report(combined_scale, food_type)
        
        return {
            'food_type': food_type,
            'solid_liquid_scale': combined_scale,
            'scale_details': self.solid_liquid_scale[combined_scale],
            'visual_analysis': visual_analysis,
            'texture_analysis': texture_analysis,
            'shape_analysis': shape_analysis,
            'surface_analysis': surface_analysis,
            'light_analysis': light_analysis,
            'ml_prediction': ml_prediction,
            'confidence_score': self._calculate_confidence(
                visual_analysis, texture_analysis, shape_analysis, ml_prediction
            ),
            'report': report
        }
    
    def _analyze_visual_indicators(self, image: np.ndarray) -> Dict:
        """Analyze visual indicators for solid-liquid scale"""
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge sharpness (solids have sharper edges)
        edge_sharpness = self._calculate_edge_sharpness(gray)
        
        # Surface texture (solids have more texture)
        surface_texture = self._calculate_surface_texture(gray)
        
        # Deformation resistance (solids resist deformation)
        deformation_resistance = self._estimate_deformation_resistance(gray)
        
        # Light reflection (liquids reflect more evenly)
        light_reflection = self._calculate_light_reflection(hsv)
        
        # Score each scale level based on indicators
        level_scores = {}
        
        for level in range(1, 6):
            indicators = self.visual_indicators[level]
            score = 0
            
            # Edge sharpness
            edge_min, edge_max = indicators['edge_sharpness']
            if edge_min <= edge_sharpness <= edge_max:
                score += 0.25
            else:
                # Penalize if outside range
                distance = min(abs(edge_sharpness - edge_min), abs(edge_sharpness - edge_max))
                score -= distance * 0.1
            
            # Surface texture
            texture_min, texture_max = indicators['surface_texture']
            if texture_min <= surface_texture <= texture_max:
                score += 0.25
            else:
                distance = min(abs(surface_texture - texture_min), abs(surface_texture - texture_max))
                score -= distance * 0.1
            
            # Deformation resistance
            deform_min, deform_max = indicators['deformation_resistance']
            if deform_min <= deformation_resistance <= deform_max:
                score += 0.25
            else:
                distance = min(abs(deformation_resistance - deform_min), abs(deformation_resistance - deform_max))
                score -= distance * 0.1
            
            # Light reflection
            light_min, light_max = indicators['light_reflection']
            if light_min <= light_reflection <= light_max:
                score += 0.25
            else:
                distance = min(abs(light_reflection - light_min), abs(light_reflection - light_max))
                score -= distance * 0.1
            
            level_scores[level] = max(0, min(1, score))
        
        return {
            'level_scores': level_scores,
            'predicted_level': max(level_scores, key=level_scores.get),
            'edge_sharpness': edge_sharpness,
            'surface_texture': surface_texture,
            'deformation_resistance': deformation_resistance,
            'light_reflection': light_reflection
        }
    
    def _calculate_edge_sharpness(self, gray: np.ndarray) -> float:
        """Calculate edge sharpness"""
        # Use Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate edge density and strength
        edge_density = np.sum(edges > 0) / edges.size
        
        # Use Sobel for edge strength
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        edge_strength = np.mean(gradient_magnitude) / 255.0
        
        # Combine density and strength
        sharpness = (edge_density + edge_strength) / 2.0
        
        return sharpness
    
    def _calculate_surface_texture(self, gray: np.ndarray) -> float:
        """Calculate surface texture complexity"""
        # Use Local Binary Pattern
        lbp = self._calculate_lbp(gray, radius=1, n_points=8)
        lbp_hist = np.histogram(lbp, bins=256)[0]
        lbp_hist = lbp_hist / lbp_hist.sum()
        
        # Calculate texture complexity from LBP histogram
        texture_complexity = 1.0 - np.sum(lbp_hist[lbp_hist > 0.01])  # Uniform patterns
        
        # Also use variance
        local_var = cv2.filter2D(gray.astype(np.float32), -1, 
                               np.ones((5, 5), np.float32) / 25)
        variance_score = np.std(local_var) / 50.0  # Normalize
        
        return (texture_complexity + variance_score) / 2.0
    
    def _calculate_lbp(self, image: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
        """Calculate Local Binary Pattern"""
        height, width = image.shape
        lbp = np.zeros((height, width), dtype=np.uint8)
        
        for i in range(radius, height - radius):
            for j in range(radius, width - radius):
                center = image[i, j]
                binary_string = []
                
                for angle in np.linspace(0, 2 * np.pi, n_points, endpoint=False):
                    x = i + radius * np.cos(angle)
                    y = j + radius * np.sin(angle)
                    
                    x1, y1 = int(x), int(y)
                    x2, y2 = min(x1 + 1, height - 1), min(y1 + 1, width - 1)
                    
                    dx, dy = x - x1, y - y1
                    
                    pixel_value = (1 - dx) * (1 - dy) * image[x1, y1] + \
                                dx * (1 - dy) * image[x2, y1] + \
                                (1 - dx) * dy * image[x1, y2] + \
                                dx * dy * image[x2, y2]
                    
                    binary_string.append(1 if pixel_value >= center else 0)
                
                lbp_value = 0
                for bit in binary_string:
                    lbp_value = (lbp_value << 1) | bit
                
                lbp[i, j] = lbp_value
        
        return lbp
    
    def _estimate_deformation_resistance(self, gray: np.ndarray) -> float:
        """Estimate resistance to deformation"""
        # This is a heuristic based on image properties
        # Solids typically have more defined shapes and resist deformation
        
        # Find contours
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.5  # Default
        
        # Analyze main contour
        main_contour = max(contours, key=cv2.contourArea)
        
        # Calculate solidity (how solid the shape is)
        area = cv2.contourArea(main_contour)
        hull = cv2.convexHull(main_contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Calculate aspect ratio stability
        x, y, w, h = cv2.boundingRect(main_contour)
        aspect_ratio = w / h if h > 0 else 1
        
        # Calculate circularity
        perimeter = cv2.arcLength(main_contour, True)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        # Combine metrics
        resistance = (solidity * 0.4 + (1.0 - abs(1.0 - aspect_ratio)) * 0.3 + circularity * 0.3)
        
        return resistance
    
    def _calculate_light_reflection(self, hsv: np.ndarray) -> float:
        """Calculate light reflection properties"""
        # Liquids tend to have more uniform light reflection
        
        value_channel = hsv[:, :, 2]
        
        # Calculate reflection uniformity
        local_var = cv2.filter2D(value_channel.astype(np.float32), -1, 
                               np.ones((7, 7), np.float32) / 49)
        reflection_uniformity = 1.0 / (1.0 + np.std(local_var))
        
        # Calculate highlight intensity
        highlight_mask = value_channel > np.percentile(value_channel, 90)
        highlight_intensity = np.mean(value_channel[highlight_mask]) / 255.0 if np.any(highlight_mask) else 0
        
        # Combine metrics
        reflection = (reflection_uniformity * 0.6 + highlight_intensity * 0.4)
        
        return reflection
    
    def _analyze_texture_properties(self, image: np.ndarray) -> Dict:
        """Analyze texture properties for solid-liquid classification"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Hardness indicator (based on edge density and texture)
        hardness = self._estimate_hardness(gray)
        
        # Elasticity indicator (based on shape flexibility)
        elasticity = self._estimate_elasticity(gray)
        
        # Viscosity indicator (based on flow patterns)
        viscosity = self._estimate_viscosity(gray)
        
        # Moisture content indicator
        moisture = self._estimate_moisture_content(image)
        
        return {
            'hardness': hardness,
            'elasticity': elasticity,
            'viscosity': viscosity,
            'moisture': moisture
        }
    
    def _estimate_hardness(self, gray: np.ndarray) -> float:
        """Estimate hardness from image"""
        # Hard materials have sharp edges and high contrast
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Contrast
        contrast = np.std(gray) / 255.0
        
        # Combine metrics
        hardness = (edge_density * 0.6 + contrast * 0.4)
        
        return hardness
    
    def _estimate_elasticity(self, gray: np.ndarray) -> Dict:
        """Estimate elasticity (would be better with video)"""
        # This is a heuristic based on static image properties
        # Elastic materials often have smooth, curved surfaces
        
        # Find contours
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {'elasticity_score': 0.5, 'flexibility': 0.5}
        
        # Analyze contour smoothness
        main_contour = max(contours, key=cv2.contourArea)
        
        # Calculate contour smoothness (approximation)
        perimeter = cv2.arcLength(main_contour, True)
        area = cv2.contourArea(main_contour)
        
        # Smooth contours have higher area/perimeter ratio
        smoothness = area / perimeter if perimeter > 0 else 0
        
        # Normalize
        elasticity_score = min(smoothness * 10, 1.0)
        flexibility = elasticity_score  # Simplified assumption
        
        return {
            'elasticity_score': elasticity_score,
            'flexibility': flexibility
        }
    
    def _estimate_viscosity(self, gray: np.ndarray) -> float:
        """Estimate viscosity (would be better with video)"""
        # This is a heuristic based on static image properties
        # High viscosity materials have smooth, uniform surfaces
        
        # Surface uniformity
        local_var = cv2.filter2D(gray.astype(np.float32), -1, 
                               np.ones((7, 7), np.float32) / 49)
        uniformity = 1.0 / (1.0 + np.std(local_var))
        
        # Edge complexity (viscous materials have fewer edges)
        edges = cv2.Canny(gray, 30, 100)
        edge_complexity = np.sum(edges > 0) / edges.size
        
        # Combine metrics (inverse edge complexity + uniformity)
        viscosity = (uniformity * 0.6 + (1.0 - edge_complexity) * 0.4)
        
        return viscosity
    
    def _estimate_moisture_content(self, image: np.ndarray) -> float:
        """Estimate moisture content from image"""
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Moisture affects saturation and value
        saturation = np.mean(hsv[:, :, 1]) / 255.0
        value = np.mean(hsv[:, :, 2]) / 255.0
        
        # High moisture often correlates with lower saturation and higher value
        moisture_indicator = (1.0 - saturation * 0.6 + value * 0.4)
        
        return moisture_indicator
    
    def _analyze_shape_deformation(self, image: np.ndarray) -> Dict:
        """Analyze shape and deformation characteristics"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find main object
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {'deformation_score': 0.5, 'shape_stability': 0.5}
        
        main_contour = max(contours, key=cv2.contourArea)
        
        # Analyze shape stability
        area = cv2.contourArea(main_contour)
        perimeter = cv2.arcLength(main_contour, True)
        
        # Calculate shape metrics
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
            compactness = area / (perimeter ** 2) if perimeter > 0 else 0
        else:
            circularity = 0
            compactness = 0
        
        # Deformation resistance (inverse of perceived softness)
        deformation_score = (circularity + compactness) / 2.0
        shape_stability = deformation_score
        
        return {
            'deformation_score': deformation_score,
            'shape_stability': shape_stability,
            'circularity': circularity,
            'compactness': compactness
        }
    
    def _analyze_surface_properties(self, image: np.ndarray) -> Dict:
        """Analyze surface properties"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Surface roughness
        roughness = self._calculate_surface_roughness(gray)
        
        # Surface sheen/gloss
        sheen = self._calculate_surface_sheen(hsv)
        
        # Surface wetness
        wetness = self._calculate_surface_wetness(hsv)
        
        return {
            'surface_roughness': roughness,
            'surface_sheen': sheen,
            'surface_wetness': wetness
        }
    
    def _calculate_surface_roughness(self, gray: np.ndarray) -> float:
        """Calculate surface roughness"""
        # Use gradient magnitude as roughness indicator
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        return np.mean(gradient_magnitude) / 255.0
    
    def _calculate_surface_sheen(self, hsv: np.ndarray) -> float:
        """Calculate surface sheen/gloss"""
        # Sheen appears as high value areas
        value_channel = hsv[:, :, 2]
        
        # Look for bright specular highlights
        bright_mask = value_channel > np.percentile(value_channel, 95)
        sheen_ratio = np.sum(bright_mask) / bright_mask.size
        
        return min(sheen_ratio * 5, 1.0)  # Scale to 0-1
    
    def _calculate_surface_wetness(self, hsv: np.ndarray) -> float:
        """Calculate surface wetness"""
        # Wet surfaces often have lower saturation and higher value
        saturation = np.mean(hsv[:, :, 1]) / 255.0
        value = np.mean(hsv[:, :, 2]) / 255.0
        
        wetness = (1.0 - saturation * 0.7 + value * 0.3)
        
        return wetness
    
    def _analyze_light_interaction(self, image: np.ndarray) -> Dict:
        """Analyze how light interacts with the surface"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Light absorption
        absorption = self._calculate_light_absorption(hsv)
        
        # Light scattering
        scattering = self._calculate_light_scattering(gray)
        
        # Light transmission
        transmission = self._calculate_light_transmission(hsv)
        
        return {
            'light_absorption': absorption,
            'light_scattering': scattering,
            'light_transmission': transmission
        }
    
    def _calculate_light_absorption(self, hsv: np.ndarray) -> float:
        """Calculate light absorption"""
        # Darker colors absorb more light
        value_channel = hsv[:, :, 2]
        absorption = 1.0 - np.mean(value_channel) / 255.0
        
        return absorption
    
    def _calculate_light_scattering(self, gray: np.ndarray) -> float:
        """Calculate light scattering"""
        # Scattered light creates more texture variation
        local_var = cv2.filter2D(gray.astype(np.float32), -1, 
                               np.ones((5, 5), np.float32) / 25)
        scattering = np.std(local_var) / 50.0  # Normalize
        
        return min(scattering, 1.0)
    
    def _calculate_light_transmission(self, hsv: np.ndarray) -> float:
        """Calculate light transmission"""
        # Higher value indicates more transmission
        value_channel = hsv[:, :, 2]
        transmission = np.mean(value_channel) / 255.0
        
        return transmission
    
    def _predict_solid_liquid_ml(self, image: np.ndarray) -> Dict:
        """Predict solid-liquid scale using ML model"""
        # Preprocess image
        input_tensor = self._preprocess_for_solid_liquid(image)
        
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
    
    def _preprocess_for_solid_liquid(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for solid-liquid classification"""
        # Resize to model input size
        resized = cv2.resize(image, (224, 224))
        
        # Convert to tensor
        tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float()
        tensor = tensor / 255.0  # Normalize to [0, 1]
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)
    
    def _combine_solid_liquid_analyses(self, visual: Dict, texture: Dict, 
                                       shape: Dict, surface: Dict, 
                                       light: Dict, ml: Dict) -> int:
        """Combine all analyses to determine final scale"""
        # Weight factors for different analyses
        weights = {
            'visual': 0.25,
            'texture': 0.25,
            'shape': 0.15,
            'surface': 0.15,
            'light': 0.1,
            'ml': 0.1
        }
        
        # Combine level scores
        combined_scores = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        
        # Visual analysis
        if 'level_scores' in visual:
            for level, score in visual['level_scores'].items():
                combined_scores[level] += score * weights['visual']
        
        # Texture analysis
        texture_score = self._texture_to_scale_score(texture)
        for level, score in texture_score.items():
            combined_scores[level] += score * weights['texture']
        
        # Shape analysis
        if 'deformation_score' in shape:
            deformation = shape['deformation_score']
            shape_scores = self._deformation_to_scale_scores(deformation)
            for level, score in shape_scores.items():
                combined_scores[level] += score * weights['shape']
        
        # Surface analysis
        surface_score = self._surface_to_scale_score(surface)
        for level, score in surface_score.items():
            combined_scores[level] += score * weights['surface']
        
        # Light analysis
        light_score = self._light_to_scale_score(light)
        for level, score in light_score.items():
            combined_scores[level] += score * weights['light']
        
        # ML prediction
        if 'level_scores' in ml:
            for level, score in ml['level_scores'].items():
                combined_scores[level] += score * weights['ml']
        
        # Return level with highest score
        return max(combined_scores, key=combined_scores.get)
    
    def _texture_to_scale_score(self, texture: Dict) -> Dict:
        """Convert texture analysis to scale scores"""
        scores = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        
        hardness = texture.get('hardness', 0.5)
        elasticity = texture.get('elasticity', {}).get('elasticity_score', 0.5)
        viscosity = texture.get('viscosity', 0.5)
        moisture = texture.get('moisture', 0.5)
        
        # Hard solids: high hardness, low elasticity, low viscosity, low moisture
        if hardness > 0.7 and elasticity < 0.3 and viscosity < 0.3 and moisture < 0.3:
            scores[1] = 0.8
        # Firm solids: moderate hardness and elasticity
        elif 0.4 < hardness <= 0.7 and 0.3 <= elasticity < 0.6:
            scores[2] = 0.8
        # Middle ground: balanced properties
        elif 0.2 < hardness <= 0.4 and 0.6 <= elasticity < 0.8:
            scores[3] = 0.8
        # Soft: low hardness, high elasticity or viscosity
        elif hardness <= 0.2 and (elasticity > 0.7 or viscosity > 0.6):
            scores[4] = 0.8
        # Liquid: very low hardness, high viscosity or moisture
        elif hardness <= 0.1 and (viscosity > 0.8 or moisture > 0.8):
            scores[5] = 0.8
        
        return scores
    
    def _deformation_to_scale_scores(self, deformation: float) -> Dict:
        """Convert deformation score to scale scores"""
        scores = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        
        if deformation > 0.8:
            scores[1] = 0.9  # Rock hard
        elif deformation > 0.6:
            scores[2] = 0.9  # Firm
        elif deformation > 0.4:
            scores[3] = 0.9  # Middle ground
        elif deformation > 0.2:
            scores[4] = 0.9  # Soft
        else:
            scores[5] = 0.9  # Liquid
        
        return scores
    
    def _surface_to_scale_score(self, surface: Dict) -> Dict:
        """Convert surface analysis to scale scores"""
        scores = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        
        roughness = surface.get('surface_roughness', 0.5)
        sheen = surface.get('surface_sheen', 0.5)
        wetness = surface.get('surface_wetness', 0.5)
        
        # Rough surfaces indicate solids
        if roughness > 0.6 and sheen < 0.4 and wetness < 0.4:
            scores[1] = 0.7
        # Moderately rough
        elif 0.3 < roughness <= 0.6:
            scores[2] = 0.7
        # Smooth but not wet
        elif roughness <= 0.3 and wetness < 0.5:
            scores[3] = 0.7
        # Soft and somewhat wet
        elif roughness <= 0.2 and 0.5 <= wetness < 0.8:
            scores[4] = 0.7
        # Very smooth and wet
        elif roughness <= 0.1 and wetness >= 0.8:
            scores[5] = 0.7
        
        return scores
    
    def _light_to_scale_score(self, light: Dict) -> Dict:
        """Convert light analysis to scale scores"""
        scores = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        
        absorption = light.get('light_absorption', 0.5)
        transmission = light.get('light_transmission', 0.5)
        
        # High absorption (solids)
        if absorption > 0.6 and transmission < 0.4:
            scores[1] = 0.6
        # Medium absorption
        elif 0.3 < absorption <= 0.6:
            scores[2] = 0.6
        # Balanced
        elif 0.2 < absorption <= 0.3:
            scores[3] = 0.6
        # High transmission
        elif absorption <= 0.2 and transmission > 0.7:
            scores[5] = 0.6
        # Medium transmission
        else:
            scores[4] = 0.6
        
        return scores
    
    def _calculate_confidence(self, visual: Dict, texture: Dict, shape: Dict, ml: Dict) -> float:
        """Calculate overall confidence in classification"""
        confidences = []
        
        if 'predicted_level' in visual:
            max_visual_score = max(visual['level_scores'].values())
            confidences.append(max_visual_score)
        
        if 'hardness' in texture:
            # Use texture properties for confidence
            texture_conf = 0.7  # Default confidence for texture analysis
            confidences.append(texture_conf)
        
        if 'deformation_score' in shape:
            # Use shape stability for confidence
            shape_conf = abs(shape['deformation_score'] - 0.5) * 2  # Convert to 0-1
            confidences.append(shape_conf)
        
        if 'predicted_level' in ml:
            max_ml_score = max(ml['level_scores'].values())
            confidences.append(max_ml_score)
        
        if confidences:
            return np.mean(confidences)
        else:
            return 0.5  # Default confidence
    
    def _generate_solid_liquid_report(self, scale: int, food_type: Optional[str]) -> Dict:
        """Generate detailed solid-liquid report"""
        scale_info = self.solid_liquid_scale[scale]
        
        report = {
            'food_type': food_type,
            'solid_liquid_scale': scale,
            'scale_name': scale_info['name'],
            'description': scale_info['description'],
            'characteristics': scale_info['characteristics'],
            'examples': scale_info['examples'],
            'physical_properties': self._get_physical_properties_for_scale(scale),
            'handling_recommendations': self._get_handling_recommendations(scale),
            'storage_recommendations': self._get_storage_recommendations(scale),
            'consumption_suggestions': self._get_consumption_suggestions(scale, food_type)
        }
        
        return report
    
    def _get_physical_properties_for_scale(self, scale: int) -> Dict:
        """Get expected physical properties for scale"""
        props = self.physical_properties
        
        return {
            'hardness_range': props['hardness'][scale],
            'elasticity_range': props['elasticity'][scale],
            'viscosity_range': props['viscosity'][scale],
            'moisture_range': props['moisture'][scale]
        }
    
    def _get_handling_recommendations(self, scale: int) -> List[str]:
        """Get handling recommendations based on scale"""
        recommendations = {
            1: [
                "Handle with care to avoid breakage",
                "Store in dry conditions",
                "Use appropriate tools for cutting"
            ],
            2: [
                "Handle gently to maintain shape",
                "Store in appropriate conditions",
                "Can be cut with standard utensils"
            ],
            3: [
                "Handle with moderate care",
                "May be cut or shaped easily",
                "Store in sealed container if needed"
            ],
            4: [
                "Handle gently to avoid spilling",
                "Store in sealed container",
                "May require utensils for serving"
            ],
            5: [
                "Pour carefully to avoid spills",
                "Store in sealed container",
                "Use appropriate serving vessel"
            ]
        }
        
        return recommendations.get(scale, [])
    
    def _get_storage_recommendations(self, scale: int) -> List[str]:
        """Get storage recommendations based on scale"""
        recommendations = {
            1: [
                "Store in dry, cool place",
                "Keep away from moisture",
                "Use airtight container if possible"
            ],
            2: [
                "Store in cool, dry place",
                "Protect from moisture",
                "Refrigerate if perishable"
            ],
            3: [
                "Store in airtight container",
                "Refrigerate if perishable",
                "Keep away from strong odors"
            ],
            4: [
                "Store in sealed container",
                "Refrigerate immediately",
                "Use within recommended time"
            ],
            5: [
                "Store in sealed, airtight container",
                "Refrigerate if required",
                "Shake before use if separated"
            ]
        }
        
        return recommendations.get(scale, [])
    
    def _get_consumption_suggestions(self, scale: int, food_type: Optional[str]) -> List[str]:
        """Get consumption suggestions based on scale and food type"""
        suggestions = {
            1: [
                "May require preparation before consumption",
                "Consider cutting into smaller pieces",
                "Pair with softer foods for balance"
            ],
            2: [
                "Can be consumed as is or with preparation",
                "Consider cooking methods for softer texture",
                "Pair with complementary textures"
            ],
            3: [
                "Ready to consume as is",
                "Can be enjoyed at room temperature",
                "Pairs well with various accompaniments"
            ],
            4: [
                "Best consumed immediately",
                "Consider temperature for optimal enjoyment",
                "May require utensils for serving"
            ],
            5: [
                "Consume at appropriate temperature",
                "Stir before serving if separated",
                "Serve in appropriate vessel"
            ]
        }
        
        base_suggestions = suggestions.get(scale, [])
        
        # Add food-type specific suggestions
        if food_type:
            food_type_lower = food_type.lower()
            
            if scale == 1 and 'candy' in food_type_lower:
                base_suggestions.append("Suck slowly to enjoy longer")
            elif scale == 5 and 'soup' in food_type_lower:
                base_suggestions.append("Serve hot for best flavor")
            elif scale == 3 and 'cheese' in food_type_lower:
                base_suggestions.append("Allow to reach room temperature for best flavor")
        
        return base_suggestions
