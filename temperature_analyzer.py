"""
Temperature Analyzer Module
Estimates food temperature using phone sensors and environmental color cues
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

class TemperatureAnalyzer:
    """Analyzes food temperature using visual and sensor data"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize temperature analyzer
        
        Args:
            model_path: Path to pre-trained model
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.scaler = StandardScaler()
        
        # Temperature ranges for different food types (Celsius)
        self.temperature_ranges = {
            'hot_foods': {
                'coffee': {'min': 60, 'max': 85, 'optimal': 70},
                'soup': {'min': 55, 'max': 80, 'optimal': 65},
                'pizza': {'min': 60, 'max': 75, 'optimal': 68},
                'pasta': {'min': 50, 'max': 70, 'optimal': 60},
                'meat': {'min': 55, 'max': 75, 'optimal': 65}
            },
            'cold_foods': {
                'ice_cream': {'min': -15, 'max': -5, 'optimal': -10},
                'salad': {'min': 2, 'max': 8, 'optimal': 4},
                'yogurt': {'min': 1, 'max': 6, 'optimal': 3},
                'juice': {'min': 2, 'max': 8, 'optimal': 4},
                'fruit': {'min': 5, 'max': 15, 'optimal': 10}
            },
            'room_temperature': {
                'bread': {'min': 18, 'max': 25, 'optimal': 22},
                'cheese': {'min': 15, 'max': 20, 'optimal': 18},
                'fruit_ripe': {'min': 18, 'max': 22, 'optimal': 20}
            }
        }
        
        # Temperature indicators based on visual cues
        self.visual_indicators = {
            'steam': {
                'color_range': [(200, 200, 255), (255, 255, 255)],  # White/gray
                'texture_pattern': 'wispy',
                'temperature_range': (50, 100)
            },
            'condensation': {
                'color_range': [(150, 150, 200), (200, 200, 255)],  # Light blue/gray
                'texture_pattern': 'droplets',
                'temperature_range': (5, 25)
            },
            'ice_crystals': {
                'color_range': [(220, 220, 255), (255, 255, 255)],  # White
                'texture_pattern': 'crystalline',
                'temperature_range': (-20, 0)
            },
            'melting': {
                'color_range': [(100, 100, 150), (150, 150, 200)],  # Darker blue/gray
                'texture_pattern': 'liquid_drips',
                'temperature_range': (0, 10)
            }
        }
    
    def _load_model(self, model_path: Optional[str]) -> nn.Module:
        """Load or create temperature estimation model"""
        if model_path and Path(model_path).exists():
            model = torch.load(model_path, map_location=self.device)
            return model
        
        # Create a CNN for temperature estimation
        model = nn.Sequential(
            # Feature extraction layers
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Regression layers
            nn.Flatten(),
            nn.Linear(256 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)  # Single output: temperature
        )
        
        return model.to(self.device)
    
    def estimate_temperature(self, image: np.ndarray, food_type: str, 
                          ambient_temperature: Optional[float] = None) -> Dict:
        """
        Estimate food temperature from image
        
        Args:
            image: Food image
            food_type: Type of food
            ambient_temperature: Ambient temperature in Celsius (optional)
            
        Returns:
            Temperature analysis results
        """
        # Extract visual temperature indicators
        visual_analysis = self._analyze_visual_indicators(image)
        
        # Extract color-based temperature cues
        color_analysis = self._analyze_color_temperature(image)
        
        # Extract texture-based temperature cues
        texture_analysis = self._analyze_texture_temperature(image)
        
        # Extract environmental cues
        environmental_analysis = self._analyze_environmental_cues(image)
        
        # Predict temperature using ML model
        ml_prediction = self._predict_temperature_ml(image)
        
        # Combine all analyses
        combined_temperature = self._combine_temperature_estimates(
            visual_analysis, color_analysis, texture_analysis, 
            environmental_analysis, ml_prediction
        )
        
        # Validate against food type expectations
        validation = self._validate_temperature(combined_temperature, food_type)
        
        # Generate recommendations
        recommendations = self._generate_temperature_recommendations(
            combined_temperature, food_type, validation
        )
        
        return {
            'estimated_temperature': combined_temperature,
            'temperature_celsius': combined_temperature,
            'temperature_fahrenheit': self._celsius_to_fahrenheit(combined_temperature),
            'visual_analysis': visual_analysis,
            'color_analysis': color_analysis,
            'texture_analysis': texture_analysis,
            'environmental_analysis': environmental_analysis,
            'ml_prediction': ml_prediction,
            'validation': validation,
            'recommendations': recommendations,
            'serving_suggestions': self._get_serving_suggestions(combined_temperature, food_type)
        }
    
    def _analyze_visual_indicators(self, image: np.ndarray) -> Dict:
        """Analyze visual temperature indicators"""
        detected_indicators = []
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Check for steam
        steam_analysis = self._detect_steam(image, hsv, gray)
        if steam_analysis['detected']:
            detected_indicators.append({
                'type': 'steam',
                'confidence': steam_analysis['confidence'],
                'estimated_temp_range': steam_analysis['temperature_range']
            })
        
        # Check for condensation
        condensation_analysis = self._detect_condensation(image, hsv, gray)
        if condensation_analysis['detected']:
            detected_indicators.append({
                'type': 'condensation',
                'confidence': condensation_analysis['confidence'],
                'estimated_temp_range': condensation_analysis['temperature_range']
            })
        
        # Check for ice crystals
        ice_analysis = self._detect_ice_crystals(image, hsv, gray)
        if ice_analysis['detected']:
            detected_indicators.append({
                'type': 'ice_crystals',
                'confidence': ice_analysis['confidence'],
                'estimated_temp_range': ice_analysis['temperature_range']
            })
        
        # Check for melting
        melting_analysis = self._detect_melting(image, hsv, gray)
        if melting_analysis['detected']:
            detected_indicators.append({
                'type': 'melting',
                'confidence': melting_analysis['confidence'],
                'estimated_temp_range': melting_analysis['temperature_range']
            })
        
        return {
            'detected_indicators': detected_indicators,
            'primary_indicator': detected_indicators[0] if detected_indicators else None,
            'indicator_count': len(detected_indicators)
        }
    
    def _detect_steam(self, image: np.ndarray, hsv: np.ndarray, gray: np.ndarray) -> Dict:
        """Detect steam in image"""
        # Steam appears as white/gray wispy areas
        
        # Look for white/gray areas
        white_mask = cv2.inRange(image, (200, 200, 200), (255, 255, 255))
        gray_mask = cv2.inRange(image, (150, 150, 150), (200, 200, 200))
        steam_mask = cv2.bitwise_or(white_mask, gray_mask)
        
        # Check for wispy texture patterns
        # Steam has low texture complexity
        kernel = np.ones((5, 5), np.float32) / 25
        local_var = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        low_texture_areas = local_var < np.percentile(local_var, 30)
        
        # Combine color and texture
        steam_areas = cv2.bitwise_and(steam_mask > 0, low_texture_areas.astype(np.uint8))
        
        steam_ratio = np.sum(steam_areas) / steam_areas.size
        
        return {
            'detected': steam_ratio > 0.02,  # 2% threshold
            'confidence': min(steam_ratio * 20, 1.0),
            'area_ratio': steam_ratio,
            'temperature_range': (60, 100)
        }
    
    def _detect_condensation(self, image: np.ndarray, hsv: np.ndarray, gray: np.ndarray) -> Dict:
        """Detect condensation on food"""
        # Condensation appears as small droplets
        
        # Look for light blue/gray areas
        light_blue_mask = cv2.inRange(image, (150, 150, 200), (200, 200, 255))
        
        # Detect droplet patterns using blob detection
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 5
        params.maxArea = 50
        params.filterByCircularity = True
        params.minCircularity = 0.5
        
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(gray)
        
        droplet_count = len(keypoints)
        droplet_density = droplet_count / (image.shape[0] * image.shape[1] / 1000)
        
        return {
            'detected': droplet_density > 0.1,
            'confidence': min(droplet_density, 1.0),
            'droplet_count': droplet_count,
            'temperature_range': (5, 25)
        }
    
    def _detect_ice_crystals(self, image: np.ndarray, hsv: np.ndarray, gray: np.ndarray) -> Dict:
        """Detect ice crystals"""
        # Ice crystals appear as bright, crystalline structures
        
        # Look for very bright areas
        bright_mask = cv2.inRange(image, (220, 220, 255), (255, 255, 255))
        
        # Detect crystalline patterns using edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Ice crystals have high edge density in bright areas
        bright_areas = bright_mask > 0
        edge_density_in_bright = np.sum(edges[bright_areas]) / np.sum(bright_areas) if np.sum(bright_areas) > 0 else 0
        
        return {
            'detected': edge_density_in_bright > 0.1,
            'confidence': min(edge_density_in_bright * 5, 1.0),
            'edge_density': edge_density_in_bright,
            'temperature_range': (-20, 0)
        }
    
    def _detect_melting(self, image: np.ndarray, hsv: np.ndarray, gray: np.ndarray) -> Dict:
        """Detect melting indicators"""
        # Melting appears as liquid drips or runoff
        
        # Look for liquid-like patterns
        # Use morphological operations to detect drip patterns
        
        # Vertical line detection for drips
        vertical_kernel = np.ones((5, 1), np.uint8)
        vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, vertical_kernel)
        
        # Look for darker liquid areas
        liquid_mask = cv2.inRange(image, (100, 100, 150), (150, 150, 200))
        
        # Combine vertical patterns with liquid colors
        melt_areas = cv2.bitwise_and(liquid_mask > 0, vertical_lines > 128)
        
        melt_ratio = np.sum(melt_areas) / melt_areas.size
        
        return {
            'detected': melt_ratio > 0.01,
            'confidence': min(melt_ratio * 15, 1.0),
            'area_ratio': melt_ratio,
            'temperature_range': (0, 10)
        }
    
    def _analyze_color_temperature(self, image: np.ndarray) -> Dict:
        """Analyze temperature based on color cues"""
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Color temperature indicators
        # Hot foods tend to have warmer colors (more red/orange)
        # Cold foods tend to have cooler colors (more blue/white)
        
        # Calculate color balance
        mean_hsv = np.mean(hsv, axis=(0, 1))
        mean_lab = np.mean(lab, axis=(0, 1))
        
        # Warmth indicator (higher = warmer)
        warmth_indicator = (mean_hsv[0] / 180.0) * 0.5 + (mean_hsv[1] / 255.0) * 0.3 + (mean_hsv[2] / 255.0) * 0.2
        
        # Coolness indicator (higher = cooler)
        coolness_indicator = (1.0 - warmth_indicator)
        
        # Estimate temperature based on color
        if warmth_indicator > 0.6:
            color_temp_estimate = 40 + (warmth_indicator - 0.6) * 60  # 40-100C
        elif warmth_indicator > 0.4:
            color_temp_estimate = 15 + (warmth_indicator - 0.4) * 125  # 15-40C
        else:
            color_temp_estimate = -10 + warmth_indicator * 62.5  # -10-15C
        
        return {
            'warmth_indicator': float(warmth_indicator),
            'coolness_indicator': float(coolness_indicator),
            'color_temperature_estimate': float(color_temp_estimate),
            'dominant_hue': float(mean_hsv[0]),
            'saturation': float(mean_hsv[1]),
            'brightness': float(mean_hsv[2])
        }
    
    def _analyze_texture_temperature(self, image: np.ndarray) -> Dict:
        """Analyze temperature based on texture cues"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Texture complexity changes with temperature
        # Hot foods may have more texture (browning, crisping)
        # Cold foods may have less texture (frozen, smooth)
        
        # Calculate texture complexity
        local_var = cv2.filter2D(gray.astype(np.float32), -1, 
                               np.ones((5, 5), np.float32) / 25)
        texture_complexity = np.std(local_var)
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Texture-based temperature estimation
        if texture_complexity > 30:
            texture_temp_estimate = 45 + (texture_complexity - 30) * 1.5  # 45-75C
        elif texture_complexity > 15:
            texture_temp_estimate = 20 + (texture_complexity - 15) * 1.7  # 20-45C
        else:
            texture_temp_estimate = -5 + texture_complexity * 1.7  # -5-20C
        
        return {
            'texture_complexity': float(texture_complexity),
            'edge_density': float(edge_density),
            'texture_temperature_estimate': float(texture_temp_estimate)
        }
    
    def _analyze_environmental_cues(self, image: np.ndarray) -> Dict:
        """Analyze environmental temperature cues"""
        # Look at background and environmental factors
        
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Analyze background color temperature
        # This is a simplified approach - in practice would segment foreground/background
        
        # Overall scene warmth
        scene_warmth = np.mean(hsv[:, :, 0]) / 180.0
        
        # Environmental temperature estimation
        if scene_warmth > 0.6:
            env_temp_estimate = 25 + (scene_warmth - 0.6) * 25  # 25-40C
        elif scene_warmth > 0.4:
            env_temp_estimate = 15 + (scene_warmth - 0.4) * 50  # 15-25C
        else:
            env_temp_estimate = 5 + scene_warmth * 25  # 5-15C
        
        return {
            'scene_warmth': float(scene_warmth),
            'environmental_temperature_estimate': float(env_temp_estimate)
        }
    
    def _predict_temperature_ml(self, image: np.ndarray) -> Dict:
        """Predict temperature using ML model"""
        # Preprocess image
        input_tensor = self._preprocess_for_temperature(image)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Perform inference
        with torch.no_grad():
            temperature_tensor = self.model(input_tensor)
            temperature = temperature_tensor.item()
        
        return {
            'ml_temperature_estimate': float(temperature),
            'confidence': 0.7  # Placeholder confidence
        }
    
    def _preprocess_for_temperature(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for temperature estimation"""
        # Resize to model input size
        resized = cv2.resize(image, (224, 224))
        
        # Convert to tensor
        tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float()
        tensor = tensor / 255.0  # Normalize to [0, 1]
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)
    
    def _combine_temperature_estimates(self, visual: Dict, color: Dict, 
                                     texture: Dict, environmental: Dict,
                                     ml_prediction: Dict) -> float:
        """Combine different temperature estimates"""
        estimates = []
        weights = []
        
        # Visual indicators (high weight if detected)
        if visual['primary_indicator']:
            indicator = visual['primary_indicator']
            temp_range = indicator['estimated_temp_range']
            temp_estimate = (temp_range[0] + temp_range[1]) / 2
            estimates.append(temp_estimate)
            weights.append(indicator['confidence'])
        
        # Color-based estimate
        estimates.append(color['color_temperature_estimate'])
        weights.append(0.3)
        
        # Texture-based estimate
        estimates.append(texture['texture_temperature_estimate'])
        weights.append(0.2)
        
        # Environmental estimate (lower weight)
        estimates.append(environmental['environmental_temperature_estimate'])
        weights.append(0.1)
        
        # ML prediction
        estimates.append(ml_prediction['ml_temperature_estimate'])
        weights.append(0.4)
        
        # Weighted average
        if estimates and weights:
            combined_temp = np.average(estimates, weights=weights)
        else:
            combined_temp = 20.0  # Default room temperature
        
        return float(combined_temp)
    
    def _validate_temperature(self, temperature: float, food_type: str) -> Dict:
        """Validate temperature against food type expectations"""
        food_type_lower = food_type.lower()
        
        # Find appropriate temperature range
        expected_range = None
        for category, foods in self.temperature_ranges.items():
            for food, ranges in foods.items():
                if food in food_type_lower:
                    expected_range = ranges
                    break
            if expected_range:
                break
        
        if not expected_range:
            return {
                'valid': True,
                'confidence': 0.5,
                'message': 'No temperature expectations for this food type'
            }
        
        min_temp, max_temp = expected_range['min'], expected_range['max']
        optimal_temp = expected_range['optimal']
        
        # Check if temperature is in expected range
        in_range = min_temp <= temperature <= max_temp
        is_optimal = abs(temperature - optimal_temp) < 5
        
        confidence = 1.0 if in_range else max(0, 1.0 - abs(temperature - optimal_temp) / 20.0)
        
        return {
            'valid': in_range,
            'optimal': is_optimal,
            'confidence': confidence,
            'expected_range': (min_temp, max_temp),
            'optimal_temperature': optimal_temp,
            'message': self._get_validation_message(in_range, is_optimal, food_type)
        }
    
    def _get_validation_message(self, in_range: bool, is_optimal: bool, food_type: str) -> str:
        """Get validation message"""
        if is_optimal:
            return f"Temperature is optimal for {food_type}"
        elif in_range:
            return f"Temperature is acceptable for {food_type}"
        else:
            return f"Temperature may not be ideal for {food_type}"
    
    def _generate_temperature_recommendations(self, temperature: float, 
                                           food_type: str, validation: Dict) -> List[str]:
        """Generate temperature-based recommendations"""
        recommendations = []
        
        if not validation['valid']:
            if temperature < validation['expected_range'][0]:
                recommendations.append(f"Consider warming up the {food_type}")
            else:
                recommendations.append(f"Consider cooling down the {food_type}")
        
        if not validation['optimal']:
            optimal = validation['optimal_temperature']
            if temperature < optimal:
                recommendations.append(f"Best served at {optimal}C ({self._celsius_to_fahrenheit(optimal)}F)")
            else:
                recommendations.append(f"Best served at {optimal}C ({self._celsius_to_fahrenheit(optimal)}F)")
        
        # Safety recommendations
        if temperature > 70:
            recommendations.append("Caution: Very hot - allow to cool before consuming")
        elif temperature < 0:
            recommendations.append("Frozen - allow to thaw if needed")
        
        return recommendations
    
    def _get_serving_suggestions(self, temperature: float, food_type: str) -> Dict:
        """Get serving suggestions based on temperature"""
        suggestions = {
            'serving_style': 'unknown',
            'accompaniments': [],
            'storage_recommendations': []
        }
        
        if temperature > 60:
            suggestions['serving_style'] = 'hot'
            suggestions['accompaniments'] = ['napkins', 'plates', 'utensils']
            suggestions['storage_recommendations'] = ['consume while hot', 'reheat if needed']
        elif temperature > 20:
            suggestions['serving_style'] = 'warm'
            suggestions['accompaniments'] = ['plates', 'utensils']
            suggestions['storage_recommendations'] = ['serve at room temperature']
        elif temperature > 5:
            suggestions['serving_style'] = 'cool'
            suggestions['accompaniments'] = ['plates', 'optional garnish']
            suggestions['storage_recommendations'] = ['refrigerate after serving']
        else:
            suggestions['serving_style'] = 'cold'
            suggestions['accompaniments'] = ['cold plates', 'garnish']
            suggestions['storage_recommendations'] = ['keep refrigerated', 'serve chilled']
        
        return suggestions
    
    def _celsius_to_fahrenheit(self, celsius: float) -> float:
        """Convert Celsius to Fahrenheit"""
        return (celsius * 9/5) + 32
    
    def _fahrenheit_to_celsius(self, fahrenheit: float) -> float:
        """Convert Fahrenheit to Celsius"""
        return (fahrenheit - 32) * 5/9
