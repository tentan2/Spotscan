"""
Acidity Analyzer Module
Estimates acidity (pH) levels using color-based analysis and food science principles
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

class AcidityAnalyzer:
    """Analyzes food acidity using visual and chemical indicators"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize acidity analyzer
        
        Args:
            model_path: Path to pre-trained model
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.scaler = StandardScaler()
        
        # pH ranges for common foods
        self.ph_ranges = {
            'highly_acidic': {'range': (0.0, 3.0), 'examples': ['lemon', 'lime', 'vinegar', 'cranberry']},
            'moderately_acidic': {'range': (3.0, 4.5), 'examples': ['orange', 'apple', 'tomato', 'pineapple']},
            'slightly_acidic': {'range': (4.5, 6.0), 'examples': ['banana', 'peach', 'grape', 'strawberry']},
            'neutral': {'range': (6.0, 7.5), 'examples': ['water', 'milk', 'most vegetables', 'bread']},
            'slightly_alkaline': {'range': (7.5, 8.5), 'examples': ['egg_white', 'seafood', 'spinach']},
            'moderately_alkaline': {'range': (8.5, 10.0), 'examples': ['baking_soda', 'certain_cheeses']},
            'highly_alkaline': {'range': (10.0, 14.0), 'examples': ['lye', 'drain_cleaner']}
        }
        
        # Color-pH correlations
        self.color_ph_correlations = {
            'red_orange': {'ph_range': (2.5, 4.5), 'foods': ['tomato', 'strawberry', 'apple', 'orange']},
            'yellow_green': {'ph_range': (3.0, 5.0), 'foods': ['lemon', 'lime', 'grape', 'kiwi']},
            'purple_blue': {'ph_range': (2.5, 4.0), 'foods': ['blueberry', 'blackberry', 'plum', 'grape']},
            'green': {'ph_range': (5.0, 7.0), 'foods': ['vegetables', 'leafy_greens', 'cucumber']},
            'brown': {'ph_range': (4.5, 6.5), 'foods': ['bread', 'grains', 'nuts', 'coffee']}
        }
        
        # Acidity indicators from visual cues
        self.acidity_indicators = {
            'high_acid': {
                'color_hue': (0, 30),  # Red to orange
                'saturation': (0.6, 1.0),
                'brightness': (0.4, 0.8),
                'texture': 'smooth',
                'surface': 'glossy'
            },
            'medium_acid': {
                'color_hue': (30, 60),  # Yellow to green
                'saturation': (0.4, 0.8),
                'brightness': (0.5, 0.9),
                'texture': 'moderate',
                'surface': 'variable'
            },
            'low_acid': {
                'color_hue': (60, 180),  # Green to purple
                'saturation': (0.2, 0.6),
                'brightness': (0.3, 0.7),
                'texture': 'rough',
                'surface': 'matte'
            }
        }
    
    def _load_model(self, model_path: Optional[str]) -> nn.Module:
        """Load or create pH estimation model"""
        if model_path and Path(model_path).exists():
            model = torch.load(model_path, map_location=self.device)
            return model
        
        # Create CNN for pH estimation
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
            
            # Regression layers
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)  # Single output: pH value
        )
        
        return model.to(self.device)
    
    def estimate_acidity(self, image: np.ndarray, food_type: Optional[str] = None) -> Dict:
        """
        Estimate acidity (pH) level from food image
        
        Args:
            image: Food image
            food_type: Type of food (optional)
            
        Returns:
            Comprehensive acidity analysis
        """
        # Analyze color-based acidity indicators
        color_analysis = self._analyze_color_acidity(image)
        
        # Analyze texture-based acidity indicators
        texture_analysis = self._analyze_texture_acidity(image)
        
        # Analyze surface properties for acidity
        surface_analysis = self._analyze_surface_acidity(image)
        
        # Analyze ripeness indicators (affects acidity)
        ripeness_analysis = self._analyze_ripeness_acidity(image, food_type)
        
        # Predict pH using ML model
        ml_prediction = self._predict_ph_ml(image)
        
        # Combine all analyses
        combined_ph = self._combine_acidity_analyses(
            color_analysis, texture_analysis, surface_analysis,
            ripeness_analysis, ml_prediction, food_type
        )
        
        # Classify acidity level
        acidity_classification = self._classify_acidity_level(combined_ph)
        
        # Generate recommendations
        recommendations = self._generate_acidity_recommendations(
            combined_ph, acidity_classification, food_type
        )
        
        return {
            'food_type': food_type,
            'estimated_ph': combined_ph,
            'ph_category': acidity_classification,
            'color_analysis': color_analysis,
            'texture_analysis': texture_analysis,
            'surface_analysis': surface_analysis,
            'ripeness_analysis': ripeness_analysis,
            'ml_prediction': ml_prediction,
            'confidence_score': self._calculate_confidence(
                color_analysis, texture_analysis, surface_analysis, ml_prediction
            ),
            'recommendations': recommendations
        }
    
    def _analyze_color_acidity(self, image: np.ndarray) -> Dict:
        """Analyze color-based acidity indicators"""
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Calculate color statistics
        mean_hsv = np.mean(hsv, axis=(0, 1))
        mean_lab = np.mean(lab, axis=(0, 1))
        
        # Extract dominant colors
        dominant_colors = self._extract_dominant_colors(image, k=5)
        
        # Analyze color distribution
        color_distribution = self._analyze_color_distribution(hsv)
        
        # Estimate pH based on color
        ph_from_color = self._estimate_ph_from_color(mean_hsv, dominant_colors)
        
        # Analyze specific color indicators
        color_indicators = self._analyze_color_indicators(mean_hsv, mean_lab)
        
        return {
            'mean_hsv': mean_hsv.tolist(),
            'mean_lab': mean_lab.tolist(),
            'dominant_colors': dominant_colors,
            'color_distribution': color_distribution,
            'estimated_ph': ph_from_color,
            'color_indicators': color_indicators
        }
    
    def _extract_dominant_colors(self, image: np.ndarray, k: int = 5) -> List[Dict]:
        """Extract dominant colors using K-means clustering"""
        # Reshape image to be a list of pixels
        pixels = image.reshape(-1, 3)
        pixels = np.float32(pixels)
        
        # Define criteria and apply K-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert centers to uint8
        centers = np.uint8(centers)
        
        # Calculate percentage of each color
        label_counts = np.bincount(labels.flatten())
        total_pixels = len(labels)
        
        # Create color information
        dominant_colors = []
        
        for i, center in enumerate(centers):
            # Convert BGR to HSV
            hsv_color = cv2.cvtColor(np.array([[center]]), cv2.COLOR_BGR2HSV)[0][0]
            
            # Convert to hex
            rgb_color = [center[2], center[1], center[0]]
            hex_color = '#{:02x}{:02x}{:02x}'.format(*rgb_color)
            
            # Calculate percentage
            percentage = label_counts[i] / total_pixels
            
            dominant_colors.append({
                'rgb': rgb_color,
                'hsv': hsv_color.tolist(),
                'hex': hex_color,
                'percentage': float(percentage),
                'hue': float(hsv_color[0])
            })
        
        # Sort by percentage
        dominant_colors.sort(key=lambda x: x['percentage'], reverse=True)
        
        return dominant_colors
    
    def _analyze_color_distribution(self, hsv: np.ndarray) -> Dict:
        """Analyze color distribution in HSV space"""
        # Calculate histograms
        h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])
        
        # Normalize histograms
        h_hist = h_hist.flatten() / h_hist.sum()
        s_hist = s_hist.flatten() / s_hist.sum()
        v_hist = v_hist.flatten() / v_hist.sum()
        
        # Find dominant hue range
        dominant_hue = np.argmax(h_hist)
        
        # Calculate color balance
        red_orange_ratio = np.sum(h_hist[0:30]) / np.sum(h_hist)
        yellow_green_ratio = np.sum(h_hist[30:90]) / np.sum(h_hist)
        green_blue_ratio = np.sum(h_hist[90:150]) / np.sum(h_hist)
        purple_red_ratio = np.sum(h_hist[150:180]) / np.sum(h_hist)
        
        return {
            'hue_histogram': h_hist.tolist(),
            'saturation_histogram': s_hist.tolist(),
            'value_histogram': v_hist.tolist(),
            'dominant_hue': int(dominant_hue),
            'color_balance': {
                'red_orange': float(red_orange_ratio),
                'yellow_green': float(yellow_green_ratio),
                'green_blue': float(green_blue_ratio),
                'purple_red': float(purple_red_ratio)
            }
        }
    
    def _estimate_ph_from_color(self, mean_hsv: np.ndarray, dominant_colors: List[Dict]) -> float:
        """Estimate pH based on color characteristics"""
        # Use dominant hue as primary indicator
        dominant_hue = mean_hsv[0]
        saturation = mean_hsv[1] / 255.0
        value = mean_hsv[2] / 255.0
        
        # Base pH estimation from hue
        if dominant_hue < 30:  # Red to orange
            base_ph = 2.5 + (dominant_hue / 30) * 2.0  # 2.5-4.5
        elif dominant_hue < 60:  # Yellow to green
            base_ph = 4.5 + ((dominant_hue - 30) / 30) * 0.5  # 4.5-5.0
        elif dominant_hue < 120:  # Green
            base_ph = 5.0 + ((dominant_hue - 60) / 60) * 2.0  # 5.0-7.0
        elif dominant_hue < 150:  # Blue-green
            base_ph = 7.0 + ((dominant_hue - 120) / 30) * 0.5  # 7.0-7.5
        else:  # Purple to red
            base_ph = 7.5 + ((dominant_hue - 150) / 30) * 1.5  # 7.5-9.0
        
        # Adjust based on saturation (higher saturation often indicates more acid)
        saturation_adjustment = (1.0 - saturation) * 0.5
        
        # Adjust based on brightness (darker colors might indicate more acid)
        brightness_adjustment = (1.0 - value) * 0.3
        
        # Combine adjustments
        adjusted_ph = base_ph - (saturation_adjustment + brightness_adjustment)
        
        # Clamp to valid pH range
        return max(0.0, min(14.0, adjusted_ph))
    
    def _analyze_color_indicators(self, hsv: np.ndarray, lab: np.ndarray) -> Dict:
        """Analyze specific color indicators for acidity"""
        hue = hsv[0]
        saturation = hsv[1] / 255.0
        value = hsv[2] / 255.0
        
        # Check against acidity indicator ranges
        indicators = {}
        
        # High acid indicators
        if 0 <= hue <= 30 and saturation > 0.6:
            indicators['high_acid_confidence'] = min(saturation * 1.5, 1.0)
        else:
            indicators['high_acid_confidence'] = 0.0
        
        # Medium acid indicators
        if 30 <= hue <= 60 and saturation > 0.4:
            indicators['medium_acid_confidence'] = min(saturation * 1.2, 1.0)
        else:
            indicators['medium_acid_confidence'] = 0.0
        
        # Low acid indicators
        if hue > 60 or saturation < 0.4:
            indicators['low_acid_confidence'] = min((1.0 - saturation) * 1.5, 1.0)
        else:
            indicators['low_acid_confidence'] = 0.0
        
        # LAB color space indicators
        # High a* value (green-red axis) can indicate acidity
        a_star = lab[1]
        indicators['lab_acidity_indicator'] = max(0.0, a_star / 128.0)
        
        return indicators
    
    def _analyze_texture_acidity(self, image: np.ndarray) -> Dict:
        """Analyze texture-based acidity indicators"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Texture complexity (acidic foods often have smoother textures)
        local_var = cv2.filter2D(gray.astype(np.float32), -1, 
                               np.ones((5, 5), np.float32) / 25)
        texture_complexity = np.std(local_var)
        
        # Edge density (acidic foods may have different edge patterns)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Surface roughness (acidic foods often appear smoother)
        surface_roughness = self._calculate_surface_roughness(gray)
        
        # Estimate pH from texture
        # Smoother textures often indicate higher acidity
        texture_ph = 7.0 - (texture_complexity / 50.0) * 2.0
        
        return {
            'texture_complexity': float(texture_complexity),
            'edge_density': float(edge_density),
            'surface_roughness': float(surface_roughness),
            'estimated_ph': float(texture_ph)
        }
    
    def _calculate_surface_roughness(self, gray: np.ndarray) -> float:
        """Calculate surface roughness"""
        # Use gradient magnitude as roughness indicator
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        return np.mean(gradient_magnitude)
    
    def _analyze_surface_acidity(self, image: np.ndarray) -> Dict:
        """Analyze surface properties for acidity indicators"""
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Surface sheen/gloss (acidic foods often have more sheen)
        value_channel = hsv[:, :, 2]
        bright_areas = np.sum(value_channel > 200)
        sheen_ratio = bright_areas / (image.shape[0] * image.shape[1])
        
        # Moisture indicators (affects perceived acidity)
        saturation_channel = hsv[:, :, 1]
        avg_saturation = np.mean(saturation_channel)
        
        # Surface reflection patterns
        reflection_analysis = self._analyze_reflection_patterns(image)
        
        # Estimate pH from surface properties
        surface_ph = 7.0 - (sheen_ratio * 3.0) - (avg_saturation / 255.0 * 2.0)
        
        return {
            'sheen_ratio': float(sheen_ratio),
            'avg_saturation': float(avg_saturation),
            'reflection_analysis': reflection_analysis,
            'estimated_ph': float(surface_ph)
        }
    
    def _analyze_reflection_patterns(self, image: np.ndarray) -> Dict:
        """Analyze reflection patterns on surface"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Look for specular highlights
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Analyze highlight patterns
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate highlight characteristics
        highlight_count = len(contours)
        total_highlight_area = sum(cv2.contourArea(c) for c in contours)
        
        # Analyze highlight shapes
        circular_highlights = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                if circularity > 0.7:
                    circular_highlights += 1
        
        return {
            'highlight_count': highlight_count,
            'total_highlight_area': float(total_highlight_area),
            'circular_highlights': circular_highlights,
            'reflection_intensity': float(total_highlight_area / (image.shape[0] * image.shape[1]))
        }
    
    def _analyze_ripeness_acidity(self, image: np.ndarray, food_type: Optional[str]) -> Dict:
        """Analyze ripeness indicators that affect acidity"""
        if not food_type:
            return {'ripeness_adjustment': 0.0, 'ripeness_stage': 'unknown'}
        
        food_type_lower = food_type.lower()
        
        # Different foods have different acidity changes with ripeness
        if any(fruit in food_type_lower for fruit in ['banana', 'apple', 'peach', 'mango']):
            # These become less acidic as they ripen
            ripeness_ph_adjustment = self._analyze_fruit_ripeness(image, food_type_lower)
        elif any(veg in food_type_lower for veg in ['tomato', 'citrus']):
            # These may become more acidic as they ripen
            ripeness_ph_adjustment = self._analyze_vegetable_ripeness(image, food_type_lower)
        else:
            ripeness_ph_adjustment = 0.0
        
        return {
            'ripeness_adjustment': ripeness_ph_adjustment,
            'food_type': food_type,
            'ripeness_stage': self._estimate_ripeness_stage(image, food_type_lower)
        }
    
    def _analyze_fruit_ripeness(self, image: np.ndarray, food_type: str) -> float:
        """Analyze fruit ripeness for acidity adjustment"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # For most fruits, ripening reduces acidity
        # Look for color changes indicating ripeness
        
        if 'banana' in food_type:
            # Green to yellow reduces acidity
            green_mask = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255))
            yellow_mask = cv2.inRange(hsv, (20, 50, 50), (35, 255, 255))
            
            green_ratio = np.sum(green_mask > 0) / green_mask.size
            yellow_ratio = np.sum(yellow_mask > 0) / yellow_mask.size
            
            # More yellow = riper = less acidic
            ripeness_factor = yellow_ratio / (green_ratio + yellow_ratio + 0.001)
            return -ripeness_factor * 1.5  # Reduce pH by up to 1.5
            
        elif 'apple' in food_type:
            # Red apples are often less acidic than green
            red_mask = cv2.inRange(hsv, (0, 50, 50), (25, 255, 255))
            green_mask = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255))
            
            red_ratio = np.sum(red_mask > 0) / red_mask.size
            green_ratio = np.sum(green_mask > 0) / green_mask.size
            
            ripeness_factor = red_ratio / (green_ratio + red_ratio + 0.001)
            return -ripeness_factor * 1.0
        
        return 0.0
    
    def _analyze_vegetable_ripeness(self, image: np.ndarray, food_type: str) -> float:
        """Analyze vegetable ripeness for acidity adjustment"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        if 'tomato' in food_type:
            # Red tomatoes are more acidic than green
            red_mask = cv2.inRange(hsv, (0, 50, 50), (25, 255, 255))
            green_mask = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255))
            
            red_ratio = np.sum(red_mask > 0) / red_mask.size
            green_ratio = np.sum(green_mask > 0) / green_mask.size
            
            ripeness_factor = red_ratio / (green_ratio + red_ratio + 0.001)
            return ripeness_factor * 0.8  # Increase pH by up to 0.8
        
        return 0.0
    
    def _estimate_ripeness_stage(self, image: np.ndarray, food_type: str) -> str:
        """Estimate ripeness stage"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        if 'banana' in food_type:
            green_mask = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255))
            yellow_mask = cv2.inRange(hsv, (20, 50, 50), (35, 255, 255))
            brown_mask = cv2.inRange(hsv, (8, 50, 50), (25, 255, 255))
            
            green_ratio = np.sum(green_mask > 0) / green_mask.size
            yellow_ratio = np.sum(yellow_mask > 0) / yellow_mask.size
            brown_ratio = np.sum(brown_mask > 0) / brown_mask.size
            
            if green_ratio > 0.5:
                return 'unripe'
            elif yellow_ratio > 0.5:
                return 'ripe'
            elif brown_ratio > 0.2:
                return 'overripe'
            else:
                return 'unknown'
        
        return 'unknown'
    
    def _predict_ph_ml(self, image: np.ndarray) -> Dict:
        """Predict pH using ML model"""
        # Preprocess image
        input_tensor = self._preprocess_for_ph(image)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Perform inference
        with torch.no_grad():
            ph_tensor = self.model(input_tensor)
            ph_value = ph_tensor.item()
        
        return {
            'ml_ph_estimate': float(ph_value),
            'confidence': 0.7  # Placeholder confidence
        }
    
    def _preprocess_for_ph(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for pH estimation"""
        # Resize to model input size
        resized = cv2.resize(image, (224, 224))
        
        # Convert to tensor
        tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float()
        tensor = tensor / 255.0  # Normalize to [0, 1]
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)
    
    def _combine_acidity_analyses(self, color: Dict, texture: Dict, surface: Dict,
                                 ripeness: Dict, ml: Dict, food_type: Optional[str]) -> float:
        """Combine all acidity analyses"""
        estimates = []
        weights = []
        
        # Color-based estimate
        if 'estimated_ph' in color:
            estimates.append(color['estimated_ph'])
            weights.append(0.3)
        
        # Texture-based estimate
        if 'estimated_ph' in texture:
            estimates.append(texture['estimated_ph'])
            weights.append(0.2)
        
        # Surface-based estimate
        if 'estimated_ph' in surface:
            estimates.append(surface['estimated_ph'])
            weights.append(0.2)
        
        # ML estimate
        if 'ml_ph_estimate' in ml:
            estimates.append(ml['ml_ph_estimate'])
            weights.append(0.3)
        
        # Weighted average
        if estimates and weights:
            combined_ph = np.average(estimates, weights=weights)
        else:
            combined_ph = 7.0  # Default neutral pH
        
        # Apply ripeness adjustment
        if 'ripeness_adjustment' in ripeness:
            combined_ph += ripeness['ripeness_adjustment']
        
        # Apply food type adjustment
        if food_type:
            food_adjustment = self._get_food_type_adjustment(food_type)
            combined_ph += food_adjustment
        
        # Clamp to valid pH range
        return max(0.0, min(14.0, combined_ph))
    
    def _get_food_type_adjustment(self, food_type: str) -> float:
        """Get pH adjustment for specific food types"""
        food_type_lower = food_type.lower()
        
        # Known pH adjustments for common foods
        adjustments = {
            'lemon': -2.5,      # Very acidic
            'lime': -2.0,        # Very acidic
            'vinegar': -2.0,      # Very acidic
            'tomato': -0.5,      # Moderately acidic
            'orange': -0.3,      # Slightly acidic
            'apple': -0.2,       # Slightly acidic
            'banana': 0.5,       # Slightly alkaline when ripe
            'spinach': 0.2,      # Slightly alkaline
            'water': 0.0,        # Neutral
            'milk': 0.1,         # Slightly alkaline
            'bread': 0.3,        # Slightly alkaline
        }
        
        for food, adjustment in adjustments.items():
            if food in food_type_lower:
                return adjustment
        
        return 0.0
    
    def _classify_acidity_level(self, ph: float) -> Dict:
        """Classify pH level into categories"""
        for level, info in self.ph_ranges.items():
            min_ph, max_ph = info['range']
            if min_ph <= ph <= max_ph:
                return {
                    'level': level,
                    'ph_range': info['range'],
                    'examples': info['examples'],
                    'description': self._get_acidity_description(level)
                }
        
        return {
            'level': 'unknown',
            'ph_range': (0, 14),
            'examples': [],
            'description': 'pH level outside normal ranges'
        }
    
    def _get_acidity_description(self, level: str) -> str:
        """Get description for acidity level"""
        descriptions = {
            'highly_acidic': 'Very acidic - may cause tooth enamel erosion',
            'moderately_acidic': 'Acidic - normal for many fruits',
            'slightly_acidic': 'Mildly acidic - generally safe',
            'neutral': 'Neutral pH - balanced',
            'slightly_alkaline': 'Mildly alkaline - generally beneficial',
            'moderately_alkaline': 'Alkaline - may affect digestion',
            'highly_alkaline': 'Very alkaline - potentially harmful'
        }
        
        return descriptions.get(level, 'Unknown acidity level')
    
    def _calculate_confidence(self, color: Dict, texture: Dict, surface: Dict, ml: Dict) -> float:
        """Calculate overall confidence in pH estimation"""
        confidences = []
        
        if 'color_indicators' in color:
            max_indicator = max(color['color_indicators'].values())
            confidences.append(max_indicator)
        
        if 'texture_complexity' in texture:
            # Lower texture complexity often indicates more confidence in acidity
            texture_conf = 1.0 - min(texture['texture_complexity'] / 50.0, 1.0)
            confidences.append(texture_conf)
        
        if 'sheen_ratio' in surface:
            surface_conf = min(surface['sheen_ratio'] * 2, 1.0)
            confidences.append(surface_conf)
        
        if 'ml_ph_estimate' in ml:
            confidences.append(0.7)  # ML model confidence
        
        if confidences:
            return np.mean(confidences)
        else:
            return 0.5  # Default confidence
    
    def _generate_acidity_recommendations(self, ph: float, classification: Dict, 
                                       food_type: Optional[str]) -> List[str]:
        """Generate recommendations based on acidity level"""
        recommendations = []
        
        level = classification.get('level', 'neutral')
        
        # General pH-based recommendations
        if ph < 3.0:
            recommendations.append("Very acidic - consume in moderation")
            recommendations.append("May cause tooth sensitivity - rinse with water after consumption")
        elif ph < 4.5:
            recommendations.append("Acidic - generally safe in normal amounts")
            recommendations.append("Consider pairing with alkaline foods to balance")
        elif ph > 9.0:
            recommendations.append("Highly alkaline - may affect digestion")
            recommendations.append("Consume in small amounts")
        elif 7.5 < ph <= 9.0:
            recommendations.append("Alkaline - may have health benefits")
            recommendations.append("Good for balancing acidic foods")
        
        # Food-specific recommendations
        if food_type:
            food_type_lower = food_type.lower()
            
            if 'citrus' in food_type_lower and ph < 3.0:
                recommendations.append("Citrus fruits are healthy but very acidic")
                recommendations.append("Excellent source of vitamin C")
            elif 'tomato' in food_type_lower and ph < 4.5:
                recommendations.append("Tomatoes contain beneficial lycopene")
                recommendations.append("Acidity helps with nutrient absorption")
            elif 'coffee' in food_type_lower and ph < 5.0:
                recommendations.append("Coffee is mildly acidic")
                recommendations.append("Consider adding milk to reduce acidity")
        
        # Health recommendations
        if 6.5 <= ph <= 7.5:
            recommendations.append("Neutral pH - ideal for maintaining body balance")
        
        return recommendations
