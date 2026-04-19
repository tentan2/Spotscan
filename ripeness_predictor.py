"""
Ripeness Predictor Module
Predicts ripeness stages and optimal consumption timing for fruits and vegetables
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
from datetime import datetime, timedelta

class RipenessPredictor:
    """Predicts ripeness stages and optimal consumption timing"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize ripeness predictor
        
        Args:
            model_path: Path to pre-trained model
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.scaler = StandardScaler()
        
        # Ripeness characteristics for different produce
        self.ripeness_profiles = self._load_ripeness_profiles()
        
        # Color ripeness indicators for common fruits
        self.color_indicators = {
            'banana': {
                'unripe': {'hsv_range': [(35, 50, 50), (65, 255, 255)], 'description': 'Green'},
                'ripe': {'hsv_range': [(20, 50, 50), (35, 255, 255)], 'description': 'Yellow with green tips'},
                'overripe': {'hsv_range': [(10, 50, 50), (25, 255, 255)], 'description': 'Yellow with brown spots'}
            },
            'avocado': {
                'unripe': {'hsv_range': [(35, 50, 50), (85, 255, 255)], 'description': 'Bright green'},
                'ripe': {'hsv_range': [(25, 50, 50), (45, 255, 255)], 'description': 'Darker green'},
                'overripe': {'hsv_range': [(10, 30, 50), (30, 255, 255)], 'description': 'Dark green/brown'}
            },
            'tomato': {
                'unripe': {'hsv_range': [(35, 50, 50), (85, 255, 255)], 'description': 'Green'},
                'ripe': {'hsv_range': [(0, 50, 50), (25, 255, 255)], 'description': 'Red'},
                'overripe': {'hsv_range': [(0, 30, 50), (15, 255, 255)], 'description': 'Dark red/soft'}
            },
            'mango': {
                'unripe': {'hsv_range': [(35, 50, 50), (85, 255, 255)], 'description': 'Green'},
                'ripe': {'hsv_range': [(20, 50, 50), (40, 255, 255)], 'description': 'Yellow-orange'},
                'overripe': {'hsv_range': [(10, 30, 50), (30, 255, 255)], 'description': 'Orange with dark spots'}
            },
            'strawberry': {
                'unripe': {'hsv_range': [(35, 50, 50), (85, 255, 255)], 'description': 'White/green'},
                'ripe': {'hsv_range': [(0, 50, 50), (10, 255, 255)], 'description': 'Bright red'},
                'overripe': {'hsv_range': [(0, 30, 50), (5, 255, 255)], 'description': 'Dark red/bruised'}
            }
        }
    
    def _load_model(self, model_path: Optional[str]) -> nn.Module:
        """Load or create ripeness prediction model"""
        if model_path and Path(model_path).exists():
            model = torch.load(model_path, map_location=self.device)
            return model
        
        # Create EfficientNet-based model for ripeness prediction
        model = nn.Sequential(
            # Input features would be extracted in preprocessing
            nn.Linear(512, 256),  # Feature dimension
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3),  # unripe, ripe, overripe
            nn.Softmax(dim=1)
        )
        
        return model.to(self.device)
    
    def _load_ripeness_profiles(self) -> Dict:
        """Load ripeness profiles for different produce"""
        return {
            'banana': {
                'unripe_days_to_ripe': 2-4,
                'ripe_days_to_overripe': 2-3,
                'optimal_uses': {
                    'unripe': ['cooking', 'banana bread', 'fried'],
                    'ripe': ['eating fresh', 'smoothies', 'fruit salad'],
                    'overripe': ['smoothies', 'baking', 'ice cream']
                },
                'storage_tips': {
                    'unripe': 'Room temperature, away from direct sunlight',
                    'ripe': 'Room temperature for 1-2 days, then refrigerate',
                    'overripe': 'Refrigerate immediately or freeze'
                }
            },
            'avocado': {
                'unripe_days_to_ripe': 3-5,
                'ripe_days_to_overripe': 2-3,
                'optimal_uses': {
                    'unripe': ['Not recommended - wait to ripen'],
                    'ripe': ['guacamole', 'toast', 'salads'],
                    'overripe': ['smoothies', 'baking', 'dressings']
                },
                'storage_tips': {
                    'unripe': 'Room temperature in paper bag to speed ripening',
                    'ripe': 'Refrigerate for 2-3 days',
                    'overripe': 'Use immediately or freeze'
                }
            },
            'tomato': {
                'unripe_days_to_ripe': 3-7,
                'ripe_days_to_overripe': 3-5,
                'optimal_uses': {
                    'unripe': ['Fried green tomatoes', 'pickling'],
                    'ripe': ['Salads', 'sandwiches', 'sauces'],
                    'overripe': ['Sauces', 'soups', 'canning']
                },
                'storage_tips': {
                    'unripe': 'Room temperature, stem side down',
                    'ripe': 'Room temperature, avoid refrigeration',
                    'overripe': 'Refrigerate and use quickly'
                }
            },
            'mango': {
                'unripe_days_to_ripe': 2-5,
                'ripe_days_to_overripe': 2-4,
                'optimal_uses': {
                    'unripe': ['Pickling', 'salads', 'cooking'],
                    'ripe': ['Eating fresh', 'smoothies', 'desserts'],
                    'overripe': ['Smoothies', 'juices', 'ice cream']
                },
                'storage_tips': {
                    'unripe': 'Room temperature in paper bag',
                    'ripe': 'Refrigerate for up to 5 days',
                    'overripe': 'Freeze for later use'
                }
            }
        }
    
    def predict_ripeness(self, image: np.ndarray, produce_type: str) -> Dict:
        """
        Predict ripeness stage and provide recommendations
        
        Args:
            image: Produce image
            produce_type: Type of produce (banana, avocado, tomato, etc.)
            
        Returns:
            Ripeness analysis and recommendations
        """
        # Extract ripeness features
        features = self._extract_ripeness_features(image, produce_type)
        
        # Analyze color-based ripeness indicators
        color_analysis = self._analyze_color_ripeness(image, produce_type)
        
        # Analyze texture-based ripeness indicators
        texture_analysis = self._analyze_texture_ripeness(image)
        
        # Analyze shape and size indicators
        shape_analysis = self._analyze_shape_ripeness(image, produce_type)
        
        # Predict ripeness using ML model
        ripeness_prediction = self._predict_ripeness_stage(features)
        
        # Calculate confidence scores
        confidence_scores = self._calculate_confidence_scores(
            color_analysis, texture_analysis, shape_analysis
        )
        
        # Get ripeness profile
        profile = self.ripeness_profiles.get(produce_type.lower(), {})
        
        # Generate timing predictions
        timing_predictions = self._predict_timing(
            ripeness_prediction['predicted_stage'], 
            profile
        )
        
        # Generate recommendations
        recommendations = self._generate_ripeness_recommendations(
            ripeness_prediction['predicted_stage'],
            produce_type,
            profile
        )
        
        return {
            'produce_type': produce_type,
            'current_stage': ripeness_prediction['predicted_stage'],
            'confidence': ripeness_prediction['confidence'],
            'color_analysis': color_analysis,
            'texture_analysis': texture_analysis,
            'shape_analysis': shape_analysis,
            'confidence_scores': confidence_scores,
            'timing_predictions': timing_predictions,
            'recommendations': recommendations,
            'optimal_uses': profile.get('optimal_uses', {}).get(ripeness_prediction['predicted_stage'], []),
            'storage_tips': profile.get('storage_tips', {}).get(ripeness_prediction['predicted_stage'], '')
        }
    
    def _extract_ripeness_features(self, image: np.ndarray, produce_type: str) -> np.ndarray:
        """Extract features for ripeness prediction"""
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Color features
        color_features = [
            np.mean(hsv, axis=(0, 1)),  # HSV means
            np.std(hsv, axis=(0, 1)),   # HSV stds
            np.mean(lab, axis=(0, 1)),  # LAB means
            np.std(lab, axis=(0, 1))    # LAB stds
        ]
        
        # Texture features
        texture_features = self._calculate_ripeness_texture_features(gray)
        
        # Shape features
        shape_features = self._calculate_shape_features(image)
        
        # Specific ripeness indicators for produce type
        specific_features = self._extract_produce_specific_features(image, produce_type)
        
        # Combine all features
        features = np.concatenate([
            np.array(color_features).flatten(),
            texture_features,
            shape_features,
            specific_features
        ])
        
        return features
    
    def _analyze_color_ripeness(self, image: np.ndarray, produce_type: str) -> Dict:
        """Analyze color-based ripeness indicators"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Get color indicators for this produce type
        indicators = self.color_indicators.get(produce_type.lower(), {})
        
        if not indicators:
            return {'error': f'No color indicators available for {produce_type}'}
        
        # Calculate color match scores for each ripeness stage
        color_scores = {}
        
        for stage, indicator in indicators.items():
            hsv_range = indicator['hsv_range']
            lower = np.array(hsv_range[0])
            upper = np.array(hsv_range[1])
            
            # Create mask for this color range
            mask = cv2.inRange(hsv, lower, upper)
            score = np.sum(mask > 0) / mask.size
            
            color_scores[stage] = {
                'score': float(score),
                'description': indicator['description'],
                'dominant_pixels': int(np.sum(mask > 0))
            }
        
        # Determine dominant color stage
        dominant_stage = max(color_scores.keys(), key=lambda x: color_scores[x]['score'])
        
        # Calculate color uniformity (ripe produce often has uniform color)
        color_uniformity = self._calculate_color_uniformity(hsv)
        
        return {
            'color_scores': color_scores,
            'dominant_color_stage': dominant_stage,
            'color_uniformity': float(color_uniformity),
            'dominant_colors': self._get_dominant_colors(image)
        }
    
    def _analyze_texture_ripeness(self, image: np.ndarray) -> Dict:
        """Analyze texture-based ripeness indicators"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate surface roughness (ripe produce often smoother)
        roughness = self._calculate_surface_roughness(gray)
        
        # Calculate skin smoothness
        smoothness = self._calculate_skin_smoothness(gray)
        
        # Detect wrinkles (overripe indicator)
        wrinkles = self._detect_wrinkles(gray)
        
        # Detect soft spots (overripe indicator)
        soft_spots = self._detect_soft_spots(image)
        
        # Calculate texture complexity
        texture_complexity = self._calculate_texture_complexity(gray)
        
        return {
            'surface_roughness': float(roughness),
            'skin_smoothness': float(smoothness),
            'wrinkle_detected': bool(wrinkles),
            'soft_spot_ratio': float(soft_spots),
            'texture_complexity': float(texture_complexity)
        }
    
    def _analyze_shape_ripeness(self, image: np.ndarray, produce_type: str) -> Dict:
        """Analyze shape-based ripeness indicators"""
        # Find contours
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {'error': 'No contours found'}
        
        # Get largest contour (main produce)
        main_contour = max(contours, key=cv2.contourArea)
        
        # Calculate shape features
        area = cv2.contourArea(main_contour)
        perimeter = cv2.arcLength(main_contour, True)
        
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        else:
            circularity = 0
        
        # Calculate bounding box
        x, y, w, h = cv2.boundingRect(main_contour)
        aspect_ratio = w / h if h > 0 else 0
        
        # Calculate convexity
        hull = cv2.convexHull(main_contour)
        hull_area = cv2.contourArea(hull)
        convexity = area / hull_area if hull_area > 0 else 0
        
        # Detect deformations (overripe indicator)
        deformations = self._detect_shape_deformations(main_contour)
        
        return {
            'area': float(area),
            'perimeter': float(perimeter),
            'circularity': float(circularity),
            'aspect_ratio': float(aspect_ratio),
            'convexity': float(convexity),
            'deformations': deformations
        }
    
    def _calculate_ripeness_texture_features(self, gray: np.ndarray) -> np.ndarray:
        """Calculate texture features specific to ripeness"""
        # Calculate Local Binary Pattern
        lbp = self._calculate_lbp(gray)
        lbp_hist = np.histogram(lbp, bins=256)[0]
        lbp_hist = lbp_hist / lbp_hist.sum()
        
        # Calculate edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Calculate local variance
        kernel = np.ones((5, 5), np.float32) / 25
        local_var = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        variance_features = [
            np.mean(local_var),
            np.std(local_var),
            np.percentile(local_var, 25),
            np.percentile(local_var, 75)
        ]
        
        return np.concatenate([lbp_hist[:50], [edge_density], variance_features])
    
    def _calculate_shape_features(self, image: np.ndarray) -> np.ndarray:
        """Calculate shape features for ripeness analysis"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return np.zeros(10)
        
        main_contour = max(contours, key=cv2.contourArea)
        
        # Calculate various shape descriptors
        area = cv2.contourArea(main_contour)
        perimeter = cv2.arcLength(main_contour, True)
        
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        else:
            circularity = 0
        
        # Moments
        moments = cv2.moments(main_contour)
        hu_moments = cv2.HuMoments(moments).flatten()
        
        # Bounding box features
        x, y, w, h = cv2.boundingRect(main_contour)
        extent = area / (w * h) if (w * h) > 0 else 0
        
        return np.array([area, perimeter, circularity, extent, *hu_moments])
    
    def _extract_produce_specific_features(self, image: np.ndarray, produce_type: str) -> np.ndarray:
        """Extract features specific to certain produce types"""
        produce_type = produce_type.lower()
        
        if produce_type == 'banana':
            return self._extract_banana_features(image)
        elif produce_type == 'avocado':
            return self._extract_avocado_features(image)
        elif produce_type == 'tomato':
            return self._extract_tomato_features(image)
        else:
            return np.array([])
    
    def _extract_banana_features(self, image: np.ndarray) -> np.ndarray:
        """Extract banana-specific features"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Brown spots (overripe indicator)
        lower_brown = np.array([8, 50, 50])
        upper_brown = np.array([25, 255, 255])
        brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
        brown_ratio = np.sum(brown_mask > 0) / brown_mask.size
        
        # Green areas (unripe indicator)
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        green_ratio = np.sum(green_mask > 0) / green_mask.size
        
        return np.array([brown_ratio, green_ratio])
    
    def _extract_avocado_features(self, image: np.ndarray) -> np.ndarray:
        """Extract avocado-specific features"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Color distribution analysis
        green_ratio = np.sum((hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 85)) / hsv.size
        brown_ratio = np.sum((hsv[:, :, 0] >= 8) & (hsv[:, :, 0] <= 25)) / hsv.size
        
        return np.array([green_ratio, brown_ratio])
    
    def _extract_tomato_features(self, image: np.ndarray) -> np.ndarray:
        """Extract tomato-specific features"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Red color intensity
        red_mask = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([25, 255, 255]))
        red_ratio = np.sum(red_mask > 0) / red_mask.size
        
        # Green color (unripe)
        green_mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
        green_ratio = np.sum(green_mask > 0) / green_mask.size
        
        return np.array([red_ratio, green_ratio])
    
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
    
    def _calculate_surface_roughness(self, gray: np.ndarray) -> float:
        """Calculate surface roughness"""
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        return np.mean(gradient_magnitude)
    
    def _calculate_skin_smoothness(self, gray: np.ndarray) -> float:
        """Calculate skin smoothness score"""
        roughness = self._calculate_surface_roughness(gray)
        return 1.0 / (1.0 + roughness)
    
    def _detect_wrinkles(self, gray: np.ndarray) -> bool:
        """Detect wrinkles in produce skin"""
        # Use edge detection to find wrinkle patterns
        edges = cv2.Canny(gray, 50, 150)
        
        # Look for fine line patterns
        kernel = np.ones((3, 1), np.uint8)
        vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Calculate line density
        line_density = np.sum(vertical_lines > 0) / vertical_lines.size
        
        return line_density > 0.05  # 5% threshold
    
    def _detect_soft_spots(self, image: np.ndarray) -> float:
        """Detect soft spots (bruises) in produce"""
        # Soft spots often appear as darker, less textured areas
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find dark areas
        value_channel = hsv[:, :, 2]
        dark_mask = value_channel < np.percentile(value_channel, 25)
        
        # Check texture in dark areas
        kernel = np.ones((5, 5), np.float32) / 25
        local_var = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        
        # Soft spots have low texture variance
        soft_areas = dark_mask & (local_var < np.percentile(local_var, 30))
        
        return np.sum(soft_areas) / soft_areas.size if soft_areas.size > 0 else 0
    
    def _calculate_texture_complexity(self, gray: np.ndarray) -> float:
        """Calculate overall texture complexity"""
        # Use entropy as a measure of texture complexity
        histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])
        histogram = histogram.flatten()
        histogram = histogram / histogram.sum()
        
        # Remove zero values
        histogram = histogram[histogram > 0]
        
        entropy = -np.sum(histogram * np.log2(histogram))
        return entropy
    
    def _detect_shape_deformations(self, contour: np.ndarray) -> Dict:
        """Detect shape deformations"""
        # Calculate contour deviations from ideal shape
        hull = cv2.convexHull(contour)
        
        # Calculate convexity defects
        if len(contour) > 3 and len(hull) > 3:
            defects = cv2.convexityDefects(contour, hull)
            if defects is not None:
                defect_count = len(defects)
            else:
                defect_count = 0
        else:
            defect_count = 0
        
        # Calculate solidity
        area = cv2.contourArea(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        return {
            'defect_count': defect_count,
            'solidity': float(solidity),
            'deformed': defect_count > 5 or solidity < 0.9
        }
    
    def _calculate_color_uniformity(self, hsv: np.ndarray) -> float:
        """Calculate color uniformity score"""
        hue_std = np.std(hsv[:, :, 0])
        sat_std = np.std(hsv[:, :, 1])
        val_std = np.std(hsv[:, :, 2])
        
        uniformity = 1.0 / (1.0 + (hue_std + sat_std + val_std) / 3.0)
        return uniformity
    
    def _get_dominant_colors(self, image: np.ndarray) -> List[str]:
        """Get dominant colors in hex format"""
        # Reshape image to be a list of pixels
        pixels = image.reshape(-1, 3)
        pixels = np.float32(pixels)
        
        # K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(pixels, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert to hex colors
        centers = np.uint8(centers)
        hex_colors = []
        
        for color in centers:
            # Convert BGR to RGB
            rgb_color = [color[2], color[1], color[0]]
            hex_color = '#{:02x}{:02x}{:02x}'.format(*rgb_color)
            hex_colors.append(hex_color)
        
        return hex_colors
    
    def _predict_ripeness_stage(self, features: np.ndarray) -> Dict:
        """Predict ripeness stage using ML model"""
        # In a real implementation, would use the trained model
        # For now, use heuristic-based prediction
        
        # Normalize features
        features_scaled = self.scaler.fit_transform(features.reshape(1, -1))
        
        # Mock prediction based on feature patterns
        # This would be replaced with actual model inference
        probabilities = np.array([0.3, 0.5, 0.2])  # unripe, ripe, overripe
        predicted_class = np.argmax(probabilities)
        
        stages = ['unripe', 'ripe', 'overripe']
        
        return {
            'predicted_stage': stages[predicted_class],
            'confidence': float(np.max(probabilities)),
            'probabilities': {
                'unripe': float(probabilities[0]),
                'ripe': float(probabilities[1]),
                'overripe': float(probabilities[2])
            }
        }
    
    def _calculate_confidence_scores(self, color_analysis: Dict, 
                                  texture_analysis: Dict, 
                                  shape_analysis: Dict) -> Dict:
        """Calculate confidence scores for different analysis types"""
        scores = {}
        
        # Color confidence
        if 'color_scores' in color_analysis:
            max_color_score = max(score['score'] for score in color_analysis['color_scores'].values())
            scores['color'] = min(max_color_score * 2, 1.0)  # Scale to 0-1
        else:
            scores['color'] = 0.5
        
        # Texture confidence
        texture_confidence = 0.7  # Base confidence
        if texture_analysis.get('wrinkle_detected'):
            texture_confidence += 0.2
        if texture_analysis.get('soft_spot_ratio', 0) > 0.1:
            texture_confidence += 0.1
        scores['texture'] = min(texture_confidence, 1.0)
        
        # Shape confidence
        if 'deformations' in shape_analysis:
            shape_confidence = 0.8
            if shape_analysis['deformations']['deformed']:
                shape_confidence = 0.6
            scores['shape'] = shape_confidence
        else:
            scores['shape'] = 0.5
        
        # Overall confidence
        scores['overall'] = np.mean(list(scores.values()))
        
        return scores
    
    def _predict_timing(self, current_stage: str, profile: Dict) -> Dict:
        """Predict timing for next ripeness stages"""
        if not profile:
            return {'error': 'No ripeness profile available'}
        
        timing = {}
        
        if current_stage == 'unripe':
            timing['days_to_ripe'] = profile.get('unripe_days_to_ripe', '3-5 days')
            timing['days_to_overripe'] = profile.get('unripe_days_to_ripe', '3-5 days') + profile.get('ripe_days_to_overripe', '2-3 days')
        elif current_stage == 'ripe':
            timing['days_to_overripe'] = profile.get('ripe_days_to_overripe', '2-3 days')
            timing['best_consumption_window'] = 'Next 2-3 days'
        else:  # overripe
            timing['days_to_spoiled'] = '1-2 days'
            timing['urgent_action'] = 'Use immediately or freeze'
        
        return timing
    
    def _generate_ripeness_recommendations(self, stage: str, produce_type: str, profile: Dict) -> List[str]:
        """Generate recommendations based on ripeness stage"""
        recommendations = []
        
        if stage == 'unripe':
            recommendations.append(f"Your {produce_type} is not yet ripe")
            recommendations.append("Leave at room temperature to ripen")
            if produce_type.lower() == 'avocado':
                recommendations.append("Place in a paper bag with an apple to speed ripening")
        elif stage == 'ripe':
            recommendations.append(f"Your {produce_type} is perfectly ripe!")
            recommendations.append("This is the best time to enjoy it fresh")
            recommendations.append("Store properly to maintain quality")
        else:  # overripe
            recommendations.append(f"Your {produce_type} is overripe")
            recommendations.append("Best used in cooking, baking, or smoothies")
            recommendations.append("Consider freezing for later use")
        
        # Add specific uses from profile
        optimal_uses = profile.get('optimal_uses', {}).get(stage, [])
        if optimal_uses:
            recommendations.append(f"Best uses: {', '.join(optimal_uses)}")
        
        return recommendations
