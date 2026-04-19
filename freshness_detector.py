"""
Freshness Detector Module
Detects freshness, spoilage, and mold in food items using computer vision
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

class FreshnessDetector:
    """Detects food freshness, spoilage, and mold patterns"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize freshness detector
        
        Args:
            model_path: Path to pre-trained model
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.scaler = StandardScaler()
        self.mold_detector = self._init_mold_detector()
        
        # Freshness thresholds for different food types
        self.freshness_thresholds = {
            'fruits': {
                'color_deviation': 30,
                'texture_roughness': 0.3,
                'mold_confidence': 0.7
            },
            'vegetables': {
                'color_deviation': 25,
                'texture_roughness': 0.25,
                'mold_confidence': 0.6
            },
            'meat': {
                'color_deviation': 40,
                'texture_roughness': 0.4,
                'mold_confidence': 0.8
            },
            'dairy': {
                'color_deviation': 35,
                'texture_roughness': 0.35,
                'mold_confidence': 0.75
            }
        }
    
    def _load_model(self, model_path: Optional[str]) -> nn.Module:
        """Load or create freshness detection model"""
        if model_path and Path(model_path).exists():
            model = torch.load(model_path, map_location=self.device)
            return model
        
        # Create a simple CNN for freshness detection
        model = nn.Sequential(
            # Convolutional layers
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Flatten and dense layers
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3)  # fresh, spoiling, spoiled
        )
        
        return model.to(self.device)
    
    def _init_mold_detector(self) -> RandomForestClassifier:
        """Initialize mold detection classifier"""
        return RandomForestClassifier(n_estimators=100, random_state=42)
    
    def detect_freshness(self, image: np.ndarray, food_type: str) -> Dict:
        """
        Detect freshness level of food
        
        Args:
            image: Food image
            food_type: Type of food (fruit, vegetable, meat, dairy)
            
        Returns:
            Freshness analysis results
        """
        # Extract features for freshness analysis
        features = self._extract_freshness_features(image)
        
        # Analyze color changes
        color_analysis = self._analyze_color_changes(image, food_type)
        
        # Analyze texture changes
        texture_analysis = self._analyze_texture_changes(image)
        
        # Detect mold
        mold_analysis = self._detect_mold(image)
        
        # Predict freshness using ML model
        freshness_prediction = self._predict_freshness(features)
        
        # Combine all analyses
        freshness_score = self._calculate_freshness_score(
            color_analysis, texture_analysis, mold_analysis, food_type
        )
        
        return {
            'freshness_level': self._classify_freshness(freshness_score),
            'freshness_score': freshness_score,
            'color_analysis': color_analysis,
            'texture_analysis': texture_analysis,
            'mold_analysis': mold_analysis,
            'prediction': freshness_prediction,
            'recommendations': self._generate_freshness_recommendations(freshness_score, food_type)
        }
    
    def _extract_freshness_features(self, image: np.ndarray) -> np.ndarray:
        """Extract features for freshness detection"""
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate color statistics
        color_features = [
            np.mean(image, axis=(0, 1)),  # RGB means
            np.std(image, axis=(0, 1)),   # RGB stds
            np.mean(hsv, axis=(0, 1)),    # HSV means
            np.std(hsv, axis=(0, 1)),     # HSV stds
            np.mean(lab, axis=(0, 1)),    # LAB means
            np.std(lab, axis=(0, 1))      # LAB stds
        ]
        
        # Calculate texture features
        texture_features = self._calculate_texture_features(gray)
        
        # Calculate edge features
        edge_features = self._calculate_edge_features(gray)
        
        # Combine all features
        features = np.concatenate([
            np.array(color_features).flatten(),
            texture_features,
            edge_features
        ])
        
        return features
    
    def _analyze_color_changes(self, image: np.ndarray, food_type: str) -> Dict:
        """Analyze color changes indicative of spoilage"""
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Analyze hue distribution
        hue_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        hue_hist = hue_hist.flatten() / hue_hist.sum()
        
        # Analyze saturation (loss of color indicates spoilage)
        saturation = hsv[:, :, 1]
        mean_saturation = np.mean(saturation)
        std_saturation = np.std(saturation)
        
        # Analyze value (brightness changes)
        value = hsv[:, :, 2]
        mean_value = np.mean(value)
        std_value = np.std(value)
        
        # Detect brown/gray spots (common spoilage indicators)
        brown_mask = self._detect_brown_spots(image)
        gray_mask = self._detect_gray_spots(image)
        
        # Calculate color uniformity
        color_uniformity = self._calculate_color_uniformity(hsv)
        
        return {
            'mean_saturation': float(mean_saturation),
            'std_saturation': float(std_saturation),
            'mean_value': float(mean_value),
            'std_value': float(std_value),
            'brown_spot_ratio': float(np.sum(brown_mask) / brown_mask.size),
            'gray_spot_ratio': float(np.sum(gray_mask) / gray_mask.size),
            'color_uniformity': float(color_uniformity),
            'hue_distribution': hue_hist.tolist()
        }
    
    def _detect_brown_spots(self, image: np.ndarray) -> np.ndarray:
        """Detect brown spots indicative of spoilage"""
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define brown color range in HSV
        lower_brown = np.array([8, 50, 50])
        upper_brown = np.array([25, 255, 255])
        
        # Create mask for brown colors
        brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
        
        # Apply morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        brown_mask = cv2.morphologyEx(brown_mask, cv2.MORPH_CLOSE, kernel)
        brown_mask = cv2.morphologyEx(brown_mask, cv2.MORPH_OPEN, kernel)
        
        return brown_mask > 0
    
    def _detect_gray_spots(self, image: np.ndarray) -> np.ndarray:
        """Detect gray/white spots (mold)"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to detect light spots
        _, gray_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations
        kernel = np.ones((5, 5), np.uint8)
        gray_mask = cv2.morphologyEx(gray_mask, cv2.MORPH_CLOSE, kernel)
        gray_mask = cv2.morphologyEx(gray_mask, cv2.MORPH_OPEN, kernel)
        
        return gray_mask > 0
    
    def _calculate_color_uniformity(self, hsv: np.ndarray) -> float:
        """Calculate color uniformity score"""
        # Calculate standard deviation across color channels
        hue_std = np.std(hsv[:, :, 0])
        sat_std = np.std(hsv[:, :, 1])
        val_std = np.std(hsv[:, :, 2])
        
        # Lower std indicates more uniform color (fresher)
        uniformity = 1.0 / (1.0 + (hue_std + sat_std + val_std) / 3.0)
        
        return uniformity
    
    def _analyze_texture_changes(self, image: np.ndarray) -> Dict:
        """Analyze texture changes indicative of spoilage"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate Local Binary Pattern
        lbp = self._calculate_lbp(gray)
        
        # Calculate texture roughness
        roughness = self._calculate_roughness(gray)
        
        # Calculate surface smoothness
        smoothness = self._calculate_smoothness(gray)
        
        # Detect slime/slippery texture (common in spoiled food)
        slime_texture = self._detect_slime_texture(gray)
        
        return {
            'lbp_mean': float(np.mean(lbp)),
            'lbp_std': float(np.std(lbp)),
            'roughness': float(roughness),
            'smoothness': float(smoothness),
            'slime_detected': bool(slime_texture),
            'texture_variance': float(np.var(gray))
        }
    
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
    
    def _calculate_roughness(self, image: np.ndarray) -> float:
        """Calculate surface roughness"""
        # Calculate gradient magnitude
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        return np.mean(gradient_magnitude)
    
    def _calculate_smoothness(self, image: np.ndarray) -> float:
        """Calculate surface smoothness"""
        # Smoothness is inverse of roughness
        roughness = self._calculate_roughness(image)
        return 1.0 / (1.0 + roughness)
    
    def _detect_slime_texture(self, image: np.ndarray) -> bool:
        """Detect slime/slippery texture patterns"""
        # Slime often appears as glossy, reflective areas
        # This is a simplified detection - in practice would use more sophisticated methods
        
        # Calculate local variance (slime areas have low variance)
        kernel = np.ones((5, 5), np.float32) / 25
        local_mean = cv2.filter2D(image.astype(np.float32), -1, kernel)
        local_var = cv2.filter2D((image.astype(np.float32) - local_mean) ** 2, -1, kernel)
        
        # Areas with very low variance might indicate slime
        low_var_areas = local_var < np.percentile(local_var, 10)
        
        # If more than 5% of image has low variance, flag as potential slime
        slime_ratio = np.sum(low_var_areas) / low_var_areas.size
        
        return slime_ratio > 0.05
    
    def _detect_mold(self, image: np.ndarray) -> Dict:
        """Detect mold patterns in food"""
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Detect common mold colors (white, green, blue, black)
        mold_masks = {}
        
        # White/gray mold
        white_lower = np.array([0, 0, 200])
        white_upper = np.array([180, 30, 255])
        mold_masks['white'] = cv2.inRange(hsv, white_lower, white_upper)
        
        # Green mold
        green_lower = np.array([35, 40, 40])
        green_upper = np.array([85, 255, 255])
        mold_masks['green'] = cv2.inRange(hsv, green_lower, green_upper)
        
        # Blue mold
        blue_lower = np.array([100, 40, 40])
        blue_upper = np.array([130, 255, 255])
        mold_masks['blue'] = cv2.inRange(hsv, blue_lower, blue_upper)
        
        # Black mold
        black_lower = np.array([0, 0, 0])
        black_upper = np.array([180, 255, 50])
        mold_masks['black'] = cv2.inRange(hsv, black_lower, black_upper)
        
        # Analyze mold patterns
        mold_analysis = {}
        total_mold_area = 0
        
        for mold_type, mask in mold_masks.items():
            # Clean up mask
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Calculate mold area
            mold_area = np.sum(mask > 0)
            mold_ratio = mold_area / mask.size
            
            # Detect mold texture (fuzzy patterns)
            mold_texture = self._analyze_mold_texture(mask)
            
            mold_analysis[mold_type] = {
                'area_ratio': float(mold_ratio),
                'detected': bool(mold_ratio > 0.01),  # 1% threshold
                'texture_score': float(mold_texture)
            }
            
            total_mold_area += mold_area
        
        # Overall mold assessment
        total_mold_ratio = total_mold_area / (image.shape[0] * image.shape[1])
        
        return {
            'overall_mold_ratio': float(total_mold_ratio),
            'mold_detected': bool(total_mold_ratio > 0.02),  # 2% threshold
            'mold_types': mold_analysis,
            'risk_level': self._assess_mold_risk(total_mold_ratio)
        }
    
    def _analyze_mold_texture(self, mask: np.ndarray) -> float:
        """Analyze texture patterns in mold areas"""
        # Convert mask to float
        mask_float = mask.astype(np.float32) / 255.0
        
        # Calculate texture complexity using edge detection
        edges = cv2.Canny(mask.astype(np.uint8), 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        return edge_density
    
    def _assess_mold_risk(self, mold_ratio: float) -> str:
        """Assess mold risk level"""
        if mold_ratio < 0.01:
            return "low"
        elif mold_ratio < 0.05:
            return "moderate"
        else:
            return "high"
    
    def _calculate_texture_features(self, gray: np.ndarray) -> np.ndarray:
        """Calculate texture features for freshness detection"""
        # Calculate LBP histogram
        lbp = self._calculate_lbp(gray)
        lbp_hist = np.histogram(lbp, bins=256)[0]
        lbp_hist = lbp_hist / lbp_hist.sum()
        
        # Calculate GLCM features (simplified)
        glcm_features = self._calculate_glcm_features(gray)
        
        return np.concatenate([lbp_hist, glcm_features])
    
    def _calculate_glcm_features(self, image: np.ndarray) -> np.ndarray:
        """Calculate Gray-Level Co-occurrence Matrix features"""
        # Simplified GLCM calculation
        # In practice, would use skimage's greycomatrix
        
        # Calculate contrast, correlation, energy, homogeneity
        contrast = np.var(image)
        correlation = np.corrcoef(image.flatten()[:-1], image.flatten()[1:])[0, 1]
        energy = np.sum(image**2) / (image.shape[0] * image.shape[1])
        homogeneity = 1.0 / (1.0 + contrast)
        
        return np.array([contrast, correlation, energy, homogeneity])
    
    def _calculate_edge_features(self, gray: np.ndarray) -> np.ndarray:
        """Calculate edge features for freshness detection"""
        # Calculate edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate edge density
        edge_density = np.sum(edges > 0) / edges.size
        
        # Calculate edge orientation histogram
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        orientations = np.arctan2(grad_y, grad_x)
        
        # Bin orientations
        orientation_hist = np.histogram(orientations.flatten(), bins=8, range=[-np.pi, np.pi])[0]
        orientation_hist = orientation_hist / orientation_hist.sum()
        
        return np.array([edge_density, *orientation_hist])
    
    def _predict_freshness(self, features: np.ndarray) -> Dict:
        """Predict freshness using ML model"""
        # In a real implementation, would use the trained model
        # For now, return placeholder values
        
        # Normalize features
        features_scaled = self.scaler.fit_transform(features.reshape(1, -1))
        
        # Mock prediction (would use actual model)
        probabilities = np.array([0.7, 0.2, 0.1])  # fresh, spoiling, spoiled
        predicted_class = np.argmax(probabilities)
        
        class_names = ['fresh', 'spoiling', 'spoiled']
        
        return {
            'predicted_class': class_names[predicted_class],
            'probabilities': {
                'fresh': float(probabilities[0]),
                'spoiling': float(probabilities[1]),
                'spoiled': float(probabilities[2])
            }
        }
    
    def _calculate_freshness_score(self, color_analysis: Dict, 
                                 texture_analysis: Dict, 
                                 mold_analysis: Dict, 
                                 food_type: str) -> float:
        """Calculate overall freshness score (0-1, higher is fresher)"""
        # Get thresholds for food type
        thresholds = self.freshness_thresholds.get(food_type, self.freshness_thresholds['fruits'])
        
        # Color score (higher saturation and uniformity = fresher)
        color_score = (color_analysis['mean_saturation'] / 255.0) * 0.5 + \
                     color_analysis['color_uniformity'] * 0.5
        
        # Texture score (moderate roughness = fresh, too smooth or too rough = bad)
        optimal_roughness = 0.2
        texture_score = 1.0 - abs(texture_analysis['roughness'] - optimal_roughness)
        
        # Mold score (no mold = fresh)
        mold_score = 1.0 - mold_analysis['overall_mold_ratio'] * 10  # Penalize mold heavily
        mold_score = max(0, mold_score)
        
        # Combine scores
        freshness_score = (color_score * 0.4 + texture_score * 0.3 + mold_score * 0.3)
        
        return max(0, min(1, freshness_score))
    
    def _classify_freshness(self, score: float) -> str:
        """Classify freshness level based on score"""
        if score > 0.8:
            return "fresh"
        elif score > 0.5:
            return "good"
        elif score > 0.3:
            return "spoiling"
        else:
            return "spoiled"
    
    def _generate_freshness_recommendations(self, score: float, food_type: str) -> List[str]:
        """Generate recommendations based on freshness"""
        recommendations = []
        
        if score > 0.8:
            recommendations.append("Food is fresh and safe to consume")
            recommendations.append("Store properly to maintain freshness")
        elif score > 0.5:
            recommendations.append("Food is still good but consume soon")
            recommendations.append("Check for any unusual odors before consuming")
        elif score > 0.3:
            recommendations.append("Food is starting to spoil")
            recommendations.append("Consider cooking thoroughly if still usable")
            recommendations.append("Discard if any mold is visible")
        else:
            recommendations.append("Food appears spoiled - do not consume")
            recommendations.append("Discard immediately to prevent illness")
        
        # Add food-specific recommendations
        if food_type == 'meat' and score < 0.7:
            recommendations.append("Meat should be discarded if not fresh")
        elif food_type == 'dairy' and score < 0.6:
            recommendations.append("Dairy products should be discarded if spoiling")
        
        return recommendations
