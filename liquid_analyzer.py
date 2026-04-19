"""
Liquid Properties Analyzer Module
Analyzes liquid foods and drinks for viscosity, transparency, cohesion, and other properties
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

class LiquidAnalyzer:
    """Analyzes liquid properties of food and drinks"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize liquid analyzer
        
        Args:
            model_path: Path to pre-trained model
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.scaler = StandardScaler()
        
        # Liquid property ranges for common liquids
        self.liquid_properties = {
            'water': {
                'viscosity': 1.0,  # mPa.s at 20C
                'transparency': 1.0,
                'cohesion': 0.5,
                'adhesion': 0.3,
                'density': 1.0  # g/cm3
            },
            'milk': {
                'viscosity': 2.0,
                'transparency': 0.7,
                'cohesion': 0.6,
                'adhesion': 0.4,
                'density': 1.03
            },
            'juice': {
                'viscosity': 1.5,
                'transparency': 0.6,
                'cohesion': 0.5,
                'adhesion': 0.3,
                'density': 1.05
            },
            'oil': {
                'viscosity': 50.0,
                'transparency': 0.8,
                'cohesion': 0.7,
                'adhesion': 0.2,
                'density': 0.92
            },
            'syrup': {
                'viscosity': 1000.0,
                'transparency': 0.3,
                'cohesion': 0.8,
                'adhesion': 0.6,
                'density': 1.3
            },
            'soup': {
                'viscosity': 5.0,
                'transparency': 0.4,
                'cohesion': 0.6,
                'adhesion': 0.5,
                'density': 1.1
            },
            'coffee': {
                'viscosity': 1.2,
                'transparency': 0.8,
                'cohesion': 0.5,
                'adhesion': 0.3,
                'density': 1.01
            },
            'tea': {
                'viscosity': 1.1,
                'transparency': 0.9,
                'cohesion': 0.5,
                'adhesion': 0.3,
                'density': 1.0
            }
        }
        
        # Viscosity categories
        self.viscosity_categories = {
            'very_low': {'range': (0.5, 2.0), 'examples': ['water', 'tea', 'coffee']},
            'low': {'range': (2.0, 5.0), 'examples': ['milk', 'juice', 'thin soups']},
            'medium': {'range': (5.0, 20.0), 'examples': ['thick soups', 'smoothies']},
            'high': {'range': (20.0, 100.0), 'examples': ['oils', 'sauces']},
            'very_high': {'range': (100.0, 10000.0), 'examples': ['syrups', 'honey']}
        }
    
    def _load_model(self, model_path: Optional[str]) -> nn.Module:
        """Load or create liquid properties model"""
        if model_path and Path(model_path).exists():
            model = torch.load(model_path, map_location=self.device)
            return model
        
        # Create CNN for liquid property prediction
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
            nn.Linear(64, 5)  # viscosity, transparency, cohesion, adhesion, density
        )
        
        return model.to(self.device)
    
    def analyze_liquid_properties(self, image: np.ndarray, liquid_type: Optional[str] = None) -> Dict:
        """
        Analyze liquid properties from image
        
        Args:
            image: Liquid image
            liquid_type: Type of liquid (optional)
            
        Returns:
            Comprehensive liquid properties analysis
        """
        # Detect liquid region
        liquid_region = self._detect_liquid_region(image)
        
        # Analyze viscosity
        viscosity_analysis = self._analyze_viscosity(image, liquid_region)
        
        # Analyze transparency
        transparency_analysis = self._analyze_transparency(image, liquid_region)
        
        # Analyze cohesion vs adhesion
        cohesion_analysis = self._analyze_cohesion_adhesion(image, liquid_region)
        
        # Analyze flow properties
        flow_analysis = self._analyze_flow_properties(image, liquid_region)
        
        # Analyze surface tension indicators
        surface_analysis = self._analyze_surface_properties(image, liquid_region)
        
        # Predict properties using ML model
        ml_prediction = self._predict_properties_ml(image, liquid_region)
        
        # Combine all analyses
        combined_properties = self._combine_property_analyses(
            viscosity_analysis, transparency_analysis, cohesion_analysis,
            flow_analysis, surface_analysis, ml_prediction
        )
        
        # Classify liquid type
        liquid_classification = self._classify_liquid_type(combined_properties)
        
        # Generate recommendations
        recommendations = self._generate_liquid_recommendations(
            combined_properties, liquid_classification
        )
        
        return {
            'liquid_type': liquid_type,
            'liquid_region': liquid_region,
            'viscosity_analysis': viscosity_analysis,
            'transparency_analysis': transparency_analysis,
            'cohesion_analysis': cohesion_analysis,
            'flow_analysis': flow_analysis,
            'surface_analysis': surface_analysis,
            'ml_prediction': ml_prediction,
            'combined_properties': combined_properties,
            'liquid_classification': liquid_classification,
            'recommendations': recommendations
        }
    
    def _detect_liquid_region(self, image: np.ndarray) -> Dict:
        """Detect liquid region in image"""
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use color and texture to detect liquid
        # Liquids often have smooth textures and specific color ranges
        
        # Texture analysis - liquids have low texture
        kernel = np.ones((5, 5), np.float32) / 25
        local_var = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        smooth_areas = local_var < np.percentile(local_var, 30)
        
        # Color analysis for common liquid colors
        # Clear liquids: high value, low saturation in some cases
        # Dark liquids: low value
        # Colored liquids: various hues
        
        # Create masks for different liquid types
        clear_mask = (hsv[:, :, 2] > 200) & (hsv[:, :, 1] < 100)
        dark_mask = hsv[:, :, 2] < 100
        colored_mask = (hsv[:, :, 1] > 50) & (hsv[:, :, 2] > 50)
        
        # Combine masks
        liquid_mask = (clear_mask | dark_mask | colored_mask) & smooth_areas
        
        # Clean up mask
        kernel = np.ones((3, 3), np.uint8)
        liquid_mask = cv2.morphologyEx(liquid_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        liquid_mask = cv2.morphologyEx(liquid_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(liquid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(main_contour)
            area = cv2.contourArea(main_contour)
            
            return {
                'detected': True,
                'bounding_box': [x, y, w, h],
                'area': area,
                'mask': liquid_mask,
                'contour': main_contour
            }
        else:
            return {'detected': False, 'mask': liquid_mask}
    
    def _analyze_viscosity(self, image: np.ndarray, liquid_region: Dict) -> Dict:
        """Analyze liquid viscosity"""
        if not liquid_region.get('detected'):
            return {'error': 'No liquid region detected'}
        
        # Extract liquid region
        bbox = liquid_region['bounding_box']
        x, y, w, h = bbox
        liquid_image = image[y:y+h, x:x+w]
        
        # Convert to grayscale
        gray = cv2.cvtColor(liquid_image, cv2.COLOR_BGR2GRAY)
        
        # Viscosity indicators:
        # 1. Bubble formation (low viscosity = more bubbles)
        # 2. Edge clarity (high viscosity = clearer edges)
        # 3. Texture uniformity (high viscosity = more uniform)
        
        # Detect bubbles
        bubble_analysis = self._detect_bubbles(gray)
        
        # Analyze edge clarity
        edge_analysis = self._analyze_edge_clarity(gray)
        
        # Analyze texture uniformity
        texture_uniformity = self._calculate_texture_uniformity(gray)
        
        # Estimate viscosity based on indicators
        viscosity_score = self._estimate_viscosity_score(
            bubble_analysis, edge_analysis, texture_uniformity
        )
        
        # Convert to actual viscosity value
        viscosity_value = self._viscosity_score_to_value(viscosity_score)
        
        return {
            'viscosity_score': viscosity_score,
            'estimated_viscosity': viscosity_value,
            'viscosity_category': self._classify_viscosity_category(viscosity_value),
            'bubble_analysis': bubble_analysis,
            'edge_analysis': edge_analysis,
            'texture_uniformity': texture_uniformity
        }
    
    def _detect_bubbles(self, gray: np.ndarray) -> Dict:
        """Detect bubbles in liquid"""
        # Bubbles appear as circular dark areas
        
        # Use blob detection for circular shapes
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 5
        params.maxArea = 100
        params.filterByCircularity = True
        params.minCircularity = 0.5
        params.filterByConvexity = True
        params.minConvexity = 0.5
        
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(gray)
        
        # Count bubbles
        bubble_count = len(keypoints)
        
        # Calculate bubble density
        image_area = gray.shape[0] * gray.shape[1]
        bubble_density = bubble_count / (image_area / 1000)  # bubbles per 1000 pixels
        
        return {
            'bubble_count': bubble_count,
            'bubble_density': bubble_density,
            'keypoints': keypoints
        }
    
    def _analyze_edge_clarity(self, gray: np.ndarray) -> Dict:
        """Analyze edge clarity (inverse of viscosity)"""
        # High viscosity liquids have clearer edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate edge density
        edge_density = np.sum(edges > 0) / edges.size
        
        # Calculate edge strength
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        edge_strength = np.mean(gradient_magnitude)
        
        return {
            'edge_density': edge_density,
            'edge_strength': edge_strength,
            'clarity_score': (edge_density + edge_strength / 255.0) / 2
        }
    
    def _calculate_texture_uniformity(self, gray: np.ndarray) -> Dict:
        """Calculate texture uniformity"""
        # High viscosity liquids are more uniform
        
        # Calculate local variance
        kernel = np.ones((7, 7), np.float32) / 49
        local_var = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        
        uniformity = 1.0 / (1.0 + np.std(local_var))
        
        return {
            'uniformity': uniformity,
            'local_variance': np.std(local_var)
        }
    
    def _estimate_viscosity_score(self, bubble: Dict, edge: Dict, texture: Dict) -> float:
        """Estimate viscosity score (0-1, higher = more viscous)"""
        # Low viscosity: more bubbles, less edge clarity, less uniform
        # High viscosity: fewer bubbles, more edge clarity, more uniform
        
        bubble_score = 1.0 - min(bubble['bubble_density'] / 10.0, 1.0)
        edge_score = edge['clarity_score']
        texture_score = texture['uniformity']
        
        # Weighted average
        viscosity_score = (bubble_score * 0.4 + edge_score * 0.3 + texture_score * 0.3)
        
        return viscosity_score
    
    def _viscosity_score_to_value(self, score: float) -> float:
        """Convert viscosity score to actual viscosity value in mPa.s"""
        # Use logarithmic scale
        if score < 0.2:
            return 0.5 + score * 7.5  # 0.5-2.0
        elif score < 0.4:
            return 2.0 + (score - 0.2) * 15.0  # 2.0-5.0
        elif score < 0.6:
            return 5.0 + (score - 0.4) * 75.0  # 5.0-20.0
        elif score < 0.8:
            return 20.0 + (score - 0.6) * 400.0  # 20.0-100.0
        else:
            return 100.0 + (score - 0.8) * 2475.0  # 100.0-10000.0
    
    def _classify_viscosity_category(self, viscosity: float) -> str:
        """Classify viscosity into categories"""
        for category, info in self.viscosity_categories.items():
            min_visc, max_visc = info['range']
            if min_visc <= viscosity <= max_visc:
                return category
        return 'unknown'
    
    def _analyze_transparency(self, image: np.ndarray, liquid_region: Dict) -> Dict:
        """Analyze liquid transparency"""
        if not liquid_region.get('detected'):
            return {'error': 'No liquid region detected'}
        
        # Extract liquid region
        bbox = liquid_region['bounding_box']
        x, y, w, h = bbox
        liquid_image = image[y:y+h, x:x+w]
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(liquid_image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(liquid_image, cv2.COLOR_BGR2GRAY)
        
        # Transparency indicators:
        # 1. Light transmission (high value in HSV)
        # 2. Color saturation (low saturation = more transparent)
        # 3. Texture clarity (clear liquids have less texture)
        
        # Light transmission
        light_transmission = np.mean(hsv[:, :, 2]) / 255.0
        
        # Saturation analysis
        saturation = np.mean(hsv[:, :, 1]) / 255.0
        transparency_from_sat = 1.0 - saturation
        
        # Texture clarity
        texture_clarity = self._calculate_texture_clarity(gray)
        
        # Background visibility (if container is visible)
        background_visibility = self._detect_background_visibility(gray)
        
        # Combine indicators
        transparency_score = (light_transmission * 0.3 + 
                          transparency_from_sat * 0.3 + 
                          texture_clarity * 0.2 + 
                          background_visibility * 0.2)
        
        return {
            'transparency_score': transparency_score,
            'light_transmission': light_transmission,
            'saturation': saturation,
            'texture_clarity': texture_clarity,
            'background_visibility': background_visibility,
            'transparency_category': self._classify_transparency_category(transparency_score)
        }
    
    def _calculate_texture_clarity(self, gray: np.ndarray) -> float:
        """Calculate texture clarity (inverse of texture complexity)"""
        # Clear liquids have less texture
        local_var = cv2.filter2D(gray.astype(np.float32), -1, 
                               np.ones((5, 5), np.float32) / 25)
        clarity = 1.0 / (1.0 + np.std(local_var))
        return clarity
    
    def _detect_background_visibility(self, gray: np.ndarray) -> float:
        """Detect if background is visible through liquid"""
        # This is a simplified implementation
        # In practice, would analyze patterns that suggest background visibility
        
        # Look for patterns that might be background
        edges = cv2.Canny(gray, 30, 100)
        
        # Calculate pattern complexity
        histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])
        histogram = histogram.flatten()
        histogram = histogram / histogram.sum()
        
        entropy = -np.sum(histogram[histogram > 0] * np.log2(histogram[histogram > 0]))
        
        # Higher entropy might indicate visible background
        visibility = min(entropy / 8.0, 1.0)
        
        return visibility
    
    def _classify_transparency_category(self, score: float) -> str:
        """Classify transparency into categories"""
        if score > 0.8:
            return 'transparent'
        elif score > 0.6:
            return 'translucent'
        elif score > 0.3:
            return 'opaque'
        else:
            return 'very_opaque'
    
    def _analyze_cohesion_adhesion(self, image: np.ndarray, liquid_region: Dict) -> Dict:
        """Analyze cohesion vs adhesion properties"""
        if not liquid_region.get('detected'):
            return {'error': 'No liquid region detected'}
        
        # Extract liquid region
        bbox = liquid_region['bounding_box']
        x, y, w, h = bbox
        liquid_image = image[y:y+h, x:x+w]
        
        # Cohesion: tendency of liquid molecules to stick together
        # Adhesion: tendency to stick to other surfaces
        
        # Cohesion indicators:
        # 1. Surface tension (droplet formation)
        # 2. Internal uniformity
        
        # Adhesion indicators:
        # 1. Meniscus formation at edges
        # 2. Wetting behavior
        
        # Analyze surface tension indicators
        surface_tension = self._analyze_surface_tension(liquid_image)
        
        # Analyze internal uniformity (cohesion)
        internal_uniformity = self._analyze_internal_uniformity(liquid_image)
        
        # Analyze edge wetting (adhesion)
        edge_wetting = self._analyze_edge_wetting(liquid_image)
        
        # Calculate cohesion and adhesion scores
        cohesion_score = (surface_tension * 0.6 + internal_uniformity * 0.4)
        adhesion_score = edge_wetting
        
        return {
            'cohesion_score': cohesion_score,
            'adhesion_score': adhesion_score,
            'surface_tension': surface_tension,
            'internal_uniformity': internal_uniformity,
            'edge_wetting': edge_wetting,
            'cohesion_adhesion_ratio': cohesion_score / (adhesion_score + 0.001)
        }
    
    def _analyze_surface_tension(self, liquid_image: np.ndarray) -> float:
        """Analyze surface tension indicators"""
        gray = cv2.cvtColor(liquid_image, cv2.COLOR_BGR2GRAY)
        
        # Look for droplet formation
        # High surface tension = more rounded droplets
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyze circularity of contours
        circularities = []
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                circularities.append(circularity)
        
        if circularities:
            avg_circularity = np.mean(circularities)
            return min(avg_circularity, 1.0)
        else:
            return 0.5  # Default
    
    def _analyze_internal_uniformity(self, liquid_image: np.ndarray) -> float:
        """Analyze internal uniformity (cohesion)"""
        gray = cv2.cvtColor(liquid_image, cv2.COLOR_BGR2GRAY)
        
        # Calculate local variance
        kernel = np.ones((7, 7), np.float32) / 49
        local_var = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        
        # Lower variance = higher uniformity = higher cohesion
        uniformity = 1.0 / (1.0 + np.std(local_var))
        
        return uniformity
    
    def _analyze_edge_wetting(self, liquid_image: np.ndarray) -> float:
        """Analyze edge wetting behavior (adhesion)"""
        gray = cv2.cvtColor(liquid_image, cv2.COLOR_BGR2GRAY)
        
        # Analyze edges of liquid region
        edges = cv2.Canny(gray, 30, 100)
        
        # Look for meniscus-like patterns at edges
        height, width = gray.shape
        
        # Check top and bottom edges
        top_edge = edges[0:5, :]
        bottom_edge = edges[-5:, :]
        left_edge = edges[:, 0:5]
        right_edge = edges[:, -5:]
        
        # Calculate edge density at borders
        edge_density = (np.sum(top_edge) + np.sum(bottom_edge) + 
                       np.sum(left_edge) + np.sum(right_edge)) / (4 * 5 * max(width, height))
        
        return min(edge_density / 255.0, 1.0)
    
    def _analyze_flow_properties(self, image: np.ndarray, liquid_region: Dict) -> Dict:
        """Analyze flow properties"""
        if not liquid_region.get('detected'):
            return {'error': 'No liquid region detected'}
        
        # Flow properties indicators:
        # 1. Streamlines (if visible)
        # 2. Vortices/swirls
        # 3. Settling particles
        
        # Extract liquid region
        bbox = liquid_region['bounding_box']
        x, y, w, h = bbox
        liquid_image = image[y:y+h, x:x+w]
        
        gray = cv2.cvtColor(liquid_image, cv2.COLOR_BGR2GRAY)
        
        # Detect flow patterns
        flow_patterns = self._detect_flow_patterns(gray)
        
        # Detect particles (settling indicates viscosity)
        particle_analysis = self._detect_particles(gray)
        
        # Analyze movement indicators (if video available, would use optical flow)
        movement_indicators = self._analyze_movement_indicators(gray)
        
        return {
            'flow_patterns': flow_patterns,
            'particle_analysis': particle_analysis,
            'movement_indicators': movement_indicators,
            'flow_score': (flow_patterns['score'] + particle_analysis['settling_score']) / 2
        }
    
    def _detect_flow_patterns(self, gray: np.ndarray) -> Dict:
        """Detect flow patterns in liquid"""
        # Look for linear patterns (streamlines)
        # and circular patterns (vortices)
        
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect lines (streamlines)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=20, maxLineGap=10)
        
        # Detect circles (vortices)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, 
                                 param1=50, param2=30, minRadius=5, maxRadius=50)
        
        line_count = len(lines) if lines is not None else 0
        circle_count = len(circles[0]) if circles is not None else 0
        
        # Calculate flow score
        flow_score = min((line_count + circle_count * 2) / 10.0, 1.0)
        
        return {
            'streamlines': line_count,
            'vortices': circle_count,
            'score': flow_score
        }
    
    def _detect_particles(self, gray: np.ndarray) -> Dict:
        """Detect particles in liquid"""
        # Particles appear as small dark spots
        
        # Use adaptive threshold to find particles
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # Remove noise
        kernel = np.ones((2, 2), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours (particles)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter by size
        particles = [c for c in contours if 5 < cv2.contourArea(c) < 100]
        
        particle_count = len(particles)
        
        # Analyze settling (particles at bottom)
        if particles:
            bottom_particles = 0
            height = gray.shape[0]
            for contour in particles:
                y, _, _, h = cv2.boundingRect(contour)
                if y + h > height * 0.8:  # In bottom 20%
                    bottom_particles += 1
            
            settling_ratio = bottom_particles / particle_count if particle_count > 0 else 0
        else:
            settling_ratio = 0
        
        return {
            'particle_count': particle_count,
            'settling_ratio': settling_ratio,
            'settling_score': settling_ratio
        }
    
    def _analyze_movement_indicators(self, gray: np.ndarray) -> Dict:
        """Analyze movement indicators"""
        # This would be more accurate with video data
        # For static images, we can only infer from certain patterns
        
        # Look for motion blur indicators
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # High variance might indicate movement
        movement_score = min(laplacian_var / 1000.0, 1.0)
        
        return {
            'movement_score': movement_score,
            'laplacian_variance': laplacian_var
        }
    
    def _analyze_surface_properties(self, image: np.ndarray, liquid_region: Dict) -> Dict:
        """Analyze surface properties"""
        if not liquid_region.get('detected'):
            return {'error': 'No liquid region detected'}
        
        # Extract liquid region
        bbox = liquid_region['bounding_box']
        x, y, w, h = bbox
        liquid_image = image[y:y+h, x:x+w]
        
        # Surface properties:
        # 1. Surface tension (meniscus)
        # 2. Foam/bubbles on surface
        # 3. Surface sheen/gloss
        
        hsv = cv2.cvtColor(liquid_image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(liquid_image, cv2.COLOR_BGR2GRAY)
        
        # Detect surface foam
        foam_analysis = self._detect_surface_foam(gray)
        
        # Analyze surface sheen
        surface_sheen = self._analyze_surface_sheen(hsv)
        
        # Detect meniscus
        meniscus_analysis = self._detect_meniscus(gray)
        
        return {
            'foam_analysis': foam_analysis,
            'surface_sheen': surface_sheen,
            'meniscus_analysis': meniscus_analysis
        }
    
    def _detect_surface_foam(self, gray: np.ndarray) -> Dict:
        """Detect foam on liquid surface"""
        # Foam appears as light, textured areas at the top
        
        height, width = gray.shape
        top_region = gray[0:height//4, :]  # Top 25%
        
        # Look for light areas with texture
        _, thresh = cv2.threshold(top_region, 200, 255, cv2.THRESH_BINARY)
        
        # Calculate foam coverage
        foam_coverage = np.sum(thresh > 0) / thresh.size
        
        # Texture analysis of foam areas
        foam_areas = thresh > 0
        if np.any(foam_areas):
            foam_texture = np.std(top_region[foam_areas])
        else:
            foam_texture = 0
        
        return {
            'foam_coverage': foam_coverage,
            'foam_texture': foam_texture,
            'foam_detected': foam_coverage > 0.1
        }
    
    def _analyze_surface_sheen(self, hsv: np.ndarray) -> Dict:
        """Analyze surface sheen/gloss"""
        # Sheen appears as high value areas
        
        value_channel = hsv[:, :, 2]
        
        # Look for bright areas
        bright_mask = value_channel > np.percentile(value_channel, 90)
        sheen_coverage = np.sum(bright_mask) / bright_mask.size
        
        # Analyze saturation (low saturation can indicate gloss)
        saturation_channel = hsv[:, :, 1]
        avg_saturation = np.mean(saturation_channel[bright_mask]) if np.any(bright_mask) else 0
        
        return {
            'sheen_coverage': sheen_coverage,
            'avg_saturation': avg_saturation,
            'sheen_detected': sheen_coverage > 0.05
        }
    
    def _detect_meniscus(self, gray: np.ndarray) -> Dict:
        """Detect meniscus formation"""
        # Meniscus appears as curved edge at container boundary
        
        edges = cv2.Canny(gray, 50, 150)
        
        # Look at left and right edges
        height, width = gray.shape
        left_edge = edges[:, 0:10]
        right_edge = edges[:, -10:]
        
        # Look for curved patterns
        left_curve = self._detect_edge_curvature(left_edge)
        right_curve = self._detect_edge_curvature(right_edge)
        
        meniscus_detected = left_curve > 0.3 or right_curve > 0.3
        
        return {
            'left_curvature': left_curve,
            'right_curvature': right_curve,
            'meniscus_detected': meniscus_detected
        }
    
    def _detect_edge_curvature(self, edge_region: np.ndarray) -> float:
        """Detect curvature in edge region"""
        # Simplified curvature detection
        # Look for non-straight edge patterns
        
        if np.sum(edge_region) == 0:
            return 0.0
        
        # Find edge points
        edge_points = np.column_stack(np.where(edge_region > 0))
        
        if len(edge_points) < 10:
            return 0.0
        
        # Fit line and calculate deviation
        x = edge_points[:, 1]
        y = edge_points[:, 0]
        
        if len(x) > 1:
            # Linear fit
            coeffs = np.polyfit(x, y, 1)
            fitted_y = np.polyval(coeffs, x)
            
            # Calculate mean squared error
            mse = np.mean((y - fitted_y) ** 2)
            
            # Higher error = more curvature
            curvature = min(mse / 100.0, 1.0)
        else:
            curvature = 0.0
        
        return curvature
    
    def _predict_properties_ml(self, image: np.ndarray, liquid_region: Dict) -> Dict:
        """Predict liquid properties using ML model"""
        if not liquid_region.get('detected'):
            return {'error': 'No liquid region detected'}
        
        # Extract liquid region
        bbox = liquid_region['bounding_box']
        x, y, w, h = bbox
        liquid_image = image[y:y+h, x:x+w]
        
        if liquid_image.size == 0:
            return {'error': 'Empty liquid region'}
        
        # Preprocess for ML
        resized = cv2.resize(liquid_image, (224, 224))
        tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float()
        tensor = tensor / 255.0
        tensor = tensor.unsqueeze(0)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            properties_tensor = self.model(tensor.to(self.device))
            properties = properties_tensor.cpu().numpy()[0]
        
        return {
            'viscosity': float(properties[0]),
            'transparency': float(properties[1]),
            'cohesion': float(properties[2]),
            'adhesion': float(properties[3]),
            'density': float(properties[4])
        }
    
    def _combine_property_analyses(self, viscosity: Dict, transparency: Dict,
                                 cohesion: Dict, flow: Dict, surface: Dict,
                                 ml: Dict) -> Dict:
        """Combine all property analyses"""
        combined = {}
        
        # Combine viscosity
        if 'estimated_viscosity' in viscosity:
            combined['viscosity'] = viscosity['estimated_viscosity']
        elif 'viscosity' in ml:
            combined['viscosity'] = ml['viscosity']
        else:
            combined['viscosity'] = 1.0  # Default
        
        # Combine transparency
        if 'transparency_score' in transparency:
            combined['transparency'] = transparency['transparency_score']
        elif 'transparency' in ml:
            combined['transparency'] = ml['transparency']
        else:
            combined['transparency'] = 0.5  # Default
        
        # Combine cohesion and adhesion
        if 'cohesion_score' in cohesion:
            combined['cohesion'] = cohesion['cohesion_score']
            combined['adhesion'] = cohesion['adhesion_score']
        elif 'cohesion' in ml:
            combined['cohesion'] = ml['cohesion']
            combined['adhesion'] = ml['adhesion']
        else:
            combined['cohesion'] = 0.5  # Default
            combined['adhesion'] = 0.3  # Default
        
        # Add density from ML
        if 'density' in ml:
            combined['density'] = ml['density']
        else:
            combined['density'] = 1.0  # Default (water)
        
        return combined
    
    def _classify_liquid_type(self, properties: Dict) -> Dict:
        """Classify liquid type based on properties"""
        best_match = None
        best_score = 0
        
        for liquid_type, reference_props in self.liquid_properties.items():
            # Calculate similarity score
            score = 0
            count = 0
            
            for prop, value in properties.items():
                if prop in reference_props:
                    ref_value = reference_props[prop]
                    # Calculate similarity (inverse of relative difference)
                    similarity = 1.0 - abs(value - ref_value) / max(value, ref_value)
                    score += similarity
                    count += 1
            
            if count > 0:
                avg_score = score / count
                if avg_score > best_score:
                    best_score = avg_score
                    best_match = liquid_type
        
        return {
            'predicted_type': best_match or 'unknown',
            'confidence': best_score,
            'reference_properties': self.liquid_properties.get(best_match, {})
        }
    
    def _generate_liquid_recommendations(self, properties: Dict, classification: Dict) -> List[str]:
        """Generate recommendations based on liquid properties"""
        recommendations = []
        
        viscosity = properties.get('viscosity', 1.0)
        transparency = properties.get('transparency', 0.5)
        predicted_type = classification.get('predicted_type', 'unknown')
        
        # Viscosity-based recommendations
        if viscosity > 100:
            recommendations.append("Very viscous liquid - may indicate high sugar content")
        elif viscosity > 10:
            recommendations.append("Thick liquid - consume in moderation")
        elif viscosity < 1:
            recommendations.append("Thin liquid - likely hydrating")
        
        # Transparency-based recommendations
        if transparency < 0.3:
            recommendations.append("Opaque liquid - may contain pulp or be highly processed")
        elif transparency > 0.8:
            recommendations.append("Clear liquid - likely pure and minimally processed")
        
        # Type-specific recommendations
        if predicted_type == 'syrup':
            recommendations.append("High sugar content - limit consumption")
        elif predicted_type == 'oil':
            recommendations.append("High fat content - use sparingly")
        elif predicted_type == 'water':
            recommendations.append("Pure water - excellent for hydration")
        
        return recommendations
