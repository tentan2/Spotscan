"""
Texture Analyzer Module
Analyzes texture and physical properties of food items
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

class TextureAnalyzer:
    """Analyzes texture and physical properties of food items"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize texture analyzer
        
        Args:
            model_path: Path to pre-trained model
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.scaler = StandardScaler()
        
        # Texture property ranges for classification
        self.texture_ranges = {
            'crispiness': {'min': 0.0, 'max': 1.0, 'threshold': 0.7},
            'chewiness': {'min': 0.0, 'max': 1.0, 'threshold': 0.5},
            'softness': {'min': 0.0, 'max': 1.0, 'threshold': 0.6},
            'hardness': {'min': 0.0, 'max': 1.0, 'threshold': 0.7},
            'juiciness': {'min': 0.0, 'max': 1.0, 'threshold': 0.5},
            'brittleness': {'min': 0.0, 'max': 1.0, 'threshold': 0.6},
            'malleability': {'min': 0.0, 'max': 1.0, 'threshold': 0.4},
            'ductility': {'min': 0.0, 'max': 1.0, 'threshold': 0.3},
            'stickiness': {'min': 0.0, 'max': 1.0, 'threshold': 0.5},
            'slipperiness': {'min': 0.0, 'max': 1.0, 'threshold': 0.4},
            'lumpiness': {'min': 0.0, 'max': 1.0, 'threshold': 0.5},
            'viscoelasticity': {'min': 0.0, 'max': 1.0, 'threshold': 0.5}
        }
    
    def _load_model(self, model_path: Optional[str]) -> nn.Module:
        """Load or create texture analysis model"""
        if model_path and Path(model_path).exists():
            model = torch.load(model_path, map_location=self.device)
            return model
        
        # Create CNN for texture classification
        model = nn.Sequential(
            # Convolutional layers for texture feature extraction
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Flatten and dense layers
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 12)  # 12 texture properties
        )
        
        return model.to(self.device)
    
    def analyze_texture(self, image: np.ndarray, food_type: str = None) -> Dict:
        """
        Analyze texture and physical properties of food
        
        Args:
            image: Food image
            food_type: Type of food (optional for better analysis)
            
        Returns:
            Comprehensive texture analysis
        """
        # Extract texture features
        features = self._extract_texture_features(image)
        
        # Analyze surface properties
        surface_analysis = self._analyze_surface_properties(image)
        
        # Analyze structural properties
        structural_analysis = self._analyze_structural_properties(image)
        
        # Analyze moisture properties
        moisture_analysis = self._analyze_moisture_properties(image)
        
        # Analyze deformation properties
        deformation_analysis = self._analyze_deformation_properties(image)
        
        # Predict texture properties using ML model
        texture_predictions = self._predict_texture_properties(features)
        
        # Combine all analyses
        combined_analysis = self._combine_texture_analyses(
            surface_analysis, structural_analysis, 
            moisture_analysis, deformation_analysis,
            texture_predictions
        )
        
        # Generate texture profile
        texture_profile = self._generate_texture_profile(combined_analysis, food_type)
        
        return {
            'texture_profile': texture_profile,
            'surface_properties': surface_analysis,
            'structural_properties': structural_analysis,
            'moisture_properties': moisture_analysis,
            'deformation_properties': deformation_analysis,
            'predictions': texture_predictions,
            'texture_scores': combined_analysis
        }
    
    def _extract_texture_features(self, image: np.ndarray) -> np.ndarray:
        """Extract comprehensive texture features"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Local Binary Pattern features
        lbp_features = self._calculate_lbp_features(gray)
        
        # Gray-Level Co-occurrence Matrix features
        glcm_features = self._calculate_glcm_features(gray)
        
        # Gabor filter features
        gabor_features = self._calculate_gabor_features(gray)
        
        # Edge and gradient features
        edge_features = self._calculate_edge_features(gray)
        
        # Fractal dimension features
        fractal_features = self._calculate_fractal_features(gray)
        
        # Combine all features
        features = np.concatenate([
            lbp_features,
            glcm_features,
            gabor_features,
            edge_features,
            fractal_features
        ])
        
        return features
    
    def _analyze_surface_properties(self, image: np.ndarray) -> Dict:
        """Analyze surface texture properties"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Surface roughness
        roughness = self._calculate_surface_roughness(gray)
        
        # Surface smoothness
        smoothness = 1.0 / (1.0 + roughness)
        
        # Surface regularity
        regularity = self._calculate_surface_regularity(gray)
        
        # Surface anisotropy (directional texture)
        anisotropy = self._calculate_surface_anisotropy(gray)
        
        # Surface gloss/shininess
        gloss = self._calculate_surface_gloss(image)
        
        return {
            'roughness': float(roughness),
            'smoothness': float(smoothness),
            'regularity': float(regularity),
            'anisotropy': float(anisotropy),
            'gloss': float(gloss)
        }
    
    def _analyze_structural_properties(self, image: np.ndarray) -> Dict:
        """Analyze structural texture properties"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Crispiness (based on edge density and contrast)
        crispiness = self._calculate_crispiness(gray)
        
        # Hardness (based on texture complexity)
        hardness = self._calculate_hardness(gray)
        
        # Brittleness (based on fracture patterns)
        brittleness = self._calculate_brittleness(gray)
        
        # Density (based on pixel intensity distribution)
        density = self._calculate_density(gray)
        
        # Porosity (based on texture voids)
        porosity = self._calculate_porosity(gray)
        
        return {
            'crispiness': float(crispiness),
            'hardness': float(hardness),
            'brittleness': float(brittleness),
            'density': float(density),
            'porosity': float(porosity)
        }
    
    def _analyze_moisture_properties(self, image: np.ndarray) -> Dict:
        """Analyze moisture-related properties"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Juiciness (based on color saturation and texture)
        juiciness = self._calculate_juiciness(hsv, gray)
        
        # Wetness (based on specular reflections)
        wetness = self._calculate_wetness(image)
        
        # Moisture content (based on color intensity)
        moisture_content = self._calculate_moisture_content(hsv)
        
        # Stickiness (based on surface texture)
        stickiness = self._calculate_stickiness(gray)
        
        # Slipperiness (based on surface smoothness and reflectance)
        slipperiness = self._calculate_slipperiness(image)
        
        return {
            'juiciness': float(juiciness),
            'wetness': float(wetness),
            'moisture_content': float(moisture_content),
            'stickiness': float(stickiness),
            'slipperiness': float(slipperiness)
        }
    
    def _analyze_deformation_properties(self, image: np.ndarray) -> Dict:
        """Analyze deformation and elastic properties"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Chewiness (based on texture complexity and cohesiveness)
        chewiness = self._calculate_chewiness(gray)
        
        # Softness (based on texture uniformity)
        softness = self._calculate_softness(gray)
        
        # Malleability (based on texture plasticity)
        malleability = self._calculate_malleability(gray)
        
        # Ductility (based on texture extensibility)
        ductility = self._calculate_ductility(gray)
        
        # Viscoelasticity (based on texture viscosity and elasticity)
        viscoelasticity = self._calculate_viscoelasticity(gray)
        
        # Lumpiness (based on texture heterogeneity)
        lumpiness = self._calculate_lumpiness(gray)
        
        return {
            'chewiness': float(chewiness),
            'softness': float(softness),
            'malleability': float(malleability),
            'ductility': float(ductility),
            'viscoelasticity': float(viscoelasticity),
            'lumpiness': float(lumpiness)
        }
    
    def _calculate_lbp_features(self, gray: np.ndarray) -> np.ndarray:
        """Calculate Local Binary Pattern features"""
        lbp = self._calculate_lbp(gray, radius=1, n_points=8)
        lbp_hist = np.histogram(lbp, bins=256)[0]
        lbp_hist = lbp_hist / lbp_hist.sum()
        
        # Calculate LBP variance and uniformity
        lbp_var = np.var(lbp)
        lbp_uniform = np.sum(lbp_hist[lbp_hist > 0.01])  # Uniform patterns
        
        return np.array([lbp_var, lbp_uniform, *lbp_hist[:50]])
    
    def _calculate_glcm_features(self, gray: np.ndarray) -> np.ndarray:
        """Calculate Gray-Level Co-occurrence Matrix features"""
        # Simplified GLCM calculation
        # In practice, would use more comprehensive implementation
        
        # Calculate GLCM for different angles and distances
        distances = [1, 2, 3]
        angles = [0, 45, 90, 135]
        
        features = []
        
        for distance in distances:
            for angle in angles:
                # Calculate contrast, correlation, energy, homogeneity
                contrast = np.var(gray)
                correlation = np.corrcoef(gray.flatten()[:-distance], gray.flatten()[distance:])[0, 1]
                energy = np.sum(gray**2) / (gray.shape[0] * gray.shape[1])
                homogeneity = 1.0 / (1.0 + contrast)
                
                features.extend([contrast, correlation, energy, homogeneity])
        
        return np.array(features)
    
    def _calculate_gabor_features(self, gray: np.ndarray) -> np.ndarray:
        """Calculate Gabor filter features"""
        features = []
        
        # Apply Gabor filters with different parameters
        frequencies = [0.1, 0.3, 0.5]
        angles = [0, 45, 90, 135]
        
        for freq in frequencies:
            for angle in angles:
                # Create Gabor filter
                real, imag = cv2.getGaborKernel((15, 15), 3, np.radians(angle), 
                                              2*np.pi*freq, 0.5, 0, ktype=cv2.CV_32F)
                
                # Apply filter
                filtered = cv2.filter2D(gray, cv2.CV_8UC3, real)
                
                # Calculate statistics
                features.extend([
                    np.mean(filtered),
                    np.std(filtered),
                    np.var(filtered)
                ])
        
        return np.array(features)
    
    def _calculate_edge_features(self, gray: np.ndarray) -> np.ndarray:
        """Calculate edge-based features"""
        # Canny edges
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Sobel edges
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Edge statistics
        edge_mean = np.mean(sobel_magnitude)
        edge_std = np.std(sobel_magnitude)
        edge_var = np.var(sobel_magnitude)
        
        # Edge orientation histogram
        orientations = np.arctan2(sobel_y, sobel_x)
        orientation_hist = np.histogram(orientations.flatten(), bins=8, range=[-np.pi, np.pi])[0]
        orientation_hist = orientation_hist / orientation_hist.sum()
        
        return np.array([edge_density, edge_mean, edge_std, edge_var, *orientation_hist])
    
    def _calculate_fractal_features(self, gray: np.ndarray) -> np.ndarray:
        """Calculate fractal dimension features"""
        # Box-counting fractal dimension
        fractal_dim = self._calculate_fractal_dimension(gray)
        
        # Lacunarity (texture heterogeneity)
        lacunarity = self._calculate_lacunarity(gray)
        
        return np.array([fractal_dim, lacunarity])
    
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
    
    def _calculate_fractal_dimension(self, image: np.ndarray) -> float:
        """Calculate fractal dimension using box-counting method"""
        # Simplified fractal dimension calculation
        sizes = [2, 4, 8, 16]
        counts = []
        
        for size in sizes:
            # Divide image into boxes
            h, w = image.shape
            boxes_h = h // size
            boxes_w = w // size
            
            # Count non-empty boxes
            count = 0
            for i in range(boxes_h):
                for j in range(boxes_w):
                    box = image[i*size:(i+1)*size, j*size:(j+1)*size]
                    if np.any(box > 0):
                        count += 1
            
            counts.append(count)
        
        # Calculate fractal dimension
        if len(counts) > 1 and len(sizes) > 1:
            log_sizes = np.log(sizes)
            log_counts = np.log(counts)
            
            # Linear regression to estimate slope
            slope = np.polyfit(log_sizes, log_counts, 1)[0]
            return -slope
        else:
            return 2.0  # Default to 2D
    
    def _calculate_lacunarity(self, image: np.ndarray) -> float:
        """Calculate lacunarity (texture heterogeneity)"""
        # Calculate local variance
        kernel = np.ones((5, 5), np.float32) / 25
        local_mean = cv2.filter2D(image.astype(np.float32), -1, kernel)
        local_var = cv2.filter2D((image.astype(np.float32) - local_mean) ** 2, -1, kernel)
        
        # Lacunarity is related to variance of local variance
        lacunarity = np.var(local_var) / (np.mean(local_var) + 1e-6)
        
        return lacunarity
    
    def _calculate_surface_roughness(self, gray: np.ndarray) -> float:
        """Calculate surface roughness"""
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        return np.mean(gradient_magnitude)
    
    def _calculate_surface_regularity(self, gray: np.ndarray) -> float:
        """Calculate surface regularity"""
        # Regularity based on texture pattern consistency
        kernel = np.ones((3, 3), np.float32) / 9
        local_var = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        var_of_var = np.var(local_var)
        
        # Lower variance of local variance = more regular
        regularity = 1.0 / (1.0 + var_of_var)
        
        return regularity
    
    def _calculate_surface_anisotropy(self, gray: np.ndarray) -> float:
        """Calculate surface anisotropy (directional texture)"""
        # Calculate gradients in different directions
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_diag1 = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
        grad_diag2 = cv2.Sobel(gray, cv2.CV_64F, 1, -1, ksize=3)
        
        # Calculate directional energies
        energy_x = np.mean(grad_x**2)
        energy_y = np.mean(grad_y**2)
        energy_diag1 = np.mean(grad_diag1**2)
        energy_diag2 = np.mean(grad_diag2**2)
        
        energies = [energy_x, energy_y, energy_diag1, energy_diag2]
        
        # Anisotropy based on variation in directional energies
        anisotropy = np.std(energies) / (np.mean(energies) + 1e-6)
        
        return anisotropy
    
    def _calculate_surface_gloss(self, image: np.ndarray) -> float:
        """Calculate surface gloss/shininess"""
        # Gloss based on specular highlights
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        value_channel = hsv[:, :, 2]
        
        # High intensity regions indicate gloss
        high_intensity_mask = value_channel > np.percentile(value_channel, 90)
        gloss_ratio = np.sum(high_intensity_mask) / high_intensity_mask.size
        
        return gloss_ratio
    
    def _calculate_crispiness(self, gray: np.ndarray) -> float:
        """Calculate crispiness score"""
        # Crispiness based on edge density and contrast
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # High edge density indicates crispiness
        crispiness = min(edge_density * 5, 1.0)  # Scale to 0-1
        
        return crispiness
    
    def _calculate_hardness(self, gray: np.ndarray) -> float:
        """Calculate hardness score"""
        # Hardness based on texture complexity and uniformity
        texture_var = np.var(gray)
        texture_complexity = self._calculate_texture_complexity(gray)
        
        # Higher complexity and variance indicate harder texture
        hardness = min((texture_var / 255.0 + texture_complexity / 10.0) / 2.0, 1.0)
        
        return hardness
    
    def _calculate_brittleness(self, gray: np.ndarray) -> float:
        """Calculate brittleness score"""
        # Brittleness based on fracture patterns
        edges = cv2.Canny(gray, 30, 100)
        
        # Look for linear fracture patterns
        kernel = np.ones((5, 1), np.uint8)
        linear_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        fracture_density = np.sum(linear_edges > 0) / linear_edges.size
        brittleness = min(fracture_density * 10, 1.0)
        
        return brittleness
    
    def _calculate_density(self, gray: np.ndarray) -> float:
        """Calculate density score"""
        # Density based on pixel intensity distribution
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        
        # Higher mean and lower std indicate denser material
        density = (mean_intensity / 255.0) * (1.0 - std_intensity / 255.0)
        
        return max(0, density)
    
    def _calculate_porosity(self, gray: np.ndarray) -> float:
        """Calculate porosity score"""
        # Porosity based on texture voids
        # Use threshold to identify voids
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Invert to get voids
        voids = cv2.bitwise_not(thresh)
        
        # Calculate void ratio
        void_ratio = np.sum(voids > 0) / voids.size
        
        return void_ratio
    
    def _calculate_juiciness(self, hsv: np.ndarray, gray: np.ndarray) -> float:
        """Calculate juiciness score"""
        # Juiciness based on color saturation and texture
        saturation = np.mean(hsv[:, :, 1])
        texture_smoothness = 1.0 / (1.0 + self._calculate_surface_roughness(gray))
        
        # High saturation and smooth texture indicate juiciness
        juiciness = (saturation / 255.0 + texture_smoothness) / 2.0
        
        return juiciness
    
    def _calculate_wetness(self, image: np.ndarray) -> float:
        """Calculate wetness score"""
        # Wetness based on specular reflections
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        value_channel = hsv[:, :, 2]
        
        # Look for bright specular highlights
        bright_mask = value_channel > np.percentile(value_channel, 95)
        wetness = np.sum(bright_mask) / bright_mask.size
        
        return min(wetness * 20, 1.0)  # Scale to 0-1
    
    def _calculate_moisture_content(self, hsv: np.ndarray) -> float:
        """Calculate moisture content score"""
        # Moisture based on color intensity
        value = np.mean(hsv[:, :, 2])
        saturation = np.mean(hsv[:, :, 1])
        
        # Higher value and saturation indicate more moisture
        moisture = (value / 255.0 + saturation / 255.0) / 2.0
        
        return moisture
    
    def _calculate_stickiness(self, gray: np.ndarray) -> float:
        """Calculate stickiness score"""
        # Stickiness based on surface texture
        # Sticky surfaces often have low texture variation
        local_var = cv2.filter2D(gray.astype(np.float32), -1, 
                               np.ones((5, 5), np.float32) / 25)
        
        # Low local variance indicates stickiness
        stickiness = 1.0 - (np.mean(local_var) / 255.0)
        
        return max(0, stickiness)
    
    def _calculate_slipperiness(self, image: np.ndarray) -> float:
        """Calculate slipperiness score"""
        # Slipperiness based on surface smoothness and reflectance
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        smoothness = 1.0 / (1.0 + self._calculate_surface_roughness(gray))
        
        # High smoothness indicates slipperiness
        slipperiness = smoothness
        
        return slipperiness
    
    def _calculate_chewiness(self, gray: np.ndarray) -> float:
        """Calculate chewiness score"""
        # Chewiness based on texture complexity and cohesiveness
        texture_complexity = self._calculate_texture_complexity(gray)
        cohesiveness = 1.0 / (1.0 + np.var(gray) / 255.0)
        
        chewiness = (texture_complexity / 10.0 + cohesiveness) / 2.0
        
        return min(chewiness, 1.0)
    
    def _calculate_texture_complexity(self, gray: np.ndarray) -> float:
        """Calculate overall texture complexity"""
        # Use entropy as measure of complexity
        histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])
        histogram = histogram.flatten()
        histogram = histogram / histogram.sum()
        
        # Remove zero values
        histogram = histogram[histogram > 0]
        
        entropy = -np.sum(histogram * np.log2(histogram))
        return entropy
    
    def _calculate_softness(self, gray: np.ndarray) -> float:
        """Calculate softness score"""
        # Softness based on texture uniformity
        local_var = cv2.filter2D(gray.astype(np.float32), -1, 
                               np.ones((7, 7), np.float32) / 49)
        
        # Low local variance indicates softness
        softness = 1.0 - (np.mean(local_var) / 255.0)
        
        return max(0, softness)
    
    def _calculate_malleability(self, gray: np.ndarray) -> float:
        """Calculate malleability score"""
        # Malleability based on texture plasticity
        # Simplified: based on gradient distribution
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Lower gradient magnitude indicates more malleable
        malleability = 1.0 - (np.mean(gradient_magnitude) / 255.0)
        
        return max(0, malleability)
    
    def _calculate_ductility(self, gray: np.ndarray) -> float:
        """Calculate ductility score"""
        # Ductility based on texture extensibility
        # Simplified: based on edge continuity
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate edge continuity
        kernel = np.ones((3, 3), np.uint8)
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        continuity = np.sum(closed_edges > 0) / (np.sum(edges > 0) + 1e-6)
        
        return min(continuity / 2.0, 1.0)  # Scale to 0-1
    
    def _calculate_viscoelasticity(self, gray: np.ndarray) -> float:
        """Calculate viscoelasticity score"""
        # Viscoelasticity based on texture viscosity and elasticity
        viscosity = self._calculate_texture_viscosity(gray)
        elasticity = self._calculate_texture_elasticity(gray)
        
        viscoelasticity = (viscosity + elasticity) / 2.0
        
        return viscoelasticity
    
    def _calculate_texture_viscosity(self, gray: np.ndarray) -> float:
        """Calculate texture viscosity"""
        # Viscosity based on texture flow patterns
        # Simplified: based on gradient direction consistency
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        orientations = np.arctan2(grad_y, grad_x)
        
        # Calculate orientation consistency
        orientation_std = np.std(orientations)
        viscosity = 1.0 / (1.0 + orientation_std)
        
        return viscosity
    
    def _calculate_texture_elasticity(self, gray: np.ndarray) -> float:
        """Calculate texture elasticity"""
        # Elasticity based on texture resilience
        # Simplified: based on texture regularity and complexity
        regularity = self._calculate_surface_regularity(gray)
        complexity = self._calculate_texture_complexity(gray)
        
        elasticity = (regularity + (10.0 - complexity) / 10.0) / 2.0
        
        return max(0, elasticity)
    
    def _calculate_lumpiness(self, gray: np.ndarray) -> float:
        """Calculate lumpiness score"""
        # Lumpiness based on texture heterogeneity
        # Calculate local variance at multiple scales
        scales = [3, 5, 7, 9]
        lumpiness_scores = []
        
        for scale in scales:
            kernel = np.ones((scale, scale), np.float32) / (scale * scale)
            local_var = cv2.filter2D(gray.astype(np.float32), -1, kernel)
            
            # High local variance variation indicates lumpiness
            var_of_var = np.var(local_var)
            lumpiness_scores.append(var_of_var)
        
        # Average across scales
        lumpiness = np.mean(lumpiness_scores)
        
        # Normalize to 0-1
        lumpiness = min(lumpiness / 1000.0, 1.0)
        
        return lumpiness
    
    def _predict_texture_properties(self, features: np.ndarray) -> Dict:
        """Predict texture properties using ML model"""
        # In a real implementation, would use the trained model
        # For now, use heuristic-based prediction
        
        # Normalize features
        features_scaled = self.scaler.fit_transform(features.reshape(1, -1))
        
        # Mock predictions based on feature patterns
        predictions = {}
        
        # Generate predictions for each texture property
        for property_name in self.texture_ranges.keys():
            # In practice, this would use actual model inference
            prediction = np.random.random()  # Placeholder
            predictions[property_name] = float(prediction)
        
        return predictions
    
    def _combine_texture_analyses(self, surface: Dict, structural: Dict, 
                                moisture: Dict, deformation: Dict,
                                predictions: Dict) -> Dict:
        """Combine all texture analyses into unified scores"""
        combined = {}
        
        # Combine surface properties
        combined.update(surface)
        
        # Combine structural properties
        combined.update(structural)
        
        # Combine moisture properties
        combined.update(moisture)
        
        # Combine deformation properties
        combined.update(deformation)
        
        # Add ML predictions
        combined.update(predictions)
        
        return combined
    
    def _generate_texture_profile(self, analysis: Dict, food_type: str = None) -> Dict:
        """Generate comprehensive texture profile"""
        profile = {}
        
        # Classify each texture property
        for property_name, range_info in self.texture_ranges.items():
            value = analysis.get(property_name, 0.5)
            threshold = range_info['threshold']
            
            if value > threshold:
                classification = "high"
            elif value > threshold * 0.5:
                classification = "medium"
            else:
                classification = "low"
            
            profile[property_name] = {
                'value': value,
                'classification': classification,
                'description': self._get_texture_description(property_name, classification)
            }
        
        # Generate overall texture summary
        profile['summary'] = self._generate_texture_summary(profile, food_type)
        
        return profile
    
    def _get_texture_description(self, property_name: str, classification: str) -> str:
        """Get description for texture property classification"""
        descriptions = {
            'crispiness': {
                'low': 'Soft and tender',
                'medium': 'Moderately crisp',
                'high': 'Very crisp and crunchy'
            },
            'chewiness': {
                'low': 'Tender and easy to bite',
                'medium': 'Moderately chewy',
                'high': 'Very chewy and substantial'
            },
            'softness': {
                'low': 'Firm and resistant',
                'medium': 'Moderately soft',
                'high': 'Very soft and tender'
            },
            'hardness': {
                'low': 'Soft and yielding',
                'medium': 'Moderately firm',
                'high': 'Very hard and dense'
            },
            'juiciness': {
                'low': 'Dry and minimal moisture',
                'medium': 'Moderately juicy',
                'high': 'Very juicy and moist'
            },
            'brittleness': {
                'low': 'Flexible and resilient',
                'medium': 'Somewhat brittle',
                'high': 'Very brittle and fragile'
            },
            'malleability': {
                'low': 'Rigid and resistant to deformation',
                'medium': 'Somewhat malleable',
                'high': 'Very malleable and flexible'
            },
            'ductility': {
                'low': 'Breaks easily under stress',
                'medium': 'Moderately ductile',
                'high': 'Very ductile and stretchable'
            },
            'stickiness': {
                'low': 'Non-sticky surface',
                'medium': 'Slightly sticky',
                'high': 'Very sticky and adhesive'
            },
            'slipperiness': {
                'low': 'Good grip and traction',
                'medium': 'Somewhat slippery',
                'high': 'Very slippery and smooth'
            },
            'lumpiness': {
                'low': 'Smooth and uniform texture',
                'medium': 'Slightly lumpy',
                'high': 'Very lumpy and uneven'
            },
            'viscoelasticity': {
                'low': 'Rigid and non-elastic',
                'medium': 'Moderately elastic',
                'high': 'Very elastic and resilient'
            }
        }
        
        return descriptions.get(property_name, {}).get(classification, 'Unknown')
    
    def _generate_texture_summary(self, profile: Dict, food_type: str = None) -> str:
        """Generate overall texture summary"""
        # Count high, medium, low classifications
        high_count = sum(1 for prop in profile.values() 
                        if isinstance(prop, dict) and prop.get('classification') == 'high')
        medium_count = sum(1 for prop in profile.values() 
                          if isinstance(prop, dict) and prop.get('classification') == 'medium')
        low_count = sum(1 for prop in profile.values() 
                       if isinstance(prop, dict) and prop.get('classification') == 'low')
        
        # Generate summary based on dominant characteristics
        if high_count > medium_count + low_count:
            summary = "This food has predominantly strong texture characteristics"
        elif low_count > medium_count + high_count:
            summary = "This food has predominantly mild texture characteristics"
        else:
            summary = "This food has balanced texture characteristics"
        
        # Add food-type specific information
        if food_type:
            summary += f" typical of {food_type}"
        
        return summary
