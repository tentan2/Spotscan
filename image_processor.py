"""
Image Processing Module
Handles image preprocessing, enhancement, and analysis operations
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from typing import Tuple, Dict, Optional
import colorsys

class ImageProcessor:
    """Advanced image processing for food analysis"""
    
    def __init__(self):
        self.target_size = (224, 224)
        
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load image from file path
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image as numpy array in BGR format
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        return image
    
    def resize_image(self, image: np.ndarray, size: Tuple[int, int] = None) -> np.ndarray:
        """
        Resize image to target dimensions
        
        Args:
            image: Input image
            size: Target size (width, height)
            
        Returns:
            Resized image
        """
        if size is None:
            size = self.target_size
        
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image quality for better analysis
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image
        """
        # Convert to PIL for enhancement
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(1.2)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(pil_image)
        pil_image = enhancer.enhance(1.1)
        
        # Convert back to OpenCV format
        enhanced = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return enhanced
    
    def analyze_color_distribution(self, image: np.ndarray) -> Dict:
        """
        Analyze color distribution in the image
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with color analysis results
        """
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Calculate dominant colors
        dominant_colors = self._get_dominant_colors(image, k=5)
        
        # Calculate color statistics
        color_stats = {
            'mean_rgb': np.mean(image, axis=(0, 1)).tolist(),
            'std_rgb': np.std(image, axis=(0, 1)).tolist(),
            'mean_hsv': np.mean(hsv, axis=(0, 1)).tolist(),
            'std_hsv': np.std(hsv, axis=(0, 1)).tolist(),
            'mean_lab': np.mean(lab, axis=(0, 1)).tolist(),
            'std_lab': np.std(lab, axis=(0, 1)).tolist(),
            'dominant_colors': dominant_colors
        }
        
        return color_stats
    
    def _get_dominant_colors(self, image: np.ndarray, k: int = 5) -> list:
        """
        Get dominant colors using K-means clustering
        
        Args:
            image: Input image
            k: Number of clusters
            
        Returns:
            List of dominant colors in hex format
        """
        # Reshape image to be a list of pixels
        pixels = image.reshape(-1, 3)
        
        # Convert to float32
        pixels = np.float32(pixels)
        
        # Define criteria and apply kmeans
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert back to uint8 and get hex colors
        centers = np.uint8(centers)
        hex_colors = []
        
        for color in centers:
            # Convert BGR to RGB
            rgb_color = [color[2], color[1], color[0]]
            hex_color = '#{:02x}{:02x}{:02x}'.format(*rgb_color)
            hex_colors.append(hex_color)
        
        return hex_colors
    
    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Detect edges in the image
        
        Args:
            image: Input image
            
        Returns:
            Edge-detected image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        return edges
    
    def analyze_texture(self, image: np.ndarray) -> Dict:
        """
        Analyze texture properties of the image
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with texture analysis results
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate Local Binary Pattern (LBP) for texture analysis
        lbp = self._calculate_lbp(gray)
        
        # Calculate texture features
        texture_features = {
            'lbp_histogram': np.histogram(lbp, bins=256)[0].tolist(),
            'lbp_mean': np.mean(lbp),
            'lbp_std': np.std(lbp),
            'contrast': self._calculate_contrast(gray),
            'homogeneity': self._calculate_homogeneity(gray),
            'entropy': self._calculate_entropy(gray)
        }
        
        return texture_features
    
    def _calculate_lbp(self, image: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
        """
        Calculate Local Binary Pattern
        
        Args:
            image: Grayscale image
            radius: Radius for LBP calculation
            n_points: Number of sampling points
            
        Returns:
            LBP image
        """
        height, width = image.shape
        lbp = np.zeros((height, width), dtype=np.uint8)
        
        for i in range(radius, height - radius):
            for j in range(radius, width - radius):
                center = image[i, j]
                binary_string = []
                
                # Sample points in circular neighborhood
                for angle in np.linspace(0, 2 * np.pi, n_points, endpoint=False):
                    x = i + radius * np.cos(angle)
                    y = j + radius * np.sin(angle)
                    
                    # Bilinear interpolation
                    x1, y1 = int(x), int(y)
                    x2, y2 = min(x1 + 1, height - 1), min(y1 + 1, width - 1)
                    
                    dx, dy = x - x1, y - y1
                    
                    pixel_value = (1 - dx) * (1 - dy) * image[x1, y1] + \
                                dx * (1 - dy) * image[x2, y1] + \
                                (1 - dx) * dy * image[x1, y2] + \
                                dx * dy * image[x2, y2]
                    
                    binary_string.append(1 if pixel_value >= center else 0)
                
                # Convert binary string to decimal
                lbp_value = 0
                for bit in binary_string:
                    lbp_value = (lbp_value << 1) | bit
                
                lbp[i, j] = lbp_value
        
        return lbp
    
    def _calculate_contrast(self, image: np.ndarray) -> float:
        """Calculate contrast measure"""
        return np.std(image)
    
    def _calculate_homogeneity(self, image: np.ndarray) -> float:
        """Calculate homogeneity measure"""
        # Simple homogeneity based on local variance
        kernel = np.ones((3, 3), np.float32) / 9
        local_mean = cv2.filter2D(image.astype(np.float32), -1, kernel)
        local_var = cv2.filter2D((image.astype(np.float32) - local_mean) ** 2, -1, kernel)
        return 1.0 / (1.0 + np.mean(local_var))
    
    def _calculate_entropy(self, image: np.ndarray) -> float:
        """Calculate entropy of the image"""
        histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
        histogram = histogram.flatten()
        histogram = histogram / histogram.sum()
        
        # Remove zero values to avoid log(0)
        histogram = histogram[histogram > 0]
        
        entropy = -np.sum(histogram * np.log2(histogram))
        return entropy
    
    def estimate_lighting_conditions(self, image: np.ndarray) -> Dict:
        """
        Estimate lighting conditions of the image
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with lighting analysis
        """
        # Convert to different color spaces
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Extract L channel (lightness)
        l_channel = lab[:, :, 0]
        
        # Calculate lighting statistics
        lighting_stats = {
            'brightness': np.mean(l_channel),
            'brightness_std': np.std(l_channel),
            'lighting_uniformity': 1.0 - (np.std(l_channel) / np.mean(l_channel)),
            'lighting_quality': self._assess_lighting_quality(l_channel)
        }
        
        return lighting_stats
    
    def _assess_lighting_quality(self, l_channel: np.ndarray) -> str:
        """
        Assess lighting quality
        
        Args:
            l_channel: Lightness channel
            
        Returns:
            Lighting quality assessment
        """
        mean_brightness = np.mean(l_channel)
        std_brightness = np.std(l_channel)
        
        if mean_brightness < 50:
            return "too_dark"
        elif mean_brightness > 200:
            return "too_bright"
        elif std_brightness > 80:
            return "uneven"
        else:
            return "good"
    
    def preprocess_for_analysis(self, image: np.ndarray) -> np.ndarray:
        """
        Complete preprocessing pipeline for food analysis
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        # Step 1: Enhance image
        enhanced = self.enhance_image(image)
        
        # Step 2: Resize if needed
        if enhanced.shape[:2] != (self.target_size[1], self.target_size[0]):
            enhanced = self.resize_image(enhanced, self.target_size)
        
        return enhanced
