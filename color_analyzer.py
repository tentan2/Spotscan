"""
Color Analyzer Module
Analyzes food colors and provides hex color output for spoilage detection and composition accuracy
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import colorsys
from sklearn.cluster import KMeans
from collections import Counter
import matplotlib.pyplot as plt
from pathlib import Path

class ColorAnalyzer:
    """Analyzes food colors with hex output for comprehensive color analysis"""
    
    def __init__(self):
        """Initialize color analyzer"""
        # Color ranges for spoilage detection
        self.spoilage_colors = {
            'brown': {
                'hsv_range': [(8, 50, 50), (25, 255, 255)],
                'description': 'Brown discoloration',
                'spoilage_indicator': True
            },
            'gray': {
                'hsv_range': [(0, 0, 50), (180, 30, 200)],
                'description': 'Gray/white mold',
                'spoilage_indicator': True
            },
            'green_mold': {
                'hsv_range': [(35, 40, 40), (85, 255, 255)],
                'description': 'Green mold spots',
                'spoilage_indicator': True
            },
            'blue_mold': {
                'hsv_range': [(100, 40, 40), (130, 255, 255)],
                'description': 'Blue mold spots',
                'spoilage_indicator': True
            },
            'black': {
                'hsv_range': [(0, 0, 0), (180, 255, 50)],
                'description': 'Black spots',
                'spoilage_indicator': True
            }
        }
        
        # Fresh color indicators for common foods
        self.fresh_colors = {
            'leafy_green': {
                'hsv_range': [(35, 50, 50), (85, 255, 255)],
                'description': 'Fresh green',
                'freshness_indicator': True
            },
            'bright_red': {
                'hsv_range': [(0, 50, 50), (25, 255, 255)],
                'description': 'Bright red',
                'freshness_indicator': True
            },
            'vibrant_orange': {
                'hsv_range': [(10, 50, 50), (25, 255, 255)],
                'description': 'Vibrant orange',
                'freshness_indicator': True
            },
            'bright_yellow': {
                'hsv_range': [(20, 50, 50), (35, 255, 255)],
                'description': 'Bright yellow',
                'freshness_indicator': True
            }
        }
    
    def analyze_colors(self, image: np.ndarray, food_type: str = None) -> Dict:
        """
        Perform comprehensive color analysis
        
        Args:
            image: Food image
            food_type: Type of food (optional for better analysis)
            
        Returns:
            Comprehensive color analysis results
        """
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Extract dominant colors
        dominant_colors = self._extract_dominant_colors(image, num_colors=10)
        
        # Analyze color distribution
        color_distribution = self._analyze_color_distribution(hsv, lab)
        
        # Detect spoilage colors
        spoilage_analysis = self._detect_spoilage_colors(hsv)
        
        # Detect freshness indicators
        freshness_analysis = self._detect_freshness_colors(hsv, food_type)
        
        # Analyze color uniformity
        uniformity_analysis = self._analyze_color_uniformity(hsv)
        
        # Calculate color statistics
        color_statistics = self._calculate_color_statistics(rgb, hsv, lab)
        
        # Generate color palette
        color_palette = self._generate_color_palette(dominant_colors)
        
        # Cross-reference with dataset
        dataset_comparison = self._cross_reference_dataset(dominant_colors, food_type)
        
        return {
            'dominant_colors': dominant_colors,
            'color_distribution': color_distribution,
            'spoilage_analysis': spoilage_analysis,
            'freshness_analysis': freshness_analysis,
            'uniformity_analysis': uniformity_analysis,
            'color_statistics': color_statistics,
            'color_palette': color_palette,
            'dataset_comparison': dataset_comparison,
            'color_summary': self._generate_color_summary(
                dominant_colors, spoilage_analysis, freshness_analysis
            )
        }
    
    def _extract_dominant_colors(self, image: np.ndarray, num_colors: int = 10) -> List[Dict]:
        """
        Extract dominant colors using K-means clustering
        
        Args:
            image: Input image
            num_colors: Number of dominant colors to extract
            
        Returns:
            List of dominant colors with hex codes and percentages
        """
        # Reshape image to be a list of pixels
        pixels = image.reshape(-1, 3)
        
        # Convert to float32 for K-means
        pixels = np.float32(pixels)
        
        # Define criteria and apply K-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert centers to uint8
        centers = np.uint8(centers)
        
        # Calculate percentage of each color
        label_counts = Counter(labels.flatten())
        total_pixels = len(labels)
        
        # Create color information
        dominant_colors = []
        
        for i, center in enumerate(centers):
            # Convert BGR to RGB
            rgb_color = [center[2], center[1], center[0]]
            
            # Convert to hex
            hex_color = '#{:02x}{:02x}{:02x}'.format(*rgb_color)
            
            # Calculate percentage
            percentage = label_counts[i] / total_pixels
            
            # Get color name
            color_name = self._get_color_name(rgb_color)
            
            dominant_colors.append({
                'rgb': rgb_color,
                'hex': hex_color,
                'percentage': float(percentage),
                'color_name': color_name,
                'pixel_count': int(label_counts[i])
            })
        
        # Sort by percentage (dominant first)
        dominant_colors.sort(key=lambda x: x['percentage'], reverse=True)
        
        return dominant_colors
    
    def _analyze_color_distribution(self, hsv: np.ndarray, lab: np.ndarray) -> Dict:
        """Analyze color distribution across different color spaces"""
        # HSV distribution
        h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])
        
        # Normalize histograms
        h_hist = h_hist.flatten() / h_hist.sum()
        s_hist = s_hist.flatten() / s_hist.sum()
        v_hist = v_hist.flatten() / v_hist.sum()
        
        # LAB distribution
        l_hist = cv2.calcHist([lab], [0], None, [256], [0, 256])
        a_hist = cv2.calcHist([lab], [1], None, [256], [0, 256])
        b_hist = cv2.calcHist([lab], [2], None, [256], [0, 256])
        
        # Normalize LAB histograms
        l_hist = l_hist.flatten() / l_hist.sum()
        a_hist = a_hist.flatten() / a_hist.sum()
        b_hist = b_hist.flatten() / b_hist.sum()
        
        # Calculate distribution statistics
        distribution_stats = {
            'hue_peak': int(np.argmax(h_hist)),
            'saturation_mean': float(np.mean(s_hist)),
            'value_mean': float(np.mean(v_hist)),
            'lightness_mean': float(np.mean(l_hist)),
            'a_channel_mean': float(np.mean(a_hist)),
            'b_channel_mean': float(np.mean(b_hist)),
            'hue_variance': float(np.var(h_hist)),
            'saturation_variance': float(np.var(s_hist)),
            'value_variance': float(np.var(v_hist))
        }
        
        return {
            'hsv_histograms': {
                'hue': h_hist.tolist(),
                'saturation': s_hist.tolist(),
                'value': v_hist.tolist()
            },
            'lab_histograms': {
                'lightness': l_hist.tolist(),
                'a_channel': a_hist.tolist(),
                'b_channel': b_hist.tolist()
            },
            'statistics': distribution_stats
        }
    
    def _detect_spoilage_colors(self, hsv: np.ndarray) -> Dict:
        """Detect colors indicative of spoilage"""
        spoilage_detected = []
        total_spoilage_area = 0
        
        for color_name, color_info in self.spoilage_colors.items():
            hsv_range = color_info['hsv_range']
            lower = np.array(hsv_range[0])
            upper = np.array(hsv_range[1])
            
            # Create mask for this color range
            mask = cv2.inRange(hsv, lower, upper)
            
            # Calculate area ratio
            area_ratio = np.sum(mask > 0) / mask.size
            
            # Clean up mask with morphological operations
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours for this color
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Analyze each contour
            spots = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(contour)
                    spots.append({
                        'area': float(area),
                        'bbox': [int(x), int(y), int(w), int(h)],
                        'center': [int(x + w/2), int(y + h/2)]
                    })
            
            if area_ratio > 0.001:  # 0.1% threshold
                spoilage_detected.append({
                    'color_type': color_name,
                    'description': color_info['description'],
                    'area_ratio': float(area_ratio),
                    'spots': spots,
                    'severity': self._assess_spoilage_severity(area_ratio, len(spots))
                })
                total_spoilage_area += area_ratio
        
        # Overall spoilage assessment
        overall_risk = self._assess_overall_spoilage_risk(total_spoilage_area, len(spoilage_detected))
        
        return {
            'spoilage_colors': spoilage_detected,
            'total_spoilage_area': float(total_spoilage_area),
            'overall_risk': overall_risk,
            'spoilage_detected': len(spoilage_detected) > 0
        }
    
    def _detect_freshness_colors(self, hsv: np.ndarray, food_type: str = None) -> Dict:
        """Detect colors indicative of freshness"""
        freshness_indicators = []
        
        for color_name, color_info in self.fresh_colors.items():
            hsv_range = color_info['hsv_range']
            lower = np.array(hsv_range[0])
            upper = np.array(hsv_range[1])
            
            # Create mask for this color range
            mask = cv2.inRange(hsv, lower, upper)
            
            # Calculate area ratio
            area_ratio = np.sum(mask > 0) / mask.size
            
            if area_ratio > 0.01:  # 1% threshold
                freshness_indicators.append({
                    'color_type': color_name,
                    'description': color_info['description'],
                    'area_ratio': float(area_ratio),
                    'freshness_score': min(area_ratio * 10, 1.0)  # Scale to 0-1
                })
        
        # Overall freshness assessment
        overall_freshness = self._assess_overall_freshness(freshness_indicators, food_type)
        
        return {
            'freshness_colors': freshness_indicators,
            'overall_freshness_score': overall_freshness,
            'fresh_indicators_detected': len(freshness_indicators) > 0
        }
    
    def _analyze_color_uniformity(self, hsv: np.ndarray) -> Dict:
        """Analyze color uniformity across the image"""
        # Calculate standard deviation for each channel
        h_std = np.std(hsv[:, :, 0])
        s_std = np.std(hsv[:, :, 1])
        v_std = np.std(hsv[:, :, 2])
        
        # Calculate overall uniformity score
        overall_std = (h_std + s_std + v_std) / 3
        uniformity_score = 1.0 / (1.0 + overall_std / 50.0)  # Normalize to 0-1
        
        # Analyze local uniformity
        kernel_size = 10
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        
        h_local_var = cv2.filter2D(hsv[:, :, 0].astype(np.float32), -1, kernel)
        s_local_var = cv2.filter2D(hsv[:, :, 1].astype(np.float32), -1, kernel)
        v_local_var = cv2.filter2D(hsv[:, :, 2].astype(np.float32), -1, kernel)
        
        local_uniformity = 1.0 / (1.0 + (np.std(h_local_var) + np.std(s_local_var) + np.std(v_local_var)) / 3.0)
        
        return {
            'channel_std_devs': {
                'hue': float(h_std),
                'saturation': float(s_std),
                'value': float(v_std)
            },
            'overall_uniformity': float(uniformity_score),
            'local_uniformity': float(local_uniformity),
            'uniformity_classification': self._classify_uniformity(uniformity_score)
        }
    
    def _calculate_color_statistics(self, rgb: np.ndarray, hsv: np.ndarray, lab: np.ndarray) -> Dict:
        """Calculate comprehensive color statistics"""
        # RGB statistics
        rgb_means = np.mean(rgb, axis=(0, 1))
        rgb_stds = np.std(rgb, axis=(0, 1))
        
        # HSV statistics
        hsv_means = np.mean(hsv, axis=(0, 1))
        hsv_stds = np.std(hsv, axis=(0, 1))
        
        # LAB statistics
        lab_means = np.mean(lab, axis=(0, 1))
        lab_stds = np.std(lab, axis=(0, 1))
        
        # Color temperature estimation
        color_temp = self._estimate_color_temperature(rgb_means)
        
        # Color saturation analysis
        saturation_analysis = self._analyze_saturation(hsv)
        
        return {
            'rgb': {
                'mean': rgb_means.tolist(),
                'std': rgb_stds.tolist(),
                'dominant_channel': ['red', 'green', 'blue'][np.argmax(rgb_means)]
            },
            'hsv': {
                'mean': hsv_means.tolist(),
                'std': hsv_stds.tolist(),
                'dominant_hue_range': self._classify_hue_range(hsv_means[0])
            },
            'lab': {
                'mean': lab_means.tolist(),
                'std': lab_stds.tolist(),
                'colorfulness': self._calculate_colorfulness(lab)
            },
            'color_temperature': color_temp,
            'saturation_analysis': saturation_analysis
        }
    
    def _generate_color_palette(self, dominant_colors: List[Dict]) -> Dict:
        """Generate organized color palette"""
        # Group colors by hue ranges
        hue_groups = {
            'red': [],
            'orange': [],
            'yellow': [],
            'green': [],
            'blue': [],
            'purple': [],
            'pink': [],
            'brown': [],
            'gray': [],
            'black': [],
            'white': []
        }
        
        for color in dominant_colors:
            hue_group = self._classify_hue_group(color['rgb'])
            hue_groups[hue_group].append(color)
        
        # Create palette summary
        palette_summary = {}
        for group_name, colors in hue_groups.items():
            if colors:
                total_percentage = sum(c['percentage'] for c in colors)
                palette_summary[group_name] = {
                    'colors': colors,
                    'total_percentage': float(total_percentage),
                    'dominant_color': max(colors, key=lambda x: x['percentage'])
                }
        
        return {
            'hue_groups': palette_summary,
            'total_colors': len(dominant_colors),
            'dominant_hue_group': max(palette_summary.keys(), 
                                   key=lambda x: palette_summary[x]['total_percentage'] if palette_summary[x] else 0)
        }
    
    def _cross_reference_dataset(self, dominant_colors: List[Dict], food_type: str = None) -> Dict:
        """Cross-reference colors with food dataset"""
        # This would cross-reference with a comprehensive food color database
        # For now, provide a simplified implementation
        
        # Expected colors for different food types
        expected_colors = {
            'apple': ['red', 'green', 'yellow'],
            'banana': ['yellow', 'green', 'brown'],
            'orange': ['orange', 'yellow'],
            'strawberry': ['red', 'pink'],
            'broccoli': ['green'],
            'carrot': ['orange', 'red'],
            'tomato': ['red', 'orange', 'green'],
            'avocado': ['green', 'brown'],
            'lettuce': ['green', 'yellow'],
            'lemon': ['yellow', 'green']
        }
        
        comparison = {
            'expected_colors': expected_colors.get(food_type.lower(), []) if food_type else [],
            'detected_colors': [color['color_name'] for color in dominant_colors[:5]],
            'color_match_score': 0.0,
            'unusual_colors': [],
            'composition_accuracy': 0.0
        }
        
        if food_type and food_type.lower() in expected_colors:
            expected = set(expected_colors[food_type.lower()])
            detected = set(comparison['detected_colors'])
            
            # Calculate match score
            matches = len(expected.intersection(detected))
            total_expected = len(expected)
            comparison['color_match_score'] = matches / total_expected if total_expected > 0 else 0
            
            # Find unusual colors
            comparison['unusual_colors'] = list(detected - expected)
            
            # Calculate composition accuracy
            comparison['composition_accuracy'] = self._calculate_composition_accuracy(
                dominant_colors, expected_colors[food_type.lower()]
            )
        
        return comparison
    
    def _get_color_name(self, rgb: List[int]) -> str:
        """Get color name from RGB values"""
        r, g, b = rgb
        
        # Simple color naming based on RGB values
        if r > 200 and g < 100 and b < 100:
            return 'red'
        elif r > 200 and g > 150 and b < 100:
            return 'orange'
        elif r > 200 and g > 200 and b < 100:
            return 'yellow'
        elif r < 100 and g > 150 and b < 100:
            return 'green'
        elif r < 100 and g < 100 and b > 150:
            return 'blue'
        elif r > 150 and g < 100 and b > 150:
            return 'purple'
        elif r > 200 and g > 150 and b > 150:
            return 'pink'
        elif r > 100 and g > 50 and b < 50:
            return 'brown'
        elif r > 150 and g > 150 and b > 150:
            return 'gray'
        elif r < 50 and g < 50 and b < 50:
            return 'black'
        elif r > 230 and g > 230 and b > 230:
            return 'white'
        else:
            return 'unknown'
    
    def _classify_hue_group(self, rgb: List[int]) -> str:
        """Classify color into hue group"""
        r, g, b = rgb
        
        # Convert to HSV for better hue classification
        hsv = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
        h = hsv[0] * 360  # Convert to degrees
        
        if h < 15 or h >= 345:
            return 'red'
        elif 15 <= h < 45:
            return 'orange'
        elif 45 <= h < 75:
            return 'yellow'
        elif 75 <= h < 165:
            return 'green'
        elif 165 <= h < 255:
            return 'blue'
        elif 255 <= h < 285:
            return 'purple'
        elif 285 <= h < 345:
            return 'pink'
        else:
            return 'unknown'
    
    def _classify_hue_range(self, hue: float) -> str:
        """Classify hue value into range"""
        if hue < 15 or hue >= 165:
            return 'red_range'
        elif 15 <= hue < 45:
            return 'orange_range'
        elif 45 <= hue < 75:
            return 'yellow_range'
        elif 75 <= hue < 165:
            return 'green_range'
        else:
            return 'undefined'
    
    def _assess_spoilage_severity(self, area_ratio: float, spot_count: int) -> str:
        """Assess severity of spoilage indicators"""
        if area_ratio < 0.005:  # Less than 0.5%
            return 'low'
        elif area_ratio < 0.02:  # Less than 2%
            return 'moderate'
        else:
            return 'high'
    
    def _assess_overall_spoilage_risk(self, total_area: float, color_count: int) -> str:
        """Assess overall spoilage risk"""
        if total_area < 0.01 and color_count <= 1:
            return 'low'
        elif total_area < 0.05 and color_count <= 2:
            return 'moderate'
        else:
            return 'high'
    
    def _assess_overall_freshness(self, indicators: List[Dict], food_type: str = None) -> float:
        """Assess overall freshness based on color indicators"""
        if not indicators:
            return 0.3  # Low freshness if no indicators
        
        # Calculate weighted freshness score
        total_score = sum(indicator['freshness_score'] for indicator in indicators)
        avg_score = total_score / len(indicators)
        
        # Adjust based on food type expectations
        if food_type:
            adjustment = self._get_freshness_adjustment(food_type, indicators)
            avg_score = min(1.0, avg_score + adjustment)
        
        return avg_score
    
    def _get_freshness_adjustment(self, food_type: str, indicators: List[Dict]) -> float:
        """Get freshness adjustment based on food type"""
        # Simplified adjustment logic
        food_type = food_type.lower()
        
        if food_type in ['apple', 'strawberry', 'tomato']:
            # Red fruits should have strong red indicators
            red_indicators = [i for i in indicators if 'red' in i['color_type']]
            if red_indicators:
                return 0.1
        elif food_type in ['banana', 'lemon']:
            # Yellow fruits should have strong yellow indicators
            yellow_indicators = [i for i in indicators if 'yellow' in i['color_type']]
            if yellow_indicators:
                return 0.1
        elif food_type in ['broccoli', 'lettuce', 'spinach']:
            # Green vegetables should have strong green indicators
            green_indicators = [i for i in indicators if 'green' in i['color_type']]
            if green_indicators:
                return 0.1
        
        return 0.0
    
    def _classify_uniformity(self, uniformity_score: float) -> str:
        """Classify color uniformity"""
        if uniformity_score > 0.8:
            return 'very_uniform'
        elif uniformity_score > 0.6:
            return 'uniform'
        elif uniformity_score > 0.4:
            return 'moderately_uniform'
        else:
            return 'non_uniform'
    
    def _estimate_color_temperature(self, rgb_means: np.ndarray) -> Dict:
        """Estimate color temperature"""
        r, g, b = rgb_means
        
        # Simplified color temperature estimation
        if r > g and r > b:
            if r - g > 50:
                temp_range = 'warm'
                temp_value = 3000 + (r - g) * 10
            else:
                temp_range = 'neutral_warm'
                temp_value = 4500
        elif b > r and b > g:
            if b - r > 50:
                temp_range = 'cool'
                temp_value = 6500 - (b - r) * 10
            else:
                temp_range = 'neutral_cool'
                temp_value = 5500
        else:
            temp_range = 'neutral'
            temp_value = 5000
        
        return {
            'range': temp_range,
            'estimated_kelvin': int(temp_value)
        }
    
    def _analyze_saturation(self, hsv: np.ndarray) -> Dict:
        """Analyze saturation levels"""
        saturation = hsv[:, :, 1]
        
        saturation_stats = {
            'mean': float(np.mean(saturation)),
            'std': float(np.std(saturation)),
            'min': float(np.min(saturation)),
            'max': float(np.max(saturation)),
            'low_saturation_ratio': float(np.sum(saturation < 50) / saturation.size),
            'high_saturation_ratio': float(np.sum(saturation > 200) / saturation.size)
        }
        
        # Classify saturation level
        mean_sat = saturation_stats['mean']
        if mean_sat < 50:
            saturation_level = 'low'
        elif mean_sat < 150:
            saturation_level = 'medium'
        else:
            saturation_level = 'high'
        
        saturation_stats['level'] = saturation_level
        
        return saturation_stats
    
    def _calculate_colorfulness(self, lab: np.ndarray) -> float:
        """Calculate colorfulness metric"""
        a_channel = lab[:, :, 1]
        b_channel = lab[:, :, 2]
        
        # Calculate colorfulness as standard deviation of a and b channels
        a_std = np.std(a_channel)
        b_std = np.std(b_channel)
        
        colorfulness = np.sqrt(a_std**2 + b_std**2)
        
        return float(colorfulness)
    
    def _calculate_composition_accuracy(self, dominant_colors: List[Dict], expected_colors: List[str]) -> float:
        """Calculate composition accuracy based on expected colors"""
        if not expected_colors:
            return 0.5  # Neutral score if no expectations
        
        # Count detected colors that match expectations
        detected_color_names = [color['color_name'] for color in dominant_colors]
        
        matches = 0
        for expected in expected_colors:
            if expected in detected_color_names:
                # Find the percentage of this color
                matching_colors = [c for c in dominant_colors if c['color_name'] == expected]
                if matching_colors:
                    matches += sum(c['percentage'] for c in matching_colors)
        
        return min(1.0, matches)
    
    def _generate_color_summary(self, dominant_colors: List[Dict], 
                               spoilage_analysis: Dict, 
                               freshness_analysis: Dict) -> Dict:
        """Generate comprehensive color summary"""
        summary = {
            'primary_color': dominant_colors[0] if dominant_colors else None,
            'color_diversity': len(dominant_colors),
            'spoilage_indicators': spoilage_analysis['spoilage_detected'],
            'freshness_indicators': freshness_analysis['fresh_indicators_detected'],
            'overall_assessment': 'unknown'
        }
        
        # Generate overall assessment
        if spoilage_analysis['spoilage_detected']:
            if spoilage_analysis['overall_risk'] == 'high':
                summary['overall_assessment'] = 'high_spoilage_risk'
            else:
                summary['overall_assessment'] = 'some_spoilage_indicators'
        elif freshness_analysis['fresh_indicators_detected']:
            if freshness_analysis['overall_freshness_score'] > 0.7:
                summary['overall_assessment'] = 'very_fresh'
            else:
                summary['overall_assessment'] = 'moderately_fresh'
        else:
            summary['overall_assessment'] = 'neutral'
        
        return summary
