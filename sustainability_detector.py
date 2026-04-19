"""
Sustainability Label Recognition Module
Detects certification labels (Fairtrade, B-Corp, Organic, etc.)
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
import json

class SustainabilityDetector:
    """Detects and analyzes sustainability certification labels"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize sustainability detector
        
        Args:
            model_path: Path to pre-trained model
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Sustainability label database
        self.sustainability_labels = {
            'organic': {
                'usda_organic': {
                    'name': 'USDA Organic',
                    'description': 'Certified organic by US Department of Agriculture',
                    'colors': ['#006633', '#FFFFFF'],
                    'shapes': ['circle', 'seal'],
                    'text_patterns': ['USDA', 'ORGANIC'],
                    'credibility': 'high',
                    'standards': ['no_synthetic_pesticides', 'no_gmos', 'no_irradiation']
                },
                'eu_organic': {
                    'name': 'EU Organic',
                    'description': 'Certified organic by European Union',
                    'colors': ['#006633', '#FFFFFF'],
                    'shapes': ['leaf', 'rectangle'],
                    'text_patterns': ['BIO', 'ORGANIC'],
                    'credibility': 'high',
                    'standards': ['strict_organic_standards', 'traceability']
                }
            },
            'fair_trade': {
                'fairtrade_international': {
                    'name': 'Fairtrade International',
                    'description': 'Fair trade certification for ethical sourcing',
                    'colors': ['#00A650', '#FFFFFF', '#000000'],
                    'shapes': ['circle', 'oval'],
                    'text_patterns': ['FAIRTRADE'],
                    'credibility': 'high',
                    'standards': ['fair_prices', 'ethical_sourcing', 'community_development']
                },
                'fair_trade_usa': {
                    'name': 'Fair Trade USA',
                    'description': 'Fair trade certification for US market',
                    'colors': ['#00A650', '#FFFFFF'],
                    'shapes': ['rectangle', 'seal'],
                    'text_patterns': ['FAIR TRADE'],
                    'credibility': 'high',
                    'standards': ['fair_wages', 'safe_conditions', 'environmental_protection']
                }
            },
            'b_corporation': {
                'b_corp': {
                    'name': 'B Corporation',
                    'description': 'Certified B Corporation for social and environmental performance',
                    'colors': ['#000000', '#FFFFFF'],
                    'shapes': ['rectangle', 'circle'],
                    'text_patterns': ['B CORP', 'CERTIFIED'],
                    'credibility': 'high',
                    'standards': ['social_performance', 'environmental_performance', 'accountability']
                }
            },
            'rainforest_alliance': {
                'rainforest_alliance': {
                    'name': 'Rainforest Alliance',
                    'description': 'Sustainable agriculture certification',
                    'colors': ['#00A650', '#FFFFFF', '#000000'],
                    'shapes': ['frog', 'seal'],
                    'text_patterns': ['RAINFOREST ALLIANCE'],
                    'credibility': 'high',
                    'standards': ['biodiversity_conservation', 'sustainable_livelihoods', 'natural_resources']
                }
            },
            'non_gmo': {
                'non_gmo_project': {
                    'name': 'Non-GMO Project Verified',
                    'description': 'Verified non-GMO by Non-GMO Project',
                    'colors': ['#FF6600', '#000000'],
                    'shapes': ['butterfly', 'seal'],
                    'text_patterns': ['NON-GMO', 'PROJECT'],
                    'credibility': 'high',
                    'standards': ['no_gmos', 'testing_protocol', 'segregation']
                }
            },
            'animal_welfare': {
                'certified_humane': {
                    'name': 'Certified Humane',
                    'description': 'Humane treatment of farm animals',
                    'colors': ['#006633', '#FFFFFF'],
                    'shapes': ['seal', 'rectangle'],
                    'text_patterns': ['CERTIFIED HUMANE'],
                    'credibility': 'high',
                    'standards': ['animal_care', 'no_antibiotics', 'no_hormones']
                },
                'animal_welfare_approved': {
                    'name': 'Animal Welfare Approved',
                    'description': 'High animal welfare standards',
                    'colors': ['#006633', '#FFFFFF'],
                    'shapes': ['seal', 'circle'],
                    'text_patterns': ['ANIMAL WELFARE'],
                    'credibility': 'high',
                    'standards': ['pasture_raised', 'family_farms', 'no_antibiotics']
                }
            },
            'environmental': {
                'leed_certified': {
                    'name': 'LEED Certified',
                    'description': 'Leadership in Energy and Environmental Design',
                    'colors': ['#006633', '#000000'],
                    'shapes': ['rectangle', 'hexagon'],
                    'text_patterns': ['LEED'],
                    'credibility': 'high',
                    'standards': ['energy_efficiency', 'water_efficiency', 'sustainable_materials']
                },
                'energy_star': {
                    'name': 'ENERGY STAR',
                    'description': 'Energy efficiency certification',
                    'colors': ['#0066CC', '#FFFFFF'],
                    'shapes': ['star', 'seal'],
                    'text_patterns': ['ENERGY STAR'],
                    'credibility': 'high',
                    'standards': ['energy_efficiency', 'environmental_protection']
                }
            }
        }
        
        # Common label locations on packaging
        self.label_locations = {
            'front_panel': {'probability': 0.6, 'description': 'Front of packaging'},
            'back_panel': {'probability': 0.3, 'description': 'Back of packaging'},
            'side_panel': {'probability': 0.1, 'description': 'Side of packaging'}
        }
        
        # Label detection parameters
        self.detection_params = {
            'min_label_size': 50,  # pixels
            'max_label_size': 500,  # pixels
            'confidence_threshold': 0.7,
            'nms_threshold': 0.3
        }
        
        # Load models after all attributes are defined
        self.label_detector = self._load_label_detector(model_path)
        self.label_classifier = self._load_label_classifier(model_path)
    
    def _get_all_label_types(self) -> List[str]:
        """Get all label types for classification"""
        label_types = []
        for category, labels in self.sustainability_labels.items():
            for label_name, label_info in labels.items():
                label_types.append(label_name)
        return label_types
    
    def _load_label_detector(self, model_path: Optional[str]) -> nn.Module:
        """Load or create label detection model"""
        if model_path and Path(model_path).exists():
            model = torch.load(model_path, map_location=self.device)
            return model
        
        # Create CNN for label detection
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
            
            # Detection layers
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),  # Single output: label probability
            nn.Sigmoid()
        )
        
        return model.to(self.device)
    
    def _load_label_classifier(self, model_path: Optional[str]) -> nn.Module:
        """Load or create label classification model"""
        if model_path and Path(f"{model_path}_classifier").exists():
            model = torch.load(f"{model_path}_classifier", map_location=self.device)
            return model
        
        # Create CNN for label classification
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
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 50),  # Number of label types (temporarily hardcoded)
            nn.Softmax(dim=1)
        )
        
        return model.to(self.device)
    
    def detect_sustainability_labels(self, image: np.ndarray) -> Dict:
        """
        Detect sustainability labels in image
        
        Args:
            image: Product packaging image
            
        Returns:
            Comprehensive label detection results
        """
        # Detect potential label regions
        label_regions = self._detect_label_regions(image)
        
        # Classify detected labels
        classified_labels = self._classify_detected_labels(image, label_regions)
        
        # Verify label authenticity
        verified_labels = self._verify_label_authenticity(classified_labels)
        
        # Analyze label credibility
        credibility_analysis = self._analyze_label_credibility(verified_labels)
        
        # Generate sustainability score
        sustainability_score = self._calculate_sustainability_score(verified_labels)
        
        # Generate recommendations
        recommendations = self._generate_sustainability_recommendations(
            verified_labels, sustainability_score
        )
        
        return {
            'detected_regions': label_regions,
            'classified_labels': classified_labels,
            'verified_labels': verified_labels,
            'credibility_analysis': credibility_analysis,
            'sustainability_score': sustainability_score,
            'recommendations': recommendations,
            'summary': self._generate_label_summary(verified_labels)
        }
    
    def _detect_label_regions(self, image: np.ndarray) -> List[Dict]:
        """Detect potential label regions in image"""
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Label regions often have specific characteristics:
        # 1. Distinct colors (green, blue, black)
        # 2. Geometric shapes (circles, rectangles, seals)
        # 3. Text content
        # 4. High contrast with background
        
        regions = []
        
        # Method 1: Color-based detection
        color_regions = self._detect_color_based_regions(hsv)
        regions.extend(color_regions)
        
        # Method 2: Shape-based detection
        shape_regions = self._detect_shape_based_regions(gray)
        regions.extend(shape_regions)
        
        # Method 3: Text-based detection
        text_regions = self._detect_text_based_regions(image)
        regions.extend(text_regions)
        
        # Method 4: ML-based detection
        ml_regions = self._detect_ml_based_regions(image)
        regions.extend(ml_regions)
        
        # Combine and filter overlapping regions
        filtered_regions = self._filter_overlapping_regions(regions)
        
        return filtered_regions
    
    def _detect_color_based_regions(self, hsv: np.ndarray) -> List[Dict]:
        """Detect regions based on common label colors"""
        regions = []
        
        # Common sustainability label colors
        label_colors = {
            'green': {'lower': (35, 50, 50), 'upper': (85, 255, 255)},
            'blue': {'lower': (100, 50, 50), 'upper': (130, 255, 255)},
            'black': {'lower': (0, 0, 0), 'upper': (180, 255, 50)},
            'white': {'lower': (0, 0, 200), 'upper': (180, 30, 255)},
            'orange': {'lower': (10, 50, 50), 'upper': (25, 255, 255)}
        }
        
        for color_name, color_range in label_colors.items():
            lower = np.array(color_range['lower'])
            upper = np.array(color_range['upper'])
            
            # Create mask for this color
            mask = cv2.inRange(hsv, lower, upper)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter by size
                if self.detection_params['min_label_size'] < area < self.detection_params['max_label_size']:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    regions.append({
                        'type': 'color_based',
                        'color': color_name,
                        'bbox': [x, y, w, h],
                        'area': area,
                        'contour': contour,
                        'confidence': 0.6
                    })
        
        return regions
    
    def _detect_shape_based_regions(self, gray: np.ndarray) -> List[Dict]:
        """Detect regions based on common label shapes"""
        regions = []
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by size
            if self.detection_params['min_label_size'] < area < self.detection_params['max_label_size']:
                # Analyze shape
                shape_info = self._analyze_shape(contour)
                
                if shape_info['is_regular']:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    regions.append({
                        'type': 'shape_based',
                        'shape': shape_info['shape_type'],
                        'bbox': [x, y, w, h],
                        'area': area,
                        'contour': contour,
                        'shape_info': shape_info,
                        'confidence': 0.5
                    })
        
        return regions
    
    def _detect_text_based_regions(self, image: np.ndarray) -> List[Dict]:
        """Detect regions containing sustainability-related text"""
        regions = []
        
        # This would use OCR in a real implementation
        # For now, use placeholder implementation
        
        # Common sustainability text patterns
        text_patterns = [
            'organic', 'fairtrade', 'b corp', 'certified', 'natural',
            'sustainable', 'eco', 'green', 'bio', 'non-gmo'
        ]
        
        # Simulate text detection (would use pytesseract in real implementation)
        height, width = image.shape[:2]
        
        # Create simulated text regions
        for i, pattern in enumerate(text_patterns):
            # Random placement for simulation
            x = (i * 100) % (width - 100)
            y = (i * 50) % (height - 50)
            w, h = 80, 40
            
            regions.append({
                'type': 'text_based',
                'text_pattern': pattern,
                'bbox': [x, y, w, h],
                'area': w * h,
                'confidence': 0.4
            })
        
        return regions
    
    def _detect_ml_based_regions(self, image: np.ndarray) -> List[Dict]:
        """Detect regions using ML model"""
        regions = []
        
        # Sliding window approach
        height, width = image.shape[:2]
        window_size = 224
        stride = 112
        
        for y in range(0, height - window_size, stride):
            for x in range(0, width - window_size, stride):
                # Extract window
                window = image[y:y+window_size, x:x+window_size]
                
                if window.shape[0] == window_size and window.shape[1] == window_size:
                    # Preprocess
                    input_tensor = self._preprocess_for_detection(window)
                    
                    # Predict
                    self.label_detector.eval()
                    with torch.no_grad():
                        probability = self.label_detector(input_tensor).item()
                    
                    # If high probability, add region
                    if probability > self.detection_params['confidence_threshold']:
                        regions.append({
                            'type': 'ml_based',
                            'bbox': [x, y, window_size, window_size],
                            'area': window_size * window_size,
                            'confidence': probability
                        })
        
        return regions
    
    def _preprocess_for_detection(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for label detection"""
        # Convert to tensor
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        tensor = tensor / 255.0  # Normalize to [0, 1]
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)
    
    def _analyze_shape(self, contour: np.ndarray) -> Dict:
        """Analyze shape of contour"""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            return {'is_regular': False, 'shape_type': 'unknown'}
        
        # Calculate shape metrics
        circularity = 4 * np.pi * area / (perimeter ** 2)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        
        # Determine shape type
        shape_type = 'unknown'
        is_regular = False
        
        if circularity > 0.7:
            shape_type = 'circle'
            is_regular = True
        elif 0.8 < aspect_ratio < 1.2:
            shape_type = 'square'
            is_regular = True
        elif aspect_ratio > 1.5:
            shape_type = 'rectangle'
            is_regular = True
        elif aspect_ratio < 0.7:
            shape_type = 'vertical_rectangle'
            is_regular = True
        
        return {
            'is_regular': is_regular,
            'shape_type': shape_type,
            'circularity': circularity,
            'aspect_ratio': aspect_ratio,
            'area': area,
            'perimeter': perimeter
        }
    
    def _filter_overlapping_regions(self, regions: List[Dict]) -> List[Dict]:
        """Filter overlapping regions using Non-Maximum Suppression"""
        if not regions:
            return []
        
        # Sort by confidence
        regions.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Convert to numpy arrays for NMS
        boxes = np.array([region['bbox'] for region in regions])
        scores = np.array([region['confidence'] for region in regions])
        
        # Apply NMS
        keep_indices = self._non_max_suppression(boxes, scores, self.detection_params['nms_threshold'])
        
        # Filter regions
        filtered_regions = [regions[i] for i in keep_indices]
        
        return filtered_regions
    
    def _non_max_suppression(self, boxes: np.ndarray, scores: np.ndarray, threshold: float) -> List[int]:
        """Non-Maximum Suppression implementation"""
        if len(boxes) == 0:
            return []
        
        # Sort by scores
        indices = np.argsort(scores)[::-1]
        keep = []
        
        while len(indices) > 0:
            # Keep the current box
            current = indices[0]
            keep.append(current)
            
            # Remove current index
            indices = indices[1:]
            
            if len(indices) == 0:
                break
            
            # Calculate IoU with remaining boxes
            current_box = boxes[current]
            remaining_boxes = boxes[indices]
            
            ious = self._calculate_iou(current_box, remaining_boxes)
            
            # Remove boxes with high IoU
            indices = indices[ious < threshold]
        
        return keep
    
    def _calculate_iou(self, box1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """Calculate Intersection over Union"""
        x1 = np.maximum(box1[0], boxes2[:, 0])
        y1 = np.maximum(box1[1], boxes2[:, 1])
        x2 = np.minimum(box1[0] + box1[2], boxes2[:, 0] + boxes2[:, 2])
        y2 = np.minimum(box1[1] + box1[3], boxes2[:, 1] + boxes2[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        area1 = box1[2] * box1[3]
        area2 = boxes2[:, 2] * boxes2[:, 3]
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-6)
    
    def _classify_detected_labels(self, image: np.ndarray, regions: List[Dict]) -> List[Dict]:
        """Classify detected labels"""
        classified_labels = []
        
        for region in regions:
            # Extract region
            x, y, w, h = region['bbox']
            label_image = image[y:y+h, x:x+w]
            
            if label_image.size == 0:
                continue
            
            # Resize for classification
            resized = cv2.resize(label_image, (224, 224))
            
            # Preprocess
            input_tensor = self._preprocess_for_classification(resized)
            
            # Classify
            self.label_classifier.eval()
            with torch.no_grad():
                outputs = self.label_classifier(input_tensor)
                probabilities = outputs.cpu().numpy()[0]
            
            # Get top predictions
            top_indices = np.argsort(probabilities)[-3:][::-1]
            
            predictions = []
            for i, idx in enumerate(top_indices):
                label_type = self._get_all_label_types()[idx]
                probability = probabilities[idx]
                
                if probability > 0.3:  # Minimum confidence threshold
                    predictions.append({
                        'label_type': label_type,
                        'probability': probability,
                        'rank': i + 1
                    })
            
            if predictions:
                classified_labels.append({
                    'region': region,
                    'predictions': predictions,
                    'top_prediction': predictions[0]
                })
        
        return classified_labels
    
    def _preprocess_for_classification(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for label classification"""
        # Convert to tensor
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        tensor = tensor / 255.0  # Normalize to [0, 1]
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)
    
    def _verify_label_authenticity(self, classified_labels: List[Dict]) -> List[Dict]:
        """Verify authenticity of classified labels"""
        verified_labels = []
        
        for label in classified_labels:
            top_prediction = label['top_prediction']
            label_type = top_prediction['label_type']
            
            # Get label information
            label_info = self._get_label_info(label_type)
            
            if label_info:
                # Verify characteristics
                verification_score = self._verify_label_characteristics(
                    label, label_info
                )
                
                # Check for common fakes
                fake_indicators = self._check_fake_indicators(label, label_info)
                
                verified_labels.append({
                    'original_label': label,
                    'label_info': label_info,
                    'verification_score': verification_score,
                    'fake_indicators': fake_indicators,
                    'is_authentic': verification_score > 0.7 and len(fake_indicators) == 0
                })
        
        return verified_labels
    
    def _get_label_info(self, label_type: str) -> Optional[Dict]:
        """Get label information from database"""
        for category, labels in self.sustainability_labels.items():
            for label_key, label_info in labels.items():
                if f"{category}_{label_key}" == label_type:
                    return {
                        'category': category,
                        'key': label_key,
                        **label_info
                    }
        return None
    
    def _verify_label_characteristics(self, label: Dict, label_info: Dict) -> float:
        """Verify label characteristics against expected"""
        verification_score = 0.5  # Base score
        
        region = label['region']
        
        # Check shape
        expected_shapes = label_info.get('shapes', [])
        if 'shape' in region:
            if region['shape'] in expected_shapes:
                verification_score += 0.2
        
        # Check color
        if 'color' in region:
            expected_colors = label_info.get('colors', [])
            # Simplified color check
            verification_score += 0.1
        
        # Check confidence
        confidence = region.get('confidence', 0)
        verification_score += confidence * 0.2
        
        return min(verification_score, 1.0)
    
    def _check_fake_indicators(self, label: Dict, label_info: Dict) -> List[str]:
        """Check for indicators of fake labels"""
        fake_indicators = []
        
        region = label['region']
        
        # Low confidence
        if region.get('confidence', 0) < 0.5:
            fake_indicators.append('low_confidence')
        
        # Unusual shape
        expected_shapes = label_info.get('shapes', [])
        if 'shape' in region and region['shape'] not in expected_shapes:
            fake_indicators.append('unusual_shape')
        
        # Unusual size
        area = region.get('area', 0)
        if area < self.detection_params['min_label_size'] * 0.5:
            fake_indicators.append('unusual_size')
        
        # Multiple conflicting predictions
        predictions = label.get('predictions', [])
        if len(predictions) > 1:
            top_prob = predictions[0]['probability']
            second_prob = predictions[1]['probability']
            if top_prob - second_prob < 0.2:
                fake_indicators.append('conflicting_predictions')
        
        return fake_indicators
    
    def _analyze_label_credibility(self, verified_labels: List[Dict]) -> Dict:
        """Analyze overall credibility of detected labels"""
        if not verified_labels:
            return {
                'overall_credibility': 'none',
                'authentic_labels': 0,
                'suspicious_labels': 0,
                'total_labels': 0
            }
        
        authentic_count = sum(1 for label in verified_labels if label['is_authentic'])
        suspicious_count = len(verified_labels) - authentic_count
        total_count = len(verified_labels)
        
        if authentic_count == 0:
            overall_credibility = 'none'
        elif suspicious_count > authentic_count:
            overall_credibility = 'low'
        elif suspicious_count > 0:
            overall_credibility = 'medium'
        else:
            overall_credibility = 'high'
        
        return {
            'overall_credibility': overall_credibility,
            'authentic_labels': authentic_count,
            'suspicious_labels': suspicious_count,
            'total_labels': total_count,
            'authenticity_percentage': (authentic_count / total_count) * 100 if total_count > 0 else 0
        }
    
    def _calculate_sustainability_score(self, verified_labels: List[Dict]) -> Dict:
        """Calculate overall sustainability score"""
        if not verified_labels:
            return {
                'overall_score': 0,
                'category_scores': {},
                'level': 'none'
            }
        
        # Category scores
        category_scores = {}
        
        for label in verified_labels:
            if label['is_authentic']:
                category = label['label_info']['category']
                if category not in category_scores:
                    category_scores[category] = 0
                category_scores[category] += 1
        
        # Normalize scores
        max_labels_per_category = 3  # Maximum expected labels per category
        for category in category_scores:
            category_scores[category] = min(category_scores[category] / max_labels_per_category, 1.0)
        
        # Calculate overall score
        if category_scores:
            overall_score = np.mean(list(category_scores.values()))
        else:
            overall_score = 0
        
        # Determine level
        if overall_score >= 0.8:
            level = 'excellent'
        elif overall_score >= 0.6:
            level = 'good'
        elif overall_score >= 0.4:
            level = 'moderate'
        elif overall_score >= 0.2:
            level = 'limited'
        else:
            level = 'none'
        
        return {
            'overall_score': overall_score,
            'category_scores': category_scores,
            'level': level
        }
    
    def _generate_sustainability_recommendations(self, verified_labels: List[Dict], 
                                             sustainability_score: Dict) -> List[str]:
        """Generate sustainability recommendations"""
        recommendations = []
        
        score = sustainability_score.get('overall_score', 0)
        level = sustainability_score.get('level', 'none')
        
        if level == 'none':
            recommendations.append("Look for sustainability certifications")
            recommendations.append("Consider products with verified environmental claims")
        elif level == 'limited':
            recommendations.append("Some sustainability efforts detected")
            recommendations.append("Look for additional certifications")
        elif level == 'moderate':
            recommendations.append("Good sustainability practices")
            recommendations.append("Support brands with multiple certifications")
        elif level == 'good':
            recommendations.append("Strong sustainability commitment")
            recommendations.append("Excellent choice for conscious consumers")
        elif level == 'excellent':
            recommendations.append("Outstanding sustainability leadership")
            recommendations.append("Industry-leading environmental practices")
        
        # Category-specific recommendations
        category_scores = sustainability_score.get('category_scores', {})
        
        if 'organic' in category_scores and category_scores['organic'] < 0.5:
            recommendations.append("Consider organic alternatives")
        
        if 'fair_trade' in category_scores and category_scores['fair_trade'] < 0.5:
            recommendations.append("Look for fair trade certifications")
        
        if 'animal_welfare' in category_scores and category_scores['animal_welfare'] < 0.5:
            recommendations.append("Consider animal welfare certified products")
        
        return recommendations
    
    def _generate_label_summary(self, verified_labels: List[Dict]) -> Dict:
        """Generate summary of detected labels"""
        if not verified_labels:
            return {
                'total_labels': 0,
                'categories': {},
                'top_certifications': [],
                'summary': 'No sustainability labels detected'
            }
        
        # Count by category
        categories = {}
        all_labels = []
        
        for label in verified_labels:
            if label['is_authentic']:
                category = label['label_info']['category']
                if category not in categories:
                    categories[category] = []
                
                label_name = label['label_info']['name']
                categories[category].append(label_name)
                all_labels.append(label_name)
        
        # Get top certifications
        top_certifications = all_labels[:5] if all_labels else []
        
        # Generate summary text
        if len(all_labels) == 0:
            summary = 'No verified sustainability labels detected'
        elif len(all_labels) == 1:
            summary = f'Detected {all_labels[0]} certification'
        else:
            summary = f'Detected {len(all_labels)} sustainability certifications: {", ".join(all_labels[:3])}'
        
        return {
            'total_labels': len(all_labels),
            'categories': categories,
            'top_certifications': top_certifications,
            'summary': summary
        }
