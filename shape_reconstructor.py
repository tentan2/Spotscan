"""
Shape Reconstructor Module
Generates 3D models of food items for improved object detection accuracy
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
from pathlib import Path
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
from scipy.interpolate import griddata

class ShapeReconstructor:
    """Reconstructs 3D shapes from 2D food images"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize shape reconstructor
        
        Args:
            model_path: Path to pre-trained model
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.depth_estimator = self._load_depth_model(model_path)
        
        # Shape templates for common food items
        self.shape_templates = self._load_shape_templates()
        
    def _load_depth_model(self, model_path: Optional[str]) -> nn.Module:
        """Load or create depth estimation model"""
        if model_path and Path(model_path).exists():
            model = torch.load(model_path, map_location=self.device)
            return model
        
        # Create a simple depth estimation network
        model = nn.Sequential(
            # Encoder
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Decoder
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        return model.to(self.device)
    
    def _load_shape_templates(self) -> Dict:
        """Load shape templates for common food items"""
        return {
            'apple': {
                'category': 'sphere',
                'aspect_ratio_range': (0.8, 1.2),
                'symmetry': 'radial',
                'typical_size': {'min': 50, 'max': 100}  # mm
            },
            'banana': {
                'category': 'ellipsoid',
                'aspect_ratio_range': (2.0, 4.0),
                'symmetry': 'bilateral',
                'typical_size': {'min': 100, 'max': 200}  # mm
            },
            'orange': {
                'category': 'sphere',
                'aspect_ratio_range': (0.9, 1.1),
                'symmetry': 'radial',
                'typical_size': {'min': 60, 'max': 90}  # mm
            },
            'strawberry': {
                'category': 'cone',
                'aspect_ratio_range': (1.0, 2.0),
                'symmetry': 'radial',
                'typical_size': {'min': 20, 'max': 40}  # mm
            },
            'carrot': {
                'category': 'cylinder',
                'aspect_ratio_range': (3.0, 8.0),
                'symmetry': 'bilateral',
                'typical_size': {'min': 80, 'max': 200}  # mm
            },
            'broccoli': {
                'category': 'complex',
                'aspect_ratio_range': (0.8, 1.5),
                'symmetry': 'irregular',
                'typical_size': {'min': 80, 'max': 150}  # mm
            },
            'tomato': {
                'category': 'sphere',
                'aspect_ratio_range': (0.8, 1.3),
                'symmetry': 'radial',
                'typical_size': {'min': 40, 'max': 80}  # mm
            },
            'avocado': {
                'category': 'ellipsoid',
                'aspect_ratio_range': (1.2, 2.0),
                'symmetry': 'bilateral',
                'typical_size': {'min': 70, 'max': 120}  # mm
            }
        }
    
    def reconstruct_3d_shape(self, image: np.ndarray, food_class: str) -> Dict:
        """
        Reconstruct 3D shape from 2D image
        
        Args:
            image: 2D food image
            food_class: Detected food class
            
        Returns:
            3D reconstruction results
        """
        # Estimate depth map
        depth_map = self._estimate_depth(image)
        
        # Extract silhouette
        silhouette = self._extract_silhouette(image)
        
        # Generate 3D point cloud
        point_cloud = self._generate_point_cloud(image, depth_map, silhouette)
        
        # Create mesh from point cloud
        mesh = self._create_mesh(point_cloud)
        
        # Analyze shape properties
        shape_properties = self._analyze_shape_properties(point_cloud, mesh)
        
        # Validate against food template
        template_validation = self._validate_against_template(
            shape_properties, food_class
        )
        
        # Generate visualization data
        visualization_data = self._generate_visualization_data(
            point_cloud, mesh, food_class
        )
        
        return {
            'food_class': food_class,
            'depth_map': depth_map,
            'silhouette': silhouette,
            'point_cloud': point_cloud,
            'mesh': mesh,
            'shape_properties': shape_properties,
            'template_validation': template_validation,
            'visualization_data': visualization_data,
            'confidence_score': template_validation.get('confidence', 0.5)
        }
    
    def _estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """Estimate depth map from 2D image"""
        # Preprocess image
        input_tensor = self._preprocess_for_depth(image)
        
        # Set model to evaluation mode
        self.depth_estimator.eval()
        
        # Perform inference
        with torch.no_grad():
            depth_tensor = self.depth_estimator(input_tensor)
            depth_map = depth_tensor.squeeze().cpu().numpy()
        
        # Post-process depth map
        depth_map = self._postprocess_depth(depth_map)
        
        return depth_map
    
    def _preprocess_for_depth(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for depth estimation"""
        # Resize to model input size
        resized = cv2.resize(image, (224, 224))
        
        # Convert to tensor
        tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float()
        tensor = tensor / 255.0  # Normalize to [0, 1]
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)
    
    def _postprocess_depth(self, depth_map: np.ndarray) -> np.ndarray:
        """Post-process depth map"""
        # Scale to reasonable depth range (0-100mm)
        depth_map = depth_map * 100
        
        # Apply Gaussian smoothing
        depth_map = cv2.GaussianBlur(depth_map, (5, 5), 0)
        
        # Resize to original image size (assuming square for now)
        depth_map = cv2.resize(depth_map, (224, 224))
        
        return depth_map
    
    def _extract_silhouette(self, image: np.ndarray) -> np.ndarray:
        """Extract silhouette from image"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to create binary mask
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Clean up with morphological operations
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        return thresh
    
    def _generate_point_cloud(self, image: np.ndarray, depth_map: np.ndarray, 
                            silhouette: np.ndarray) -> np.ndarray:
        """Generate 3D point cloud from depth map and silhouette"""
        height, width = depth_map.shape
        
        # Create coordinate grids
        x = np.linspace(-width/2, width/2, width)
        y = np.linspace(-height/2, height/2, height)
        xx, yy = np.meshgrid(x, y)
        
        # Apply silhouette mask
        mask = silhouette > 0
        
        # Extract 3D points where silhouette is present
        points = []
        for i in range(height):
            for j in range(width):
                if mask[i, j]:
                    x_coord = xx[i, j]
                    y_coord = yy[i, j]
                    z_coord = depth_map[i, j]
                    points.append([x_coord, y_coord, z_coord])
        
        return np.array(points)
    
    def _create_mesh(self, point_cloud: np.ndarray) -> Dict:
        """Create mesh from point cloud"""
        if len(point_cloud) < 4:
            return {'error': 'Insufficient points for mesh creation'}
        
        try:
            # Create convex hull
            hull = ConvexHull(point_cloud)
            
            # Extract vertices and faces
            vertices = point_cloud[hull.vertices]
            faces = hull.simplices
            
            return {
                'vertices': vertices,
                'faces': faces,
                'num_vertices': len(vertices),
                'num_faces': len(faces),
                'volume': hull.volume,
                'surface_area': hull.area
            }
        except Exception as e:
            return {'error': f'Mesh creation failed: {str(e)}'}
    
    def _analyze_shape_properties(self, point_cloud: np.ndarray, mesh: Dict) -> Dict:
        """Analyze shape properties"""
        if len(point_cloud) == 0:
            return {'error': 'No point cloud data'}
        
        # Basic statistics
        centroid = np.mean(point_cloud, axis=0)
        
        # Bounding box
        min_coords = np.min(point_cloud, axis=0)
        max_coords = np.max(point_cloud, axis=0)
        dimensions = max_coords - min_coords
        
        # Aspect ratios
        aspect_ratios = {
            'xy': dimensions[0] / dimensions[1] if dimensions[1] > 0 else 0,
            'xz': dimensions[0] / dimensions[2] if dimensions[2] > 0 else 0,
            'yz': dimensions[1] / dimensions[2] if dimensions[2] > 0 else 0
        }
        
        # Shape complexity
        if 'surface_area' in mesh and 'volume' in mesh:
            surface_to_volume = mesh['surface_area'] / mesh['volume'] if mesh['volume'] > 0 else 0
        else:
            surface_to_volume = 0
        
        # Symmetry analysis
        symmetry_scores = self._analyze_symmetry(point_cloud, centroid)
        
        # Compactness
        max_dimension = np.max(dimensions)
        compactness = mesh.get('volume', 0) / (max_dimension ** 3) if max_dimension > 0 else 0
        
        return {
            'centroid': centroid.tolist(),
            'dimensions': dimensions.tolist(),
            'aspect_ratios': {k: float(v) for k, v in aspect_ratios.items()},
            'surface_to_volume_ratio': float(surface_to_volume),
            'symmetry_scores': symmetry_scores,
            'compactness': float(compactness),
            'num_points': len(point_cloud)
        }
    
    def _analyze_symmetry(self, point_cloud: np.ndarray, centroid: np.ndarray) -> Dict:
        """Analyze symmetry of point cloud"""
        # Center the point cloud
        centered = point_cloud - centroid
        
        # Check reflection symmetry across planes
        symmetry_scores = {}
        
        # XY plane symmetry (top-bottom)
        xy_reflected = centered.copy()
        xy_reflected[:, 2] = -xy_reflected[:, 2]
        xy_symmetry = self._calculate_symmetry_score(centered, xy_reflected)
        
        # XZ plane symmetry (left-right)
        xz_reflected = centered.copy()
        xz_reflected[:, 1] = -xz_reflected[:, 1]
        xz_symmetry = self._calculate_symmetry_score(centered, xz_reflected)
        
        # YZ plane symmetry (front-back)
        yz_reflected = centered.copy()
        yz_reflected[:, 0] = -yz_reflected[:, 0]
        yz_symmetry = self._calculate_symmetry_score(centered, yz_reflected)
        
        # Rotational symmetry (around Z axis)
        rotational_symmetry = self._calculate_rotational_symmetry(centered)
        
        return {
            'xy_plane': float(xy_symmetry),
            'xz_plane': float(xz_symmetry),
            'yz_plane': float(yz_symmetry),
            'rotational_z': float(rotational_symmetry)
        }
    
    def _calculate_symmetry_score(self, original: np.ndarray, reflected: np.ndarray) -> float:
        """Calculate symmetry score between original and reflected points"""
        # Use nearest neighbor matching
        from scipy.spatial import cKDTree
        
        tree = cKDTree(reflected)
        distances, _ = tree.query(original, k=1)
        
        # Calculate symmetry score (lower distances = higher symmetry)
        mean_distance = np.mean(distances)
        max_distance = np.max(np.linalg.norm(original, axis=1))
        
        symmetry_score = 1.0 - (mean_distance / max_distance) if max_distance > 0 else 0
        return max(0, symmetry_score)
    
    def _calculate_rotational_symmetry(self, centered: np.ndarray) -> float:
        """Calculate rotational symmetry around Z axis"""
        # Check symmetry at different angles
        angles = [90, 180, 270]  # degrees
        symmetry_scores = []
        
        for angle in angles:
            # Rotate points around Z axis
            theta = np.radians(angle)
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
            
            rotated = centered @ rotation_matrix.T
            symmetry_score = self._calculate_symmetry_score(centered, rotated)
            symmetry_scores.append(symmetry_score)
        
        # Return maximum symmetry score
        return max(symmetry_scores) if symmetry_scores else 0
    
    def _validate_against_template(self, shape_properties: Dict, food_class: str) -> Dict:
        """Validate reconstructed shape against food template"""
        template = self.shape_templates.get(food_class.lower())
        
        if not template:
            return {
                'valid': False,
                'confidence': 0.3,
                'error': f'No template available for {food_class}'
            }
        
        validation_results = {
            'valid': True,
            'confidence': 0.5,
            'matches': {},
            'violations': []
        }
        
        # Check aspect ratio
        if 'aspect_ratios' in shape_properties:
            xy_ratio = shape_properties['aspect_ratios']['xy']
            min_ratio, max_ratio = template['aspect_ratio_range']
            
            if min_ratio <= xy_ratio <= max_ratio:
                validation_results['matches']['aspect_ratio'] = True
                validation_results['confidence'] += 0.1
            else:
                validation_results['violations'].append(
                    f'Aspect ratio {xy_ratio:.2f} outside expected range {min_ratio}-{max_ratio}'
                )
                validation_results['confidence'] -= 0.1
        
        # Check symmetry
        if 'symmetry_scores' in shape_properties:
            expected_symmetry = template['symmetry']
            
            if expected_symmetry == 'radial':
                # High rotational symmetry expected
                rot_sym = shape_properties['symmetry_scores']['rotational_z']
                if rot_sym > 0.7:
                    validation_results['matches']['symmetry'] = True
                    validation_results['confidence'] += 0.1
                else:
                    validation_results['violations'].append(
                        f'Low rotational symmetry {rot_sym:.2f} for radial shape'
                    )
                    validation_results['confidence'] -= 0.1
            
            elif expected_symmetry == 'bilateral':
                # High plane symmetry expected
                plane_syms = [
                    shape_properties['symmetry_scores']['xy_plane'],
                    shape_properties['symmetry_scores']['xz_plane'],
                    shape_properties['symmetry_scores']['yz_plane']
                ]
                max_plane_sym = max(plane_syms)
                
                if max_plane_sym > 0.7:
                    validation_results['matches']['symmetry'] = True
                    validation_results['confidence'] += 0.1
                else:
                    validation_results['violations'].append(
                        f'Low bilateral symmetry {max_plane_sym:.2f}'
                    )
                    validation_results['confidence'] -= 0.1
        
        # Check size (if we have calibration info)
        if 'dimensions' in shape_properties:
            max_dim = max(shape_properties['dimensions'])
            typical_size = template['typical_size']
            
            # This would require camera calibration for accurate sizing
            # For now, just note that size check is pending
            validation_results['size_check'] = 'pending_calibration'
        
        # Clamp confidence to [0, 1]
        validation_results['confidence'] = max(0, min(1, validation_results['confidence']))
        
        # Determine if validation passed
        validation_results['valid'] = validation_results['confidence'] > 0.5
        
        return validation_results
    
    def _generate_visualization_data(self, point_cloud: np.ndarray, mesh: Dict, 
                                  food_class: str) -> Dict:
        """Generate data for 3D visualization"""
        visualization = {
            'point_cloud_data': {
                'points': point_cloud.tolist(),
                'num_points': len(point_cloud)
            },
            'mesh_data': mesh,
            'food_class': food_class,
            'rendering_hints': self._get_rendering_hints(food_class)
        }
        
        # Generate multiple viewing angles
        if len(point_cloud) > 0:
            visualization['view_angles'] = self._generate_view_angles(point_cloud)
        
        return visualization
    
    def _get_rendering_hints(self, food_class: str) -> Dict:
        """Get rendering hints for specific food types"""
        hints = {
            'default': {
                'color': '#FF6B6B',
                'opacity': 0.8,
                'wireframe': False,
                'lighting': True
            }
        }
        
        # Food-specific colors
        food_colors = {
            'apple': '#FF4444',
            'banana': '#FFE135',
            'orange': '#FFA500',
            'strawberry': '#FF1744',
            'carrot': '#FF6B35',
            'broccoli': '#4CAF50',
            'tomato': '#FF5722',
            'avocado': '#689F38'
        }
        
        if food_class.lower() in food_colors:
            hints['default']['color'] = food_colors[food_class.lower()]
        
        return hints
    
    def _generate_view_angles(self, point_cloud: np.ndarray) -> List[Dict]:
        """Generate multiple viewing angles for visualization"""
        views = []
        
        # Standard viewing angles
        angles = [
            {'azimuth': 0, 'elevation': 0, 'name': 'front'},
            {'azimuth': 90, 'elevation': 0, 'name': 'side'},
            {'azimuth': 45, 'elevation': 30, 'name': 'isometric'},
            {'azimuth': 0, 'elevation': 90, 'name': 'top'},
            {'azimuth': 180, 'elevation': 0, 'name': 'back'}
        ]
        
        for angle in angles:
            views.append({
                'azimuth': angle['azimuth'],
                'elevation': angle['elevation'],
                'name': angle['name']
            })
        
        return views
    
    def try_again_reconstruction(self, image: np.ndarray, food_class: str, 
                              previous_result: Dict) -> Dict:
        """
        Attempt reconstruction with different parameters
        
        Args:
            image: Input image
            food_class: Food class
            previous_result: Previous reconstruction result
            
        Returns:
            New reconstruction result
        """
        # Try different depth estimation parameters
        new_result = self.reconstruct_3d_shape(image, food_class)
        
        # Compare with previous result
        if previous_result.get('confidence_score', 0) < new_result.get('confidence_score', 0):
            return new_result
        else:
            # Try alternative reconstruction method
            return self._alternative_reconstruction(image, food_class)
    
    def _alternative_reconstruction(self, image: np.ndarray, food_class: str) -> Dict:
        """Alternative reconstruction method using different approach"""
        # Use template-based reconstruction
        template = self.shape_templates.get(food_class.lower())
        
        if not template:
            return {'error': 'No template available for alternative reconstruction'}
        
        # Extract silhouette
        silhouette = self._extract_silhouette(image)
        
        # Generate synthetic point cloud based on template
        point_cloud = self._generate_template_point_cloud(silhouette, template)
        
        # Create mesh
        mesh = self._create_mesh(point_cloud)
        
        # Analyze properties
        shape_properties = self._analyze_shape_properties(point_cloud, mesh)
        
        return {
            'food_class': food_class,
            'point_cloud': point_cloud,
            'mesh': mesh,
            'shape_properties': shape_properties,
            'template_validation': {'valid': True, 'confidence': 0.6},
            'reconstruction_method': 'template_based'
        }
    
    def _generate_template_point_cloud(self, silhouette: np.ndarray, 
                                     template: Dict) -> np.ndarray:
        """Generate point cloud based on template shape"""
        height, width = silhouette.shape
        
        # Find bounding box of silhouette
        y_coords, x_coords = np.where(silhouette > 0)
        if len(x_coords) == 0:
            return np.array([])
        
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        
        # Generate points based on template category
        category = template['category']
        
        if category == 'sphere':
            point_cloud = self._generate_sphere_points(x_min, x_max, y_min, y_max)
        elif category == 'ellipsoid':
            point_cloud = self._generate_ellipsoid_points(x_min, x_max, y_min, y_max)
        elif category == 'cylinder':
            point_cloud = self._generate_cylinder_points(x_min, x_max, y_min, y_max)
        elif category == 'cone':
            point_cloud = self._generate_cone_points(x_min, x_max, y_min, y_max)
        else:
            # Default to generic shape
            point_cloud = self._generate_generic_points(x_coords, y_coords)
        
        return point_cloud
    
    def _generate_sphere_points(self, x_min: int, x_max: int, y_min: int, y_max: int) -> np.ndarray:
        """Generate sphere point cloud"""
        # Create sphere with radius based on silhouette size
        radius = min((x_max - x_min), (y_max - y_min)) / 2
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        
        points = []
        num_points = 1000
        
        for _ in range(num_points):
            # Generate random point on sphere
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)
            
            x = radius * np.sin(phi) * np.cos(theta) + center_x
            y = radius * np.sin(phi) * np.sin(theta) + center_y
            z = radius * np.cos(phi)
            
            points.append([x, y, z])
        
        return np.array(points)
    
    def _generate_ellipsoid_points(self, x_min: int, x_max: int, y_min: int, y_max: int) -> np.ndarray:
        """Generate ellipsoid point cloud"""
        width = x_max - x_min
        height = y_max - y_min
        
        # Ellipsoid radii
        a = width / 2  # x-axis radius
        b = height / 2  # y-axis radius
        c = min(a, b)  # z-axis radius (depth)
        
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        
        points = []
        num_points = 1000
        
        for _ in range(num_points):
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)
            
            x = a * np.sin(phi) * np.cos(theta) + center_x
            y = b * np.sin(phi) * np.sin(theta) + center_y
            z = c * np.cos(phi)
            
            points.append([x, y, z])
        
        return np.array(points)
    
    def _generate_cylinder_points(self, x_min: int, x_max: int, y_min: int, y_max: int) -> np.ndarray:
        """Generate cylinder point cloud"""
        radius = min((x_max - x_min), (y_max - y_min)) / 4
        height = max((x_max - x_min), (y_max - y_min))
        
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        
        points = []
        num_points = 1000
        
        for _ in range(num_points):
            theta = np.random.uniform(0, 2 * np.pi)
            z = np.random.uniform(-height/2, height/2)
            
            x = radius * np.cos(theta) + center_x
            y = radius * np.sin(theta) + center_y
            
            points.append([x, y, z])
        
        return np.array(points)
    
    def _generate_cone_points(self, x_min: int, x_max: int, y_min: int, y_max: int) -> np.ndarray:
        """Generate cone point cloud"""
        base_radius = min((x_max - x_min), (y_max - y_min)) / 2
        height = max((x_max - x_min), (y_max - y_min))
        
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        
        points = []
        num_points = 1000
        
        for _ in range(num_points):
            theta = np.random.uniform(0, 2 * np.pi)
            z = np.random.uniform(0, height)
            
            # Radius decreases linearly with height
            r = base_radius * (1 - z / height)
            
            x = r * np.cos(theta) + center_x
            y = r * np.sin(theta) + center_y
            
            points.append([x, y, z])
        
        return np.array(points)
    
    def _generate_generic_points(self, x_coords: np.ndarray, y_coords: np.ndarray) -> np.ndarray:
        """Generate generic point cloud from silhouette points"""
        points = []
        
        for x, y in zip(x_coords, y_coords):
            # Add some random depth
            z = np.random.uniform(0, 50)
            points.append([float(x), float(y), z])
        
        return np.array(points)
