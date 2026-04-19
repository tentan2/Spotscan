"""
Input Methods Module
Handles multiple input methods: camera, video, image upload, web images
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import requests
from typing import Optional, List, Tuple
import tempfile
import os
from pathlib import Path

class InputMethodManager:
    """Manages multiple input methods for the Spotscan application"""
    
    def __init__(self):
        """Initialize input method manager"""
        self.supported_formats = ['jpg', 'jpeg', 'png', 'webp', 'bmp', 'tiff']
        self.max_file_size = 50 * 1024 * 1024  # 50MB
        self.temp_dir = tempfile.mkdtemp()
    
    def handle_image_upload(self) -> Optional[np.ndarray]:
        """Handle image upload with enhanced features"""
        st.subheader("Upload Image")
        
        # File uploader with enhanced options
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=self.supported_formats,
            help=f"Supported formats: {', '.join(self.supported_formats)}. Max size: 50MB",
            key="image_uploader"
        )
        
        if uploaded_file is not None:
            # Validate file size
            if hasattr(uploaded_file, 'size') and uploaded_file.size > self.max_file_size:
                st.error("File too large. Maximum size is 50MB.")
                return None
            
            try:
                # Read and process image
                image = Image.open(uploaded_file)
                
                # Display image info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Format", uploaded_file.type.split('/')[-1].upper())
                with col2:
                    st.metric("Size", f"{uploaded_file.size / (1024*1024):.1f} MB")
                with col3:
                    st.metric("Dimensions", f"{image.width}x{image.height}")
                
                # Image preview
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Convert to numpy array
                image_array = np.array(image)
                
                # Convert RGB to BGR for OpenCV
                if len(image_array.shape) == 3:
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                
                # Image enhancement options
                if st.checkbox("Enhance image before analysis"):
                    image_array = self.enhance_image(image_array)
                
                return image_array
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                return None
        
        return None
    
    def handle_camera_input(self) -> Optional[np.ndarray]:
        """Handle camera input with enhanced features"""
        st.subheader("Camera Capture")
        
        # Camera input
        camera_image = st.camera_input("Take a picture")
        
        if camera_image is not None:
            try:
                # Convert to numpy array
                image_array = np.array(camera_image)
                
                # Convert RGB to BGR for OpenCV
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                
                # Display camera info
                st.success(f"Image captured: {image_array.shape[1]}x{image_array.shape[0]} pixels")
                
                # Camera-specific enhancements
                if st.checkbox("Enhance camera image"):
                    image_array = self.enhance_camera_image(image_array)
                
                return image_array
                
            except Exception as e:
                st.error(f"Error processing camera image: {str(e)}")
                return None
        
        return None
    
    def handle_web_image(self) -> Optional[np.ndarray]:
        """Handle web image from URL"""
        st.subheader("Web Image")
        
        url = st.text_input("Enter image URL:", placeholder="https://example.com/image.jpg")
        
        if url and st.button("Load Image"):
            try:
                # Validate URL
                if not self.is_valid_image_url(url):
                    st.error("Invalid image URL")
                    return None
                
                # Download image
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                
                # Check content type
                content_type = response.headers.get('content-type', '')
                if not content_type.startswith('image/'):
                    st.error("URL does not point to an image")
                    return None
                
                # Load image
                image = Image.open(io.BytesIO(response.content))
                
                # Display image info
                st.success(f"Image loaded from {url}")
                st.info(f"Size: {len(response.content) / (1024*1024):.1f} MB")
                
                # Image preview
                st.image(image, caption="Web Image", use_column_width=True)
                
                # Convert to numpy array
                image_array = np.array(image)
                
                # Convert RGB to BGR for OpenCV
                if len(image_array.shape) == 3:
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                
                return image_array
                
            except requests.RequestException as e:
                st.error(f"Error downloading image: {str(e)}")
                return None
            except Exception as e:
                st.error(f"Error processing web image: {str(e)}")
                return None
        
        return None
    
    def handle_video_input(self) -> Optional[List[np.ndarray]]:
        """Handle video input and frame extraction"""
        st.subheader("Video Analysis")
        
        uploaded_video = st.file_uploader(
            "Upload video",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video to extract frames for analysis"
        )
        
        if uploaded_video is not None:
            try:
                # Save video temporarily
                temp_path = os.path.join(self.temp_dir, f"temp_video.{uploaded_video.type.split('/')[-1]}")
                with open(temp_path, "wb") as f:
                    f.write(uploaded_video.getbuffer())
                
                # Extract frames
                frames = self.extract_video_frames(temp_path)
                
                if frames:
                    st.success(f"Extracted {len(frames)} frames from video")
                    
                    # Frame selection options
                    frame_selection = st.selectbox(
                        "Select frame for analysis:",
                        ["First frame", "Middle frame", "Last frame", "Best quality frame"]
                    )
                    
                    selected_frame = self.select_frame(frames, frame_selection)
                    
                    if selected_frame is not None:
                        # Display selected frame
                        st.image(selected_frame, caption="Selected Frame", use_column_width=True)
                        return [selected_frame]
                
                # Multiple frames analysis option
                if st.checkbox("Analyze multiple frames"):
                    num_frames = st.slider("Number of frames to analyze", 1, min(10, len(frames)), 3)
                    selected_frames = frames[::max(1, len(frames)//num_frames)][:num_frames]
                    return selected_frames
                
            except Exception as e:
                st.error(f"Error processing video: {str(e)}")
                return None
        
        return None
    
    def handle_sample_images(self) -> Optional[np.ndarray]:
        """Handle sample image selection"""
        st.subheader("Sample Images")
        
        # Sample image categories
        categories = {
            "Fruits": ["apple", "banana", "orange", "strawberry"],
            "Vegetables": ["carrot", "broccoli", "tomato", "lettuce"],
            "Grains": ["bread", "rice", "pasta", "cereal"],
            "Proteins": ["chicken", "beef", "fish", "eggs"],
            "Beverages": ["coffee", "juice", "milk", "water"]
        }
        
        category = st.selectbox("Select category:", list(categories.keys()))
        
        if category:
            food_items = categories[category]
            food_item = st.selectbox("Select food item:", food_items)
            
            if food_item and st.button("Load Sample"):
                # This would load from a sample image database
                # For now, create a placeholder
                st.info(f"Sample image for {food_item} would be loaded here")
                
                # Create a placeholder image
                placeholder = np.zeros((400, 400, 3), dtype=np.uint8)
                placeholder[:] = (100, 150, 200)  # Blue placeholder
                
                # Add text
                cv2.putText(placeholder, food_item.upper(), (100, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                st.image(placeholder, caption=f"Sample: {food_item}", use_column_width=True)
                
                return placeholder
        
        return None
    
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Enhance image with various options"""
        st.subheader("Image Enhancement")
        
        # Enhancement options
        col1, col2 = st.columns(2)
        
        with col1:
            brightness = st.slider("Brightness", -50, 50, 0)
            contrast = st.slider("Contrast", -50, 50, 0)
        
        with col2:
            saturation = st.slider("Saturation", -50, 50, 0)
            sharpness = st.slider("Sharpness", 0, 100, 0)
        
        # Apply enhancements
        enhanced = image.copy()
        
        # Brightness and contrast
        if brightness != 0 or contrast != 0:
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1 + contrast/100, beta=brightness)
        
        # Saturation
        if saturation != 0:
            hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] + saturation, 0, 255)
            enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Sharpness
        if sharpness > 0:
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel * (sharpness / 100))
        
        return enhanced
    
    def enhance_camera_image(self, image: np.ndarray) -> np.ndarray:
        """Enhance camera-captured image"""
        # Camera-specific enhancements
        # Reduce noise, improve contrast, auto-white balance
        
        # Denoise
        denoised = cv2.fastNlMeansDenoisingColored(image)
        
        # Auto white balance (simplified)
        result = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        avg_a = np.mean(result[:, :, 1])
        avg_b = np.mean(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - (avg_a - 128)
        result[:, :, 2] = result[:, :, 2] - (avg_b - 128)
        enhanced = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def is_valid_image_url(self, url: str) -> bool:
        """Validate if URL is likely to point to an image"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']
        
        # Check file extension
        for ext in image_extensions:
            if url.lower().endswith(ext):
                return True
        
        # Check common image hosting patterns
        image_hosts = ['i.imgur.com', 'images.unsplash.com', 'picsum.photos', 'placeimg.com']
        for host in image_hosts:
            if host in url.lower():
                return True
        
        return False
    
    def extract_video_frames(self, video_path: str) -> List[np.ndarray]:
        """Extract frames from video file"""
        frames = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                st.error("Could not open video file")
                return frames
            
            frame_count = 0
            success, frame = cap.read()
            
            while success and frame_count < 100:  # Limit to 100 frames
                frames.append(frame)
                success, frame = cap.read()
                frame_count += 1
            
            cap.release()
            
        except Exception as e:
            st.error(f"Error extracting frames: {str(e)}")
        
        return frames
    
    def select_frame(self, frames: List[np.ndarray], selection_type: str) -> Optional[np.ndarray]:
        """Select frame based on selection type"""
        if not frames:
            return None
        
        if selection_type == "First frame":
            return frames[0]
        elif selection_type == "Middle frame":
            return frames[len(frames) // 2]
        elif selection_type == "Last frame":
            return frames[-1]
        elif selection_type == "Best quality frame":
            # Select frame with highest quality (simple metric: highest variance)
            best_frame = None
            best_score = 0
            
            for frame in frames:
                # Calculate quality score (variance as a simple metric)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                score = np.var(gray)
                
                if score > best_score:
                    best_score = score
                    best_frame = frame
            
            return best_frame
        
        return frames[0]
    
    def get_input_method(self) -> Tuple[str, Optional[np.ndarray]]:
        """Get input from selected method"""
        st.header("Input Method Selection")
        
        input_method = st.selectbox(
            "Choose Input Method:",
            ["Upload Image", "Camera", "Web Image", "Video", "Sample Images"],
            index=0
        )
        
        image = None
        
        if input_method == "Upload Image":
            image = self.handle_image_upload()
        elif input_method == "Camera":
            image = self.handle_camera_input()
        elif input_method == "Web Image":
            image = self.handle_web_image()
        elif input_method == "Video":
            frames = self.handle_video_input()
            if frames:
                image = frames[0]  # Use first frame
        elif input_method == "Sample Images":
            image = self.handle_sample_images()
        
        return input_method, image
    
    def display_input_info(self, image: np.ndarray, input_method: str):
        """Display information about the input image"""
        if image is None:
            return
        
        st.subheader("Image Information")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Input Method", input_method.replace(" ", "\n"))
        
        with col2:
            height, width = image.shape[:2]
            st.metric("Resolution", f"{width}x{height}")
        
        with col3:
            channels = image.shape[2] if len(image.shape) > 2 else 1
            st.metric("Channels", channels)
        
        with col4:
            size_mb = image.nbytes / (1024 * 1024)
            st.metric("Size", f"{size_mb:.2f} MB")
        
        # Additional image statistics
        if st.checkbox("Show detailed statistics"):
            self.display_detailed_statistics(image)
    
    def display_detailed_statistics(self, image: np.ndarray):
        """Display detailed image statistics"""
        st.write("#### Detailed Statistics")
        
        # Color statistics
        if len(image.shape) == 3:
            # BGR channels
            b, g, r = cv2.split(image)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Blue Channel**")
                st.metric("Mean", f"{np.mean(b):.1f}")
                st.metric("Std", f"{np.std(b):.1f}")
            
            with col2:
                st.write("**Green Channel**")
                st.metric("Mean", f"{np.mean(g):.1f}")
                st.metric("Std", f"{np.std(g):.1f}")
            
            with col3:
                st.write("**Red Channel**")
                st.metric("Mean", f"{np.mean(r):.1f}")
                st.metric("Std", f"{np.std(r):.1f}")
        
        # Histogram
        st.write("**Intensity Histogram**")
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        # Simple histogram display
        st.bar_chart(hist.flatten()[:50])  # Show first 50 bins
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except:
            pass
