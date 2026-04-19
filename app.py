"""
Spotscan Main Application
Comprehensive food analysis UI using Streamlit
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import sys
import os
from pathlib import Path

# Add the src directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

# Import all analysis modules
from core.food_detector import FoodDetector
from core.image_processor import ImageProcessor
from core.model_manager import ModelManager
from analysis.nutrition_analyzer import NutritionAnalyzer
from analysis.freshness_detector import FreshnessDetector
from analysis.ripeness_predictor import RipenessPredictor
from analysis.texture_analyzer import TextureAnalyzer
from analysis.color_analyzer import ColorAnalyzer
from analysis.shape_reconstructor import ShapeReconstructor
from analysis.ocr_analyzer import OCRAnalyzer
from analysis.artificial_natural_classifier import ArtificialNaturalClassifier
from analysis.processed_food_classifier import ProcessedFoodClassifier
from analysis.liquid_analyzer import LiquidAnalyzer
from analysis.acidity_analyzer import AcidityAnalyzer
from analysis.temperature_analyzer import TemperatureAnalyzer
from analysis.portion_analyzer import PortionAnalyzer
from analysis.solid_liquid_classifier import SolidLiquidClassifier
from analysis.safety_checker import SafetyChecker
from analysis.sustainability_detector import SustainabilityDetector
from analysis.visual_calorie_estimator import VisualCalorieEstimator
from analysis.enhanced_visual_estimator import EnhancedVisualEstimator
from analysis.vit_analyzer import ViTAnalyzer

class SpotscanApp:
    """Main Spotscan application class"""
    
    def __init__(self):
        """Initialize the application"""
        self.setup_page_config()
        self.initialize_modules()
        self.setup_session_state()
    
    def setup_page_config(self):
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title="Spotscan - AI Food Analysis",
            page_icon=":apple:",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def initialize_modules(self):
        """Initialize all analysis modules"""
        self.food_detector = FoodDetector()
        self.image_processor = ImageProcessor()
        self.model_manager = ModelManager()
        
        # Analysis modules
        self.nutrition_analyzer = NutritionAnalyzer()
        self.freshness_detector = FreshnessDetector()
        self.ripeness_predictor = RipenessPredictor()
        self.texture_analyzer = TextureAnalyzer()
        self.color_analyzer = ColorAnalyzer()
        self.shape_reconstructor = ShapeReconstructor()
        self.ocr_analyzer = OCRAnalyzer()
        self.artificial_natural_classifier = ArtificialNaturalClassifier()
        self.processed_food_classifier = ProcessedFoodClassifier()
        self.liquid_analyzer = LiquidAnalyzer()
        self.acidity_analyzer = AcidityAnalyzer()
        self.temperature_analyzer = TemperatureAnalyzer()
        self.portion_analyzer = PortionAnalyzer()
        self.solid_liquid_classifier = SolidLiquidClassifier()
        self.safety_checker = SafetyChecker()
        self.sustainability_detector = SustainabilityDetector()
        self.visual_calorie_estimator = VisualCalorieEstimator()
        self.enhanced_visual_estimator = EnhancedVisualEstimator()
        self.vit_analyzer = ViTAnalyzer()
    
    def setup_session_state(self):
        """Setup session state variables"""
        if 'current_image' not in st.session_state:
            st.session_state.current_image = None
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = {}
        if 'food_class' not in st.session_state:
            st.session_state.food_class = None
    
    def run(self):
        """Run the main application"""
        self.display_header()
        self.display_sidebar()
        self.display_main_content()
    
    def display_header(self):
        """Display application header"""
        st.title("Spotscan - AI Food Analysis Platform")
        st.markdown("---")
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Features", "19")
        with col2:
            st.metric("Analysis Modules", "15")
        with col3:
            st.metric("AI Models", "8")
        with col4:
            st.metric("Datasets", "2")
    
    def display_sidebar(self):
        """Display sidebar with controls"""
        st.sidebar.header("Controls")
        
        # Image input section
        st.sidebar.subheader("Image Input")
        
        input_method = st.sidebar.selectbox(
            "Input Method",
            ["Upload Image", "Camera", "Sample Images"]
        )
        
        if input_method == "Upload Image":
            self.handle_image_upload()
        elif input_method == "Camera":
            self.handle_camera_input()
        elif input_method == "Sample Images":
            self.handle_sample_images()
        
        # Analysis options
        st.sidebar.subheader("Analysis Options")
        
        # Food type input
        st.session_state.food_class = st.sidebar.text_input(
            "Food Type (optional)",
            value=st.session_state.food_class or ""
        )
        
        # Select analyses to run
        analysis_options = st.sidebar.multiselect(
            "Select Analyses",
            [
                "Food Detection",
                "Nutritional Analysis",
                "Freshness Detection",
                "Ripeness Prediction",
                "Texture Analysis",
                "Color Analysis",
                "3D Shape Reconstruction",
                "OCR Ingredient Detection",
                "Artificial vs Natural",
                "Processed Food Level",
                "Liquid Properties",
                "Acidity (pH) Estimation",
                "Temperature Measurement",
                "Size & Portion Estimation",
                "Visual Calorie Estimation",
                "Enhanced Visual Estimation",
                "ViT Food Classification",
                "Solid-to-Liquid Scale",
                "Health & Safety Warnings",
                "Sustainability Labels"
            ],
            default=["Food Detection", "Nutritional Analysis"]
        )
        
        # Run analysis button
        if st.sidebar.button("Run Analysis", type="primary"):
            if st.session_state.current_image is not None:
                self.run_comprehensive_analysis(analysis_options)
            else:
                st.sidebar.error("Please upload an image first")
        
        # Quick analysis presets
        st.sidebar.subheader("Quick Presets")
        
        if st.sidebar.button("Quick Health Check"):
            self.run_preset_analysis("health")
        
        if st.sidebar.button("Freshness Check"):
            self.run_preset_analysis("freshness")
        
        if st.sidebar.button("Complete Analysis"):
            self.run_preset_analysis("complete")
    
    def handle_image_upload(self):
        """Handle image upload"""
        uploaded_file = st.sidebar.file_uploader(
            "Choose an image",
            type=['jpg', 'jpeg', 'png', 'webp']
        )
        
        if uploaded_file is not None:
            # Convert to PIL Image
            image = Image.open(uploaded_file)
            # Convert to numpy array
            image_array = np.array(image)
            
            # Convert BGR to RGB for OpenCV
            if len(image_array.shape) == 3:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            st.session_state.current_image = image_array
            st.sidebar.success("Image uploaded successfully!")
    
    def handle_camera_input(self):
        """Handle camera input"""
        camera_image = st.sidebar.camera_input("Take a picture")
        
        if camera_image is not None:
            # Convert to numpy array
            image_array = np.array(camera_image)
            # Convert RGB to BGR for OpenCV
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            st.session_state.current_image = image_array
            st.sidebar.success("Image captured successfully!")
    
    def handle_sample_images(self):
        """Handle sample image selection"""
        st.sidebar.info("Sample images feature coming soon!")
    
    def run_comprehensive_analysis(self, selected_analyses):
        """Run comprehensive analysis based on selection"""
        if st.session_state.current_image is None:
            st.error("No image available for analysis")
            return
        
        image = st.session_state.current_image
        food_type = st.session_state.food_class
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = {}
        total_analyses = len(selected_analyses)
        
        for i, analysis in enumerate(selected_analyses):
            status_text.text(f"Running {analysis}...")
            
            try:
                if analysis == "Food Detection":
                    results['food_detection'] = self.food_detector.detect_food(image)
                elif analysis == "Nutritional Analysis":
                    results['nutrition'] = self.nutrition_analyzer.analyze_nutrition(image, food_type)
                elif analysis == "Freshness Detection":
                    results['freshness'] = self.freshness_detector.detect_freshness(image, food_type)
                elif analysis == "Ripeness Prediction":
                    results['ripeness'] = self.ripeness_predictor.predict_ripeness(image, food_type)
                elif analysis == "Texture Analysis":
                    results['texture'] = self.texture_analyzer.analyze_texture(image, food_type)
                elif analysis == "Color Analysis":
                    results['color'] = self.color_analyzer.analyze_colors(image, food_type)
                elif analysis == "3D Shape Reconstruction":
                    results['shape'] = self.shape_reconstructor.reconstruct_3d_shape(image, food_type)
                elif analysis == "OCR Ingredient Detection":
                    results['ocr'] = self.ocr_analyzer.analyze_packaging_text(image)
                elif analysis == "Artificial vs Natural":
                    # This would need ingredients list from OCR
                    ingredients = results.get('ocr', {}).get('comprehensive_report', {}).get('ingredients_list', [])
                    results['artificial_natural'] = self.artificial_natural_classifier.classify_ingredients(ingredients)
                elif analysis == "Processed Food Level":
                    ingredients = results.get('ocr', {}).get('comprehensive_report', {}).get('ingredients_list', [])
                    results['processed_food'] = self.processed_food_classifier.classify_processing_level(image, food_type, ingredients)
                elif analysis == "Liquid Properties":
                    results['liquid'] = self.liquid_analyzer.analyze_liquid_properties(image, food_type)
                elif analysis == "Acidity (pH) Estimation":
                    results['acidity'] = self.acidity_analyzer.estimate_acidity(image, food_type)
                elif analysis == "Temperature Measurement":
                    results['temperature'] = self.temperature_analyzer.estimate_temperature(image, food_type)
                elif analysis == "Size & Portion Estimation":
                    results['portion'] = self.portion_analyzer.analyze_portion(image, food_type)
                elif analysis == "Visual Calorie Estimation":
                    results['visual_calories'] = self.visual_calorie_estimator.analyze_complete_visual_calories(image, food_type)
                elif analysis == "Enhanced Visual Estimation":
                    results['enhanced_visual'] = self.enhanced_visual_estimator.analyze_enhanced_visual(image, food_type)
                elif analysis == "ViT Food Classification":
                    results['vit_classification'] = self.vit_analyzer.analyze(image)
                elif analysis == "Solid-to-Liquid Scale":
                    results['solid_liquid'] = self.solid_liquid_classifier.classify_solid_liquid_scale(image, food_type)
                elif analysis == "Health & Safety Warnings":
                    # Get nutrition data from previous analysis
                    nutrition_data = results.get('nutrition', {}).get('nutritional_info', {})
                    ingredients = results.get('ocr', {}).get('comprehensive_report', {}).get('ingredients_list', [])
                    results['safety'] = self.safety_checker.analyze_safety(nutrition_data, ingredients, food_type)
                elif analysis == "Sustainability Labels":
                    results['sustainability'] = self.sustainability_detector.detect_sustainability_labels(image)
                
            except Exception as e:
                st.error(f"Error in {analysis}: {str(e)}")
                results[analysis.lower().replace(' ', '_')] = {'error': str(e)}
            
            # Update progress
            progress = (i + 1) / total_analyses
            progress_bar.progress(progress)
        
        # Store results
        st.session_state.analysis_results = results
        progress_bar.progress(1.0)
        status_text.text("Analysis complete!")
        
        st.success("Analysis completed successfully!")
    
    def run_preset_analysis(self, preset_type):
        """Run preset analysis configurations"""
        if preset_type == "health":
            analyses = [
                "Food Detection",
                "Nutritional Analysis",
                "Artificial vs Natural",
                "Health & Safety Warnings"
            ]
        elif preset_type == "freshness":
            analyses = [
                "Food Detection",
                "Freshness Detection",
                "Ripeness Prediction",
                "Color Analysis"
            ]
        elif preset_type == "complete":
            analyses = [
                "Food Detection",
                "Nutritional Analysis",
                "Freshness Detection",
                "Ripeness Prediction",
                "Texture Analysis",
                "Color Analysis",
                "Artificial vs Natural",
                "Health & Safety Warnings"
            ]
        else:
            analyses = ["Food Detection"]
        
        self.run_comprehensive_analysis(analyses)
    
    def display_main_content(self):
        """Display main content area"""
        if st.session_state.current_image is not None:
            # Display image
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Input Image")
                # Convert BGR to RGB for display
                display_image = cv2.cvtColor(st.session_state.current_image, cv2.COLOR_BGR2RGB)
                st.image(display_image, use_column_width=True)
                
                # Display food type if detected
                if st.session_state.food_class:
                    st.info(f"Food Type: {st.session_state.food_class}")
            
            with col2:
                self.display_analysis_results()
        else:
            # Welcome message
            st.markdown("""
            ## Welcome to Spotscan! :apple:
            
            **AI-Powered Food Analysis Platform**
            
            ### Features:
            - **19 Advanced Analysis Modules**
            - **Computer Vision & Machine Learning**
            - **Comprehensive Food Insights**
            
            ### Getting Started:
            1. **Upload an image** using the sidebar
            2. **Select analyses** you want to run
            3. **Click "Run Analysis"** to get results
            
            ### Quick Presets:
            - **Health Check**: Nutrition, ingredients, safety warnings
            - **Freshness Check**: Ripeness, spoilage, quality assessment
            - **Complete Analysis**: Full comprehensive analysis
            
            ---
            *Powered by Food-101 and Nutrition5k datasets*
            """)
    
    def display_analysis_results(self):
        """Display analysis results"""
        if not st.session_state.analysis_results:
            st.info("No analysis results yet. Upload an image and run analysis.")
            return
        
        results = st.session_state.analysis_results
        
        # Create tabs for different result categories
        tabs = st.tabs([
            "Overview",
            "Nutrition & Health",
            "Quality & Freshness",
            "Physical Properties",
            "Safety & Sustainability"
        ])
        
        with tabs[0]:
            self.display_overview_tab(results)
        
        with tabs[1]:
            self.display_nutrition_health_tab(results)
        
        with tabs[2]:
            self.display_quality_freshness_tab(results)
        
        with tabs[3]:
            self.display_physical_properties_tab(results)
        
        with tabs[4]:
            self.display_safety_sustainability_tab(results)
    
    def display_overview_tab(self, results):
        """Display overview tab"""
        st.subheader("Analysis Overview")
        
        # Summary metrics
        if 'food_detection' in results:
            food_result = results['food_detection']
            if 'food_class' in food_result:
                st.success(f"Detected Food: {food_result['food_class']}")
                st.info(f"Confidence: {food_result.get('confidence', 'N/A'):.2%}")
        
        # ViT Classification
        if 'vit_classification' in results:
            vit_result = results['vit_classification']
            if vit_result.get('top_prediction'):
                top_pred = vit_result['top_prediction']
                st.success(f"ViT Classification: {top_pred['class'].replace('_', ' ').title()}")
                st.info(f"Confidence: {top_pred['confidence']:.2%}")
                
                # Show top predictions
                if len(vit_result.get('predictions', [])) > 1:
                    st.write("**Top Predictions:**")
                    for i, pred in enumerate(vit_result['predictions'][:3], 1):
                        st.write(f"{i}. {pred['class'].replace('_', ' ').title()}: {pred['confidence']:.2%}")
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'nutrition' in results:
                nutrition = results['nutrition']
                if 'calories' in nutrition:
                    st.metric("Calories", f"{nutrition['calories']:.0f}")
        
        with col2:
            if 'freshness' in results:
                freshness = results['freshness']
                if 'freshness_score' in freshness:
                    st.metric("Freshness", f"{freshness['freshness_score']:.1%}")
        
        with col3:
            if 'safety' in results:
                safety = results['safety']
                if 'safety_assessment' in safety:
                    level = safety['safety_assessment'].get('overall_level', 'Unknown')
                    st.metric("Safety Level", level.title())
        
        with col4:
            if 'sustainability' in results:
                sustainability = results['sustainability']
                if 'sustainability_score' in sustainability:
                    score = sustainability['sustainability_score'].get('overall_score', 0)
                    st.metric("Sustainability", f"{score:.1%}")
        
        # Analysis summary
        st.subheader("Analysis Summary")
        
        completed_analyses = []
        for key, result in results.items():
            if 'error' not in result:
                completed_analyses.append(key.replace('_', ' ').title())
        
        if completed_analyses:
            st.write("Completed Analyses:")
            for analysis in completed_analyses:
                st.write(f"  :white_check_mark: {analysis}")
        else:
            st.warning("No successful analyses completed")
    
    def display_nutrition_health_tab(self, results):
        """Display nutrition and health tab"""
        st.subheader("Nutrition & Health Analysis")
        
        # Nutritional Analysis
        if 'nutrition' in results:
            st.write("#### Nutritional Information")
            nutrition = results['nutrition']
            
            # Display key nutrients
            nutrients = ['calories', 'protein', 'carbohydrates', 'fat', 'fiber', 'sugar', 'sodium']
            
            col1, col2 = st.columns(2)
            with col1:
                for nutrient in nutrients[:4]:
                    if nutrient in nutrition:
                        value = nutrition[nutrient]
                        unit = 'g' if nutrient != 'calories' else 'kcal'
                        st.metric(nutrient.title(), f"{value:.1f} {unit}")
            
            with col2:
                for nutrient in nutrients[4:]:
                    if nutrient in nutrition:
                        value = nutrition[nutrient]
                        st.metric(nutrient.title(), f"{value:.1f} g")
        
        # Visual Calorie Estimation
        if 'visual_calories' in results:
            st.write("#### Visual Calorie Estimation")
            visual_calories = results['visual_calories']
            
            if 'calorie_analysis' in visual_calories:
                calorie_data = visual_calories['calorie_analysis']
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Estimated Calories", 
                        f"{calorie_data.get('calorie_estimate', {}).get('estimated_calories', 'N/A')} kcal"
                    )
                
                with col2:
                    confidence = calorie_data.get('confidence', 0)
                    st.metric("Confidence", f"{confidence:.2%}")
                
                with col3:
                    st.metric("Category", calorie_data.get('food_category', 'N/A'))
                
                # Additional details
                if 'estimation_range' in calorie_data.get('calorie_estimate', {}):
                    range_data = calorie_data['calorie_estimate']['estimation_range']
                    st.info(f"Estimation Range: {range_data[0]} - {range_data[1]} kcal")
                
                if 'serving_size_equivalent' in calorie_data.get('calorie_estimate', {}):
                    serving = calorie_data['calorie_estimate']['serving_size_equivalent']
                    st.info(f"Serving Size: {serving}")
                
                # Nutritional context
                if 'nutritional_context' in visual_calories:
                    context = visual_calories['nutritional_context']
                    st.write("**Nutritional Context:**")
                    st.write(f"- Daily Calorie %: {context.get('daily_calorie_percentage', 'N/A')}%")
                    st.write(f"- Meal Context: {context.get('meal_context', 'N/A')}")
                    st.write(f"- Calorie Density: {context.get('calorie_density', 'N/A')}")
                
                # Recommendations
                if 'recommendations' in visual_calories:
                    st.write("**Recommendations:**")
                    for rec in visual_calories['recommendations']:
                        st.write(f"- {rec}")
        
        # Enhanced Visual Estimation
        if 'enhanced_visual' in results:
            st.write("#### Enhanced Visual Estimation")
            enhanced_visual = results['enhanced_visual']
            
            # Visual Ingredients Analysis
            if 'ingredient_analysis' in enhanced_visual:
                ingredient_data = enhanced_visual['ingredient_analysis']
                st.write("**Visual Ingredient Detection:**")
                
                if ingredient_data['detected_ingredients']:
                    for ingredient in ingredient_data['detected_ingredients']:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(ingredient['name'], f"{ingredient['estimated_weight_grams']:.1f}g")
                        with col2:
                            st.metric("Coverage", f"{ingredient['coverage_percentage']:.1f}%")
                        with col3:
                            st.metric("Confidence", f"{ingredient['confidence']:.2%}")
                    
                    st.info(f"Total Coverage: {ingredient_data['total_coverage_percentage']:.1f}%")
                    st.info(f"Dominant Ingredient: {ingredient_data.get('dominant_ingredient', {}).get('name', 'N/A')}")
                else:
                    st.warning("No ingredients detected visually")
            
            # Portion Analysis
            if 'portion_analysis' in enhanced_visual:
                portion_data = enhanced_visual['portion_analysis']
                st.write("**Plate-Based Portion Analysis:**")
                
                if portion_data.get('method') == 'plate_based':
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Plate Diameter", f"{portion_data.get('plate_diameter_cm', 'N/A')}cm")
                    with col2:
                        st.metric("Total Weight", f"{portion_data.get('total_estimated_weight_grams', 'N/A')}g")
                    with col3:
                        st.metric("Portions", len(portion_data.get('portions', [])))
                    
                    # Display individual portions
                    if portion_data.get('portions'):
                        st.write("**Individual Portions:**")
                        for portion in portion_data['portions']:
                            st.write(f"- {portion['name']}: {portion['estimated_weight_grams']:.1f}g ({portion['portion_size']})")
                else:
                    st.warning("Plate not detected - using heuristic analysis")
            
            # Volume Analysis
            if 'volume_analysis' in enhanced_visual:
                volume_data = enhanced_visual['volume_analysis']
                st.write("**Drink Volume Estimation:**")
                
                if volume_data.get('detected'):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Volume", f"{volume_data.get('estimated_volume_ml', 'N/A')}ml")
                    with col2:
                        st.metric("Glass Type", volume_data.get('glass_type', 'N/A'))
                    with col3:
                        st.metric("Confidence", f"{volume_data.get('confidence', 'N/A'):.2%}")
                    
                    st.info(f"Dimensions: {volume_data.get('width_cm', 'N/A')}cm x {volume_data.get('height_cm', 'N/A')}cm")
                else:
                    st.warning("No drink detected")
            
            # Nutrition Facts Recognition
            if 'nutrition_facts_analysis' in enhanced_visual:
                nutrition_facts = enhanced_visual['nutrition_facts_analysis']
                st.write("**Nutrition Facts Recognition:**")
                
                if nutrition_facts.get('detected'):
                    st.write("**Estimated Nutrition from Label:**")
                    nutrition_data = nutrition_facts.get('estimated_nutrition', {})
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Calories", nutrition_data.get('calories', 'N/A'))
                        st.metric("Fat", f"{nutrition_data.get('fat', 'N/A')}g")
                        st.metric("Carbs", f"{nutrition_data.get('carbohydrates', 'N/A')}g")
                        st.metric("Protein", f"{nutrition_data.get('protein', 'N/A')}g")
                    
                    with col2:
                        st.metric("Fiber", f"{nutrition_data.get('fiber', 'N/A')}g")
                        st.metric("Sugar", f"{nutrition_data.get('sugar', 'N/A')}g")
                        st.metric("Sodium", f"{nutrition_data.get('sodium', 'N/A')}mg")
                        st.metric("Cholesterol", f"{nutrition_data.get('cholesterol', 'N/A')}mg")
                    
                    st.info(f"Recognition Confidence: {nutrition_facts.get('confidence', 'N/A'):.2%}")
                else:
                    st.warning("No nutrition facts panel detected")
            
            # Combined Analysis Summary
            if 'combined_analysis' in enhanced_visual:
                combined = enhanced_visual['combined_analysis']
                st.write("**Combined Visual Analysis:**")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Weight", f"{combined.get('total_estimated_weight_grams', 'N/A')}g")
                with col2:
                    st.metric("Total Volume", f"{combined.get('total_estimated_volume_ml', 'N/A')}ml")
                with col3:
                    st.metric("Overall Confidence", f"{combined.get('overall_confidence', 'N/A'):.2%}")
        
        # Artificial vs Natural
        if 'artificial_natural' in results:
            st.write("#### Ingredient Classification")
            artificial_natural = results['artificial_natural']
            
            if 'overall_statistics' in artificial_natural:
                stats = artificial_natural['overall_statistics']
                natural_pct = stats.get('natural_percentage', 0)
                artificial_pct = stats.get('artificial_percentage', 0)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Natural Ingredients", f"{natural_pct:.1f}%")
                with col2:
                    st.metric("Artificial Ingredients", f"{artificial_pct:.1f}%")
    
    def display_quality_freshness_tab(self, results):
        """Display quality and freshness tab"""
        st.subheader("Quality & Freshness Analysis")
        
        # Freshness Detection
        if 'freshness' in results:
            st.write("#### Freshness Assessment")
            freshness = results['freshness']
            
            if 'freshness_score' in freshness:
                score = freshness['freshness_score']
                st.metric("Freshness Score", f"{score:.1%}")
                
                # Status indicator
                if score > 0.8:
                    st.success("Very Fresh")
                elif score > 0.6:
                    st.info("Fresh")
                elif score > 0.4:
                    st.warning("Moderately Fresh")
                else:
                    st.error("Not Fresh")
            
            if 'recommendations' in freshness:
                st.write("**Recommendations:**")
                for rec in freshness['recommendations']:
                    st.write(f"  :bulb: {rec}")
        
        # Ripeness Prediction
        if 'ripeness' in results:
            st.write("#### Ripeness Prediction")
            ripeness = results['ripeness']
            
            if 'ripeness_stage' in ripeness:
                stage = ripeness['ripeness_stage']
                st.metric("Ripeness Stage", stage.title())
            
            if 'optimal_use' in ripeness:
                st.write("**Optimal Use:**")
                for use in ripeness['optimal_use']:
                    st.write(f"  :bulb: {use}")
        
        # Color Analysis
        if 'color' in results:
            st.write("#### Color Analysis")
            color = results['color']
            
            if 'dominant_colors' in color:
                dominant = color['dominant_colors'][:5]  # Top 5 colors
                for i, color_info in enumerate(dominant):
                    col_color, col_info = st.columns([1, 4])
                    with col_color:
                        # Display color swatch
                        st.markdown(
                            f'<div style="width:50px;height:50px;background-color:{color_info["hex"]};border:1px solid #ccc;"></div>',
                            unsafe_allow_html=True
                        )
                    with col_info:
                        st.write(f"**{color_info['color_name'].title()}**")
                        st.write(f"Hex: {color_info['hex']}")
                        st.write(f"{color_info['percentage']:.1%}")
    
    def display_physical_properties_tab(self, results):
        """Display physical properties tab"""
        st.subheader("Physical Properties Analysis")
        
        # Texture Analysis
        if 'texture' in results:
            st.write("#### Texture Analysis")
            texture = results['texture']
            
            if 'texture_properties' in texture:
                properties = texture['texture_properties']
                
                # Display key texture properties
                key_properties = ['crispiness', 'chewiness', 'softness', 'hardness', 'juiciness']
                
                cols = st.columns(3)
                for i, prop in enumerate(key_properties[:6]):
                    if prop in properties:
                        with cols[i % 3]:
                            value = properties[prop]
                            st.metric(prop.title(), f"{value:.2f}")
        
        # Temperature Analysis
        if 'temperature' in results:
            st.write("#### Temperature Analysis")
            temp = results['temperature']
            
            if 'estimated_temperature' in temp:
                temp_c = temp['estimated_temperature']
                temp_f = temp.get('temperature_fahrenheit', temp_c * 9/5 + 32)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Temperature (C)", f"{temp_c:.1f}C")
                with col2:
                    st.metric("Temperature (F)", f"{temp_f:.1f}F")
            
            if 'recommendations' in temp:
                st.write("**Recommendations:**")
                for rec in temp['recommendations']:
                    st.write(f"  :bulb: {rec}")
        
        # Portion Analysis
        if 'portion' in results:
            st.write("#### Portion Analysis")
            portion = results['portion']
            
            if 'weight_analysis' in portion:
                weight = portion['weight_analysis']
                st.metric("Estimated Weight", f"{weight.get('weight', 0):.1f} g")
            
            if 'portion_classification' in portion:
                classification = portion['portion_classification']
                st.metric("Portion Size", classification.get('size_category', 'Unknown').title())
        
        # Solid-Liquid Scale
        if 'solid_liquid' in results:
            st.write("#### Solid-Liquid Classification")
            solid_liquid = results['solid_liquid']
            
            if 'solid_liquid_scale' in solid_liquid:
                scale = solid_liquid['solid_liquid_scale']
                scale_info = solid_liquid.get('scale_details', {})
                
                st.metric("Consistency", scale_info.get('name', 'Unknown'))
                st.write(scale_info.get('description', ''))
        
        # Acidity Analysis
        if 'acidity' in results:
            st.write("#### Acidity (pH) Analysis")
            acidity = results['acidity']
            
            if 'estimated_ph' in acidity:
                ph = acidity['estimated_ph']
                ph_category = acidity.get('ph_category', {})
                
                st.metric("pH Level", f"{ph:.2f}")
                st.write(f"Category: {ph_category.get('level', 'Unknown').title()}")
    
    def display_safety_sustainability_tab(self, results):
        """Display safety and sustainability tab"""
        st.subheader("Safety & Sustainability Analysis")
        
        # Safety Warnings (already shown in nutrition tab, but can add more detail here)
        if 'safety' in results:
            st.write("#### Detailed Safety Analysis")
            safety = results['safety']
            
            if 'recommendations' in safety:
                st.write("**Safety Recommendations:**")
                for rec in safety['recommendations']:
                    st.write(f"  :warning: {rec}")
        
        # Sustainability Labels
        if 'sustainability' in results:
            st.write("#### Sustainability Certifications")
            sustainability = results['sustainability']
            
            if 'summary' in sustainability:
                summary = sustainability['summary']
                st.info(summary.get('summary', 'No sustainability labels detected'))
            
            if 'verified_labels' in sustainability:
                verified = sustainability['verified_labels']
                
                for label in verified:
                    if label['is_authentic']:
                        label_info = label['label_info']
                        with st.expander(f":green_heart: {label_info.get('name', 'Unknown')}"):
                            st.write(f"**Description:** {label_info.get('description', 'No description')}")
                            st.write(f"**Credibility:** {label_info.get('credibility', 'Unknown').title()}")
                            
                            if 'standards' in label_info:
                                st.write("**Standards:**")
                                for standard in label_info['standards']:
                                    st.write(f"  :white_check_mark: {standard.replace('_', ' ').title()}")
            
            if 'recommendations' in sustainability:
                st.write("**Sustainability Recommendations:**")
                for rec in sustainability['recommendations']:
                    st.write(f"  :bulb: {rec}")
        
        # OCR Results (if available)
        if 'ocr' in results:
            st.write("#### Ingredient Analysis")
            ocr = results['ocr']
            
            if 'comprehensive_report' in ocr:
                report = ocr['comprehensive_report']
                
                if 'total_ingredients' in report:
                    st.metric("Total Ingredients", report['total_ingredients'])
                
                if 'natural_percentage' in report:
                    st.metric("Natural Ingredients", f"{report['natural_percentage']:.1f}%")
                
                if 'artificial_percentage' in report:
                    st.metric("Artificial Ingredients", f"{report['artificial_percentage']:.1f}%")

def main():
    """Main function to run the app"""
    app = SpotscanApp()
    app.run()

if __name__ == "__main__":
    main()
