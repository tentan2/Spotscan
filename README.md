# Spotscan
Food Analysis Platform
# Spotscan - AI-Powered Food Analysis Platform

Spotscan is an advanced AI-powered food analysis platform that uses computer vision, OCR, machine learning, and food-science datasets to provide comprehensive food analysis. Unlike traditional nutrition apps, Spotscan analyzes the actual food in front of the camera - its color, texture, ripeness, shape, ingredients, and more.

## Features

### Core Analysis
- **Nutritional Estimation**: Comprehensive nutrient analysis including calories, vitamins, minerals, and more
- **Freshness & Spoilage Detection**: Detects color changes, texture abnormalities, and mold patterns
- **Ripeness Prediction**: Determines ripeness stage and optimal consumption timing
- **Texture Analysis**: Evaluates crispiness, chewiness, juiciness, and other physical properties
- **Color Analysis**: Captures exact hex colors for spoilage detection and composition accuracy
- **3D Shape Reconstruction**: Generates 3D models for improved object detection accuracy

### Advanced Features
- **Ingredient Detection (OCR)**: Reads and analyzes ingredient lists from packaging
- **Artificial vs Natural Classification**: Categorizes ingredients and flags artificial additives
- **Processed Food Level System**: Classifies foods on a 5-level processing scale
- **Liquid Properties**: Measures viscosity, transparency, and cohesion
- **Acidity (pH) Estimation**: Estimates acidity levels using color-based analysis
- **Temperature Measurement**: Estimates food temperature and serving recommendations
- **Size & Volume Estimation**: Visual estimation of weight, volume, and portion sizes

### Safety & Health
- **Health & Safety Warnings**: Flags high sodium, sugar, artificial ingredients
- **Sustainability Label Recognition**: Detects certification labels (Fairtrade, B-Corp, etc.)
- **Flavor Simulant Labeling**: Proposes new labeling standards for transparency

### Input Methods
- Live camera capture
- Video analysis
- Image uploads
- Website image analysis

### Tiers
- **Free Tier**: For individual consumers
- **Corporate Tier**: For food industry, restaurants, and government regulators

## Technology Stack

- **Backend**: Python, FastAPI, PyTorch
- **Computer Vision**: OpenCV, Torchvision
- **Machine Learning**: Scikit-learn, TensorFlow
- **OCR**: Tesseract
- **Frontend**: Streamlit, React
- **Database**: PostgreSQL, Redis
- **Datasets**: Food-101, Nutrition5k

## Installation

```bash
# Clone the repository
git clone https://github.com/tentan2/spotscan.git
cd spotscan

# Install dependencies
pip install -r requirements.txt

# Download datasets (requires Kaggle API)
python scripts/download_datasets.py

# Run the application
streamlit run src/ui/app.py
```

## Dataset Setup

1. **Food-101 Dataset**: Contains 101,000 images across 101 food categories
2. **Nutrition5k Dataset**: Provides comprehensive nutritional data

Both datasets are automatically downloaded and processed during setup.

## Project Structure

```
spotscan/
|-- data/                 # Dataset storage
|-- models/              # Trained models
|-- src/
|   |-- core/            # Core functionality
|   |-- analysis/        # Analysis modules
|   |-- ui/              # User interface
|   |-- utils/           # Utility functions
|-- tests/               # Test suite
|-- docs/                # Documentation
```

## Contributing

This project is open-source and welcomes contributions. Please see the contributing guidelines for more information.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
