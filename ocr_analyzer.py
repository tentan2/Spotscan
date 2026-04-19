"""
OCR Analyzer Module
Reads and analyzes ingredient lists from food packaging using Optical Character Recognition
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import pytesseract
import re
from pathlib import Path
import json
from collections import Counter
import requests
from bs4 import BeautifulSoup

class OCRAnalyzer:
    """Analyzes food packaging text using OCR and ingredient detection"""
    
    def __init__(self, tesseract_path: Optional[str] = None):
        """
        Initialize OCR analyzer
        
        Args:
            tesseract_path: Path to Tesseract executable (if needed)
        """
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # Ingredient databases and patterns
        self.ingredient_patterns = self._load_ingredient_patterns()
        self.additive_database = self._load_additive_database()
        self.allergen_keywords = self._load_allergen_keywords()
        
        # Common packaging text regions
        self.text_regions = {
            'ingredients_list': {
                'keywords': ['ingredients', 'ingredientes', 'zutaten', 'ingredients'],
                'position': 'center_bottom'
            },
            'nutrition_facts': {
                'keywords': ['nutrition', 'nutrition facts', 'nutritional information'],
                'position': 'side_or_back'
            },
            'allergens': {
                'keywords': ['allergens', 'contains', 'may contain'],
                'position': 'bottom'
            }
        }
    
    def _load_ingredient_patterns(self) -> Dict:
        """Load ingredient recognition patterns"""
        return {
            'natural_ingredients': {
                'fruits': ['apple', 'banana', 'orange', 'strawberry', 'blueberry', 'raspberry', 
                          'mango', 'pineapple', 'grape', 'peach', 'pear', 'cherry', 'kiwi'],
                'vegetables': ['carrot', 'broccoli', 'spinach', 'tomato', 'potato', 'onion', 
                              'garlic', 'pepper', 'cucumber', 'lettuce', 'celery', 'corn'],
                'grains': ['wheat', 'rice', 'oats', 'barley', 'quinoa', 'corn', 'rye', 'millet'],
                'proteins': ['chicken', 'beef', 'pork', 'fish', 'egg', 'milk', 'cheese', 'yogurt'],
                'nuts': ['almond', 'walnut', 'pecan', 'cashew', 'pistachio', 'hazelnut'],
                'seeds': ['sunflower', 'pumpkin', 'sesame', 'chia', 'flax', 'hemp']
            },
            'artificial_additives': {
                'colors': ['red 40', 'yellow 5', 'blue 1', 'tartrazine', 'sunset yellow', 
                          'carmine', 'annatto', 'beta carotene'],
                'preservatives': ['sodium benzoate', 'potassium sorbate', 'calcium propionate', 
                                'sodium nitrite', 'bha', 'bht', 'propyl gallate'],
                'sweeteners': ['aspartame', 'sucralose', 'acesulfame potassium', 'saccharin', 
                             'stevia', 'xylitol', 'sorbitol', 'mannitol'],
                'flavors': ['artificial flavor', 'natural flavor', 'vanillin', 'ethyl vanillin', 
                          'maltol', 'ethyl maltol'],
                'emulsifiers': ['lecithin', 'xanthan gum', 'guar gum', 'carrageenan', 
                              'pectin', 'cellulose gum']
            }
        }
    
    def _load_additive_database(self) -> Dict:
        """Load comprehensive additive database"""
        return {
            'E_numbers': {
                'E100': 'Curcumin',
                'E101': 'Riboflavin',
                'E102': 'Tartrazine',
                'E104': 'Quinoline Yellow',
                'E110': 'Sunset Yellow',
                'E120': 'Cochineal',
                'E122': 'Carmoisine',
                'E124': 'Ponceau',
                'E129': 'Allura Red',
                'E131': 'Patent Blue',
                'E132': 'Indigo Carmine',
                'E133': 'Brilliant Blue',
                'E140': 'Chlorophylls',
                'E141': 'Copper complexes of chlorophylls',
                'E142': 'Chlorophyllin',
                'E150a': 'Caramel I',
                'E150b': 'Caramel II',
                'E150c': 'Caramel III',
                'E150d': 'Caramel IV',
                'E160a': 'Carotenes',
                'E160b': 'Annatto',
                'E160c': 'Paprika extract',
                'E160d': 'Lycopene',
                'E160e': 'Beta-apo-8\'-carotenal',
                'E160f': 'Ethyl ester of beta-apo-8\'-carotenic acid',
                'E161b': 'Lutein',
                'E161g': 'Canthaxanthin',
                'E162': 'Beetroot Red',
                'E163': 'Anthocyanins',
                'E170': 'Calcium carbonate',
                'E171': 'Titanium dioxide',
                'E172': 'Iron oxides',
                'E200': 'Sorbic acid',
                'E201': 'Sodium sorbate',
                'E202': 'Potassium sorbate',
                'E203': 'Calcium sorbate',
                'E210': 'Benzoic acid',
                'E211': 'Sodium benzoate',
                'E212': 'Potassium benzoate',
                'E213': 'Calcium benzoate',
                'E214': 'Parahydroxybenzoic acid',
                'E215': 'Sodium parahydroxybenzoate',
                'E216': 'Propyl parahydroxybenzoate',
                'E217': 'Sodium propyl parahydroxybenzoate',
                'E218': 'Methyl parahydroxybenzoate',
                'E219': 'Sodium methyl parahydroxybenzoate',
                'E220': 'Sulphur dioxide',
                'E221': 'Sodium sulphite',
                'E222': 'Sodium bisulphite',
                'E223': 'Sodium metabisulphite',
                'E224': 'Potassium bisulphite',
                'E225': 'Potassium metabisulphite',
                'E226': 'Calcium sulphite',
                'E227': 'Calcium bisulphite',
                'E228': 'Potassium bisulphite',
                'E250': 'Sodium nitrite',
                'E251': 'Sodium nitrate',
                'E252': 'Potassium nitrate',
                'E280': 'Propionic acid',
                'E281': 'Sodium propionate',
                'E282': 'Calcium propionate',
                'E283': 'Potassium propionate'
            }
        }
    
    def _load_allergen_keywords(self) -> Dict:
        """Load allergen detection keywords"""
        return {
            'major_allergens': [
                'milk', 'egg', 'fish', 'shellfish', 'tree nuts', 'peanuts', 
                'wheat', 'soybean', 'sesame'
            ],
            'allergen_indicators': [
                'contains', 'may contain', 'processed in a facility that also processes',
                'allergen', 'allergy', 'allergic'
            ]
        }
    
    def analyze_packaging_text(self, image: np.ndarray) -> Dict:
        """
        Analyze text from food packaging image
        
        Args:
            image: Packaging image
            
        Returns:
            Comprehensive text analysis results
        """
        # Preprocess image for OCR
        processed_image = self._preprocess_for_ocr(image)
        
        # Extract text using OCR
        extracted_text = self._extract_text(processed_image)
        
        # Identify text regions
        text_regions = self._identify_text_regions(extracted_text)
        
        # Parse ingredients list
        ingredients_analysis = self._parse_ingredients(text_regions.get('ingredients_list', ''))
        
        # Detect additives
        additives_analysis = self._detect_additives(text_regions.get('ingredients_list', ''))
        
        # Identify allergens
        allergens_analysis = self._identify_allergens(extracted_text)
        
        # Analyze nutritional information
        nutrition_analysis = self._parse_nutrition_facts(text_regions.get('nutrition_facts', ''))
        
        # Detect deceptive branding
        branding_analysis = self._detect_deceptive_branding(
            text_regions, ingredients_analysis
        )
        
        # Generate comprehensive report
        comprehensive_report = self._generate_comprehensive_report(
            extracted_text, text_regions, ingredients_analysis, 
            additives_analysis, allergens_analysis, nutrition_analysis, branding_analysis
        )
        
        return {
            'extracted_text': extracted_text,
            'text_regions': text_regions,
            'ingredients_analysis': ingredients_analysis,
            'additives_analysis': additives_analysis,
            'allergens_analysis': allergens_analysis,
            'nutrition_analysis': nutrition_analysis,
            'branding_analysis': branding_analysis,
            'comprehensive_report': comprehensive_report
        }
    
    def _preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image to improve OCR accuracy"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(binary)
        
        # Enhance contrast
        enhanced = cv2.convertScaleAbs(denoised, alpha=1.2, beta=10)
        
        # Remove small noise
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def _extract_text(self, image: np.ndarray) -> str:
        """Extract text using OCR"""
        try:
            # Configure Tesseract for better text recognition
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,()-/'
            
            # Extract text
            text = pytesseract.image_to_string(image, config=custom_config)
            
            return text.strip()
        except Exception as e:
            return f"OCR Error: {str(e)}"
    
    def _identify_text_regions(self, full_text: str) -> Dict[str, str]:
        """Identify and categorize different text regions"""
        text_regions = {}
        lines = full_text.split('\n')
        
        # Find ingredients list
        ingredients_start = -1
        ingredients_end = -1
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            
            # Check for ingredients section start
            if any(keyword in line_lower for keyword in self.text_regions['ingredients_list']['keywords']):
                ingredients_start = i + 1
                continue
            
            # If we found ingredients start, look for end
            if ingredients_start != -1:
                # End of ingredients section (next major section)
                if any(keyword in line_lower for keyword in ['nutrition', 'facts', 'serving', 'calories']):
                    ingredients_end = i
                    break
                # End if line is very short (likely end of list)
                elif len(line.strip()) < 3 and i > ingredients_start + 1:
                    ingredients_end = i
                    break
        
        # Extract ingredients list
        if ingredients_start != -1:
            if ingredients_end == -1:
                ingredients_end = len(lines)
            ingredients_text = '\n'.join(lines[ingredients_start:ingredients_end])
            text_regions['ingredients_list'] = ingredients_text
        
        # Find nutrition facts
        nutrition_start = -1
        nutrition_end = -1
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            
            if any(keyword in line_lower for keyword in self.text_regions['nutrition_facts']['keywords']):
                nutrition_start = i
                continue
            
            if nutrition_start != -1:
                # End of nutrition section
                if len(line.strip()) < 3 and i > nutrition_start + 1:
                    nutrition_end = i
                    break
        
        if nutrition_start != -1:
            if nutrition_end == -1:
                nutrition_end = len(lines)
            nutrition_text = '\n'.join(lines[nutrition_start:nutrition_end])
            text_regions['nutrition_facts'] = nutrition_text
        
        # Store full text for other analyses
        text_regions['full_text'] = full_text
        
        return text_regions
    
    def _parse_ingredients(self, ingredients_text: str) -> Dict:
        """Parse and analyze ingredients list"""
        if not ingredients_text:
            return {'error': 'No ingredients text found'}
        
        # Clean and split ingredients
        ingredients_text = ingredients_text.replace(';', ',').replace(' and ', ', ')
        ingredients = [ing.strip() for ing in ingredients_text.split(',') if ing.strip()]
        
        # Analyze each ingredient
        ingredient_analysis = []
        total_ingredients = len(ingredients)
        
        for i, ingredient in enumerate(ingredients):
            # Classify ingredient
            classification = self._classify_ingredient(ingredient)
            
            # Check if it's a compound ingredient
            is_compound = self._is_compound_ingredient(ingredient)
            
            # Extract sub-ingredients if compound
            sub_ingredients = []
            if is_compound:
                sub_ingredients = self._extract_sub_ingredients(ingredient)
            
            ingredient_analysis.append({
                'name': ingredient,
                'position': i + 1,
                'percentage': self._estimate_ingredient_percentage(i + 1, total_ingredients),
                'classification': classification,
                'is_compound': is_compound,
                'sub_ingredients': sub_ingredients,
                'processing_level': self._assess_processing_level(ingredient)
            })
        
        # Calculate composition statistics
        natural_count = sum(1 for ing in ingredient_analysis if ing['classification'] == 'natural')
        artificial_count = sum(1 for ing in ingredient_analysis if ing['classification'] == 'artificial')
        
        return {
            'total_ingredients': total_ingredients,
            'ingredient_list': ingredient_analysis,
            'natural_ingredients': natural_count,
            'artificial_ingredients': artificial_count,
            'natural_percentage': (natural_count / total_ingredients) * 100 if total_ingredients > 0 else 0,
            'artificial_percentage': (artificial_count / total_ingredients) * 100 if total_ingredients > 0 else 0
        }
    
    def _classify_ingredient(self, ingredient: str) -> str:
        """Classify ingredient as natural or artificial"""
        ingredient_lower = ingredient.lower()
        
        # Check against natural ingredients database
        for category, items in self.ingredient_patterns['natural_ingredients'].items():
            if any(natural_item in ingredient_lower for natural_item in items):
                return 'natural'
        
        # Check against artificial additives database
        for category, items in self.ingredient_patterns['artificial_additives'].items():
            if any(artificial_item in ingredient_lower for artificial_item in items):
                return 'artificial'
        
        # Check E-numbers
        if re.search(r'e\d{3,4}', ingredient_lower):
            return 'artificial'
        
        # Check for chemical-sounding names
        chemical_patterns = [
            r'\b(acid|ate|ite|ide|hydro|oxy|mono|di|tri|poly)\b',
            r'\b(sodium|potassium|calcium|magnesium)\b',
            r'\b(dextrose|fructose|glucose|sucrose|lactose)\b'
        ]
        
        for pattern in chemical_patterns:
            if re.search(pattern, ingredient_lower):
                return 'artificial'
        
        # Default to natural if uncertain
        return 'natural'
    
    def _is_compound_ingredient(self, ingredient: str) -> bool:
        """Check if ingredient is a compound ingredient"""
        # Look for indicators of compound ingredients
        compound_indicators = [
            'powder', 'extract', 'concentrate', 'puree', 'paste', 'sauce',
            'mix', 'blend', 'flavor', 'seasoning', 'spice', 'herb'
        ]
        
        ingredient_lower = ingredient.lower()
        return any(indicator in ingredient_lower for indicator in compound_indicators)
    
    def _extract_sub_ingredients(self, compound_ingredient: str) -> List[str]:
        """Extract sub-ingredients from compound ingredient"""
        # Look for parentheses containing sub-ingredients
        match = re.search(r'\((.*?)\)', compound_ingredient)
        if match:
            sub_ingredients_text = match.group(1)
            sub_ingredients = [ing.strip() for ing in sub_ingredients_text.split(',')]
            return sub_ingredients
        
        return []
    
    def _estimate_ingredient_percentage(self, position: int, total_ingredients: int) -> float:
        """Estimate ingredient percentage based on position in list"""
        # Ingredients are listed in descending order by weight
        # This is a rough estimation
        if position == 1:
            return max(20.0, 100.0 / total_ingredients)
        elif position == 2:
            return max(15.0, 80.0 / total_ingredients)
        elif position <= 5:
            return max(10.0, 60.0 / total_ingredients)
        else:
            return max(5.0, 40.0 / total_ingredients)
    
    def _assess_processing_level(self, ingredient: str) -> int:
        """Assess processing level (1-5 scale)"""
        ingredient_lower = ingredient.lower()
        
        # Level 1: Whole, unaltered foods
        level1_keywords = ['whole', 'fresh', 'raw', 'dried']
        if any(keyword in ingredient_lower for keyword in level1_keywords):
            return 1
        
        # Level 2: Minimally processed
        level2_keywords = ['chopped', 'cut', 'sliced', 'crushed', 'ground']
        if any(keyword in ingredient_lower for keyword in level2_keywords):
            return 2
        
        # Level 3: Naturally processed
        level3_keywords = ['fermented', 'cultured', 'aged', 'cured', 'smoked']
        if any(keyword in ingredient_lower for keyword in level3_keywords):
            return 3
        
        # Level 4: Combined natural foods
        level4_keywords = ['blend', 'mix', 'puree', 'concentrate']
        if any(keyword in ingredient_lower for keyword in level4_keywords):
            return 4
        
        # Level 5: Ultra-processed
        level5_keywords = ['hydrogenated', 'modified', 'synthetic', 'artificial']
        if any(keyword in ingredient_lower for keyword in level5_keywords):
            return 5
        
        # Default based on classification
        return 3
    
    def _detect_additives(self, ingredients_text: str) -> Dict:
        """Detect artificial additives in ingredients"""
        if not ingredients_text:
            return {'error': 'No ingredients text found'}
        
        detected_additives = []
        ingredients_lower = ingredients_text.lower()
        
        # Check for artificial additives
        for category, additives in self.ingredient_patterns['artificial_additives'].items():
            for additive in additives:
                if additive in ingredients_lower:
                    detected_additives.append({
                        'name': additive,
                        'category': category,
                        'type': 'artificial'
                    })
        
        # Check for E-numbers
        e_number_matches = re.findall(r'e\d{3,4}', ingredients_lower)
        for e_number in e_number_matches:
            e_number_upper = e_number.upper()
            if e_number_upper in self.additive_database['E_numbers']:
                detected_additives.append({
                    'name': e_number_upper,
                    'description': self.additive_database['E_numbers'][e_number_upper],
                    'category': 'E_number',
                    'type': 'artificial'
                })
        
        # Count additives by category
        additive_counts = {}
        for additive in detected_additives:
            category = additive['category']
            additive_counts[category] = additive_counts.get(category, 0) + 1
        
        return {
            'total_additives': len(detected_additives),
            'detected_additives': detected_additives,
            'additive_counts': additive_counts,
            'additive_density': len(detected_additives) / len(ingredients_text.split(',')) if ingredients_text else 0
        }
    
    def _identify_allergens(self, full_text: str) -> Dict:
        """Identify allergens in text"""
        text_lower = full_text.lower()
        detected_allergens = []
        
        # Check for major allergens
        for allergen in self.allergen_keywords['major_allergens']:
            if allergen in text_lower:
                detected_allergens.append({
                    'name': allergen,
                    'type': 'major_allergen',
                    'severity': 'high'
                })
        
        # Check for allergen indicators
        for indicator in self.allergen_keywords['allergen_indicators']:
            if indicator in text_lower:
                # Extract context around indicator
                context_start = max(0, text_lower.find(indicator) - 50)
                context_end = min(len(text_lower), text_lower.find(indicator) + 100)
                context = text_lower[context_start:context_end]
                
                detected_allergens.append({
                    'name': indicator,
                    'type': 'indicator',
                    'severity': 'medium',
                    'context': context
                })
        
        return {
            'detected_allergens': detected_allergens,
            'allergen_count': len(detected_allergens),
            'has_major_allergens': any(a['type'] == 'major_allergen' for a in detected_allergens)
        }
    
    def _parse_nutrition_facts(self, nutrition_text: str) -> Dict:
        """Parse nutrition facts from text"""
        if not nutrition_text:
            return {'error': 'No nutrition text found'}
        
        nutrition_data = {}
        
        # Common nutrition patterns
        nutrition_patterns = {
            'calories': r'calories?\s*:?\s*(\d+)',
            'fat': r'fat\s*:?\s*([\d.]+)g?',
            'saturated_fat': r'saturated\s+fat\s*:?\s*([\d.]+)g?',
            'trans_fat': r'trans\s+fat\s*:?\s*([\d.]+)g?',
            'cholesterol': r'cholesterol\s*:?\s*([\d.]+)mg?',
            'sodium': r'sodium\s*:?\s*([\d.]+)mg?',
            'carbohydrates': r'carbohydrates?\s*:?\s*([\d.]+)g?',
            'fiber': r'fiber\s*:?\s*([\d.]+)g?',
            'sugar': r'sugar\s*:?\s*([\d.]+)g?',
            'protein': r'protein\s*:?\s*([\d.]+)g?',
            'vitamin_a': r'vitamin\s*a\s*:?\s*([\d.]+)%',
            'vitamin_c': r'vitamin\s*c\s*:?\s*([\d.]+)%',
            'calcium': r'calcium\s*:?\s*([\d.]+)%',
            'iron': r'iron\s*:?\s*([\d.]+)%'
        }
        
        for nutrient, pattern in nutrition_patterns.items():
            match = re.search(pattern, nutrition_text.lower())
            if match:
                nutrition_data[nutrient] = float(match.group(1))
        
        return {
            'nutrition_data': nutrition_data,
            'total_nutrients': len(nutrition_data),
            'has_calorie_info': 'calories' in nutrition_data,
            'has_macronutrients': any(n in nutrition_data for n in ['fat', 'carbohydrates', 'protein'])
        }
    
    def _detect_deceptive_branding(self, text_regions: Dict, ingredients_analysis: Dict) -> Dict:
        """Detect deceptive branding practices"""
        deceptive_indicators = []
        
        # Check for flavor simulant indicators
        if 'ingredients_list' in text_regions:
            ingredients_text = text_regions['ingredients_list'].lower()
            
            # Look for "flavored" products
            flavor_matches = re.findall(r'(\w+)\s+flavored', ingredients_text)
            for flavor in flavor_matches:
                # Check if the flavor ingredient is actually present
                flavor_present = flavor in ingredients_text
                if not flavor_present:
                    deceptive_indicators.append({
                        'type': 'flavor_simulant',
                        'description': f'"{flavor} flavored" but no {flavor} found in ingredients',
                        'severity': 'high'
                    })
        
        # Check for misleading health claims
        full_text = text_regions.get('full_text', '').lower()
        health_claims = ['natural', 'healthy', 'low fat', 'sugar free', 'organic']
        
        for claim in health_claims:
            if claim in full_text:
                # Verify claim against ingredients
                if ingredients_analysis.get('artificial_percentage', 0) > 20 and claim in ['natural', 'healthy']:
                    deceptive_indicators.append({
                        'type': 'misleading_health_claim',
                        'description': f'Claims "{claim}" but contains {ingredients_analysis["artificial_percentage"]:.1f}% artificial ingredients',
                        'severity': 'medium'
                    })
        
        # Check for "made with real fruit" claims
        if 'real fruit' in full_text:
            fruit_ingredients = [ing for ing in ingredients_analysis.get('ingredient_list', []) 
                               if 'fruit' in ing['name'].lower()]
            if len(fruit_ingredients) == 0:
                deceptive_indicators.append({
                    'type': 'false_fruit_claim',
                    'description': 'Claims "made with real fruit" but no fruit ingredients found',
                    'severity': 'high'
                })
        
        return {
            'deceptive_indicators': deceptive_indicators,
            'deceptive_count': len(deceptive_indicators),
            'has_deceptive_branding': len(deceptive_indicators) > 0,
            'overall_risk': 'high' if len(deceptive_indicators) > 2 else 'medium' if len(deceptive_indicators) > 0 else 'low'
        }
    
    def _generate_comprehensive_report(self, extracted_text: str, text_regions: Dict,
                                    ingredients_analysis: Dict, additives_analysis: Dict,
                                    allergens_analysis: Dict, nutrition_analysis: Dict,
                                    branding_analysis: Dict) -> Dict:
        """Generate comprehensive analysis report"""
        report = {
            'summary': {
                'total_ingredients': ingredients_analysis.get('total_ingredients', 0),
                'natural_percentage': ingredients_analysis.get('natural_percentage', 0),
                'artificial_percentage': ingredients_analysis.get('artificial_percentage', 0),
                'total_additives': additives_analysis.get('total_additives', 0),
                'allergen_count': allergens_analysis.get('allergen_count', 0),
                'deceptive_indicators': branding_analysis.get('deceptive_count', 0)
            },
            'health_score': self._calculate_health_score(ingredients_analysis, additives_analysis),
            'recommendations': self._generate_recommendations(
                ingredients_analysis, additives_analysis, allergens_analysis, branding_analysis
            ),
            'warnings': self._generate_warnings(allergens_analysis, branding_analysis),
            'processing_level': self._calculate_overall_processing_level(ingredients_analysis)
        }
        
        return report
    
    def _calculate_health_score(self, ingredients_analysis: Dict, additives_analysis: Dict) -> Dict:
        """Calculate overall health score"""
        base_score = 100
        
        # Deduct points for artificial ingredients
        artificial_percentage = ingredients_analysis.get('artificial_percentage', 0)
        base_score -= artificial_percentage * 0.5
        
        # Deduct points for additives
        additive_count = additives_analysis.get('total_additives', 0)
        base_score -= additive_count * 2
        
        # Bonus for high natural percentage
        natural_percentage = ingredients_analysis.get('natural_percentage', 0)
        if natural_percentage > 80:
            base_score += 10
        
        health_score = max(0, min(100, base_score))
        
        return {
            'score': health_score,
            'grade': self._get_health_grade(health_score),
            'factors': {
                'natural_ingredients': natural_percentage,
                'artificial_ingredients': artificial_percentage,
                'additive_count': additive_count
            }
        }
    
    def _get_health_grade(self, score: float) -> str:
        """Get health grade from score"""
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'
    
    def _generate_recommendations(self, ingredients_analysis: Dict, additives_analysis: Dict,
                                allergens_analysis: Dict, branding_analysis: Dict) -> List[str]:
        """Generate health and safety recommendations"""
        recommendations = []
        
        # Natural vs artificial recommendations
        artificial_percentage = ingredients_analysis.get('artificial_percentage', 0)
        if artificial_percentage > 30:
            recommendations.append("Consider choosing products with fewer artificial ingredients")
        elif artificial_percentage > 50:
            recommendations.append("This product contains a high percentage of artificial ingredients")
        
        # Additive recommendations
        additive_count = additives_analysis.get('total_additives', 0)
        if additive_count > 5:
            recommendations.append("Product contains many additives - consider more natural alternatives")
        
        # Allergen recommendations
        if allergens_analysis.get('has_major_allergens'):
            recommendations.append("Product contains major allergens - check if suitable for your dietary needs")
        
        # Branding recommendations
        if branding_analysis.get('has_deceptive_branding'):
            recommendations.append("Be aware of potentially misleading claims on this product")
        
        return recommendations
    
    def _generate_warnings(self, allergens_analysis: Dict, branding_analysis: Dict) -> List[str]:
        """Generate safety and regulatory warnings"""
        warnings = []
        
        # Allergen warnings
        if allergens_analysis.get('has_major_allergens'):
            warnings.append("WARNING: Contains major food allergens")
        
        # Deceptive branding warnings
        deceptive_risk = branding_analysis.get('overall_risk', 'low')
        if deceptive_risk == 'high':
            warnings.append("WARNING: Product may have deceptive branding practices")
        
        return warnings
    
    def _calculate_overall_processing_level(self, ingredients_analysis: Dict) -> int:
        """Calculate overall processing level (1-5 scale)"""
        ingredient_list = ingredients_analysis.get('ingredient_list', [])
        
        if not ingredient_list:
            return 3  # Default
        
        processing_levels = [ing.get('processing_level', 3) for ing in ingredient_list]
        avg_level = sum(processing_levels) / len(processing_levels)
        
        return round(avg_level)
