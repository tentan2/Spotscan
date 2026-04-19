"""
Artificial vs Natural Classifier Module
Categorizes ingredients as artificial or natural and flags deceptive branding
"""

import re
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json

class ArtificialNaturalClassifier:
    """Classifies ingredients as artificial or natural with comprehensive analysis"""
    
    def __init__(self):
        """Initialize the classifier with comprehensive databases"""
        self.natural_database = self._load_natural_database()
        self.artificial_database = self._load_artificial_database()
        self.chemical_patterns = self._load_chemical_patterns()
        self.processing_indicators = self._load_processing_indicators()
        self.deceptive_patterns = self._load_deceptive_patterns()
    
    def _load_natural_database(self) -> Dict:
        """Load comprehensive natural ingredients database"""
        return {
            'fruits': [
                'apple', 'banana', 'orange', 'strawberry', 'blueberry', 'raspberry', 'blackberry',
                'mango', 'pineapple', 'grape', 'peach', 'pear', 'cherry', 'kiwi', 'lemon', 'lime',
                'grapefruit', 'plum', 'apricot', 'fig', 'pomegranate', 'cranberry', 'watermelon',
                'cantaloupe', 'honeydew', 'coconut', 'avocado', 'papaya', 'guava', 'passion fruit'
            ],
            'vegetables': [
                'carrot', 'broccoli', 'spinach', 'tomato', 'potato', 'onion', 'garlic', 'pepper',
                'cucumber', 'lettuce', 'celery', 'corn', 'peas', 'beans', 'cabbage', 'cauliflower',
                'zucchini', 'squash', 'pumpkin', 'sweet potato', 'beet', 'radish', 'turnip',
                'parsnip', 'asparagus', 'brussels sprouts', 'kale', 'chard', 'collard greens',
                'mushroom', 'eggplant', 'bell pepper', 'chili pepper', 'jalapeno'
            ],
            'grains': [
                'wheat', 'rice', 'oats', 'barley', 'quinoa', 'corn', 'rye', 'millet', 'spelt',
                'kamut', 'farro', 'bulgur', 'couscous', 'amaranth', 'teff', 'buckwheat',
                'sorghum', 'wild rice', 'brown rice', 'white rice', 'jasmine rice', 'basmati rice'
            ],
            'proteins': [
                'chicken', 'beef', 'pork', 'lamb', 'turkey', 'fish', 'salmon', 'tuna', 'cod',
                'shrimp', 'crab', 'lobster', 'egg', 'milk', 'cheese', 'yogurt', 'butter',
                'cream', 'tofu', 'tempeh', 'seitan', 'beans', 'lentils', 'chickpeas', 'peas'
            ],
            'nuts': [
                'almond', 'walnut', 'pecan', 'cashew', 'pistachio', 'hazelnut', 'macadamia',
                'brazil nut', 'pine nut', 'chestnut', 'acorn'
            ],
            'seeds': [
                'sunflower', 'pumpkin', 'sesame', 'chia', 'flax', 'hemp', 'poppy', 'mustard'
            ],
            'herbs_spices': [
                'basil', 'oregano', 'thyme', 'rosemary', 'sage', 'parsley', 'cilantro', 'dill',
                'mint', 'chives', 'bay leaf', 'cinnamon', 'nutmeg', 'clove', 'ginger', 'turmeric',
                'paprika', 'cayenne', 'chili powder', 'cumin', 'coriander', 'cardamom', 'anise',
                'fennel', 'saffron', 'vanilla', 'allspice', 'pepper', 'salt'
            ],
            'natural_sweeteners': [
                'honey', 'maple syrup', 'agave', 'coconut sugar', 'date sugar', 'molasses',
                'brown sugar', 'raw sugar', 'cane sugar', 'fruit juice concentrate'
            ],
            'natural_fats': [
                'olive oil', 'coconut oil', 'avocado oil', 'butter', 'ghee', 'lard', 'tallow'
            ]
        }
    
    def _load_artificial_database(self) -> Dict:
        """Load comprehensive artificial ingredients database"""
        return {
            'artificial_colors': [
                'red 40', 'red 3', 'yellow 5', 'yellow 6', 'blue 1', 'blue 2', 'green 3',
                'tartrazine', 'sunset yellow', 'carmine', 'annatto', 'beta carotene (synthetic)',
                'allura red', 'brilliant blue', 'indigo carmine', 'citrus red', 'erythrosine'
            ],
            'artificial_preservatives': [
                'sodium benzoate', 'potassium sorbate', 'calcium propionate', 'sodium nitrite',
                'sodium nitrate', 'bha', 'bht', 'propyl gallate', 'tbhq', 'sodium bisulfite',
                'potassium bisulfite', 'sodium metabisulfite', 'potassium metabisulphite',
                'sulphur dioxide', 'calcium sulphite', 'disodium edta', 'calcium disodium edta'
            ],
            'artificial_sweeteners': [
                'aspartame', 'sucralose', 'acesulfame potassium', 'saccharin', 'neotame',
                'stevia (extract)', 'xylitol', 'sorbitol', 'mannitol', 'erythritol',
                'isomalt', 'maltitol', 'lactitol', 'hydrogenated starch hydrolysates'
            ],
            'artificial_flavors': [
                'artificial flavor', 'natural flavor (artificial source)', 'vanillin',
                'ethyl vanillin', 'maltol', 'ethyl maltol', 'dihydroxyacetophenone',
                'methyl anthranilate', 'ethyl butyrate', 'ethyl propionate', 'ethyl acetate'
            ],
            'artificial_emulsifiers': [
                'polysorbate 80', 'polysorbate 60', 'polysorbate 20', 'sodium stearoyl lactylate',
                'calcium stearoyl lactylate', 'datem', 'mono and diglycerides', 'sodium carboxymethylcellulose',
                'hydroxypropyl methylcellulose', 'microcrystalline cellulose'
            ],
            'artificial_fats': [
                'hydrogenated vegetable oil', 'partially hydrogenated oil', 'interesterified oil',
                'margarine', 'vegetable shortening', 'palm oil (refined)', 'canola oil (refined)',
                'soybean oil (refined)', 'corn oil (refined)', 'cottonseed oil', 'vegetable oil'
            ],
            'chemical_additives': [
                'monosodium glutamate', 'disodium inosinate', 'disodium guanylate',
                'sodium citrate', 'calcium chloride', 'magnesium chloride', 'potassium chloride',
                'sodium phosphate', 'calcium phosphate', 'sodium bicarbonate', 'calcium carbonate'
            ]
        }
    
    def _load_chemical_patterns(self) -> List[str]:
        """Load regex patterns for identifying chemical compounds"""
        return [
            r'\b(acid|ate|ite|ide|hydro|oxy|mono|di|tri|poly|tetra|penta|hexa)\b',
            r'\b(sodium|potassium|calcium|magnesium|ammonium)\b',
            r'\b(dextrose|fructose|glucose|sucrose|lactose|maltose|galactose)\b',
            r'\b(propylene|ethylene|glycol|stearate|laurate|oleate|palmitate)\b',
            r'\b(cellulose|pectin|gum|starch|protein|isolat|concentrat)\b',
            r'\b(extract|oil|fat|wax|resin|latex)\b',
            r'\b(carbon|sulfate|nitrate|chloride|oxide|hydroxide)\b'
        ]
    
    def _load_processing_indicators(self) -> Dict:
        """Load indicators of processing level"""
        return {
            'minimally_processed': [
                'fresh', 'raw', 'whole', 'dried', 'frozen', 'chopped', 'cut', 'sliced'
            ],
            'moderately_processed': [
                'roasted', 'toasted', 'steamed', 'boiled', 'baked', 'grilled', 'fried',
                'pureed', 'crushed', 'ground', 'milled', 'pressed', 'extracted'
            ],
            'highly_processed': [
                'hydrogenated', 'fractionated', 'interesterified', 'refined', 'bleached',
                'deodorized', 'solvent extracted', 'chemically extracted', 'synthetic',
                'artificial', 'imitation', 'flavor', 'enhanced'
            ]
        }
    
    def _load_deceptive_patterns(self) -> Dict:
        """Load patterns for detecting deceptive branding"""
        return {
            'flavor_simulants': [
                r'(\w+)\s+flavored',
                r'(\w+)\s+flavour',
                r'(\w+)\s+flavoring',
                r'(\w+)\s+flavouring',
                r'artificial\s+(\w+)\s+flavor',
                r'natural\s+(\w+)\s+flavor'
            ],
            'misleading_health_claims': [
                r'natural\s+ingredients?\s+only',
                r'no\s+artificial\s+(?:flavors?|colors?|preservatives?)',
                r'made\s+with\s+real\s+(\w+)',
                r'contains\s+real\s+(\w+)',
                r'(\w+)\s+juice\s+blend',
                r'(\w+)\s+concentrate'
            ],
            'hidden_artificial': [
                r'natural\s+flavors?',  # Often contains artificial components
                r'artificial\s+colors?',
                r'preserved\s+with',
                r'enhanced\s+with'
            ]
        }
    
    def classify_ingredients(self, ingredients_list: List[str]) -> Dict:
        """
        Classify a list of ingredients as artificial or natural
        
        Args:
            ingredients_list: List of ingredient names
            
        Returns:
            Comprehensive classification results
        """
        classification_results = []
        
        for ingredient in ingredients_list:
            classification = self._classify_single_ingredient(ingredient)
            classification_results.append(classification)
        
        # Calculate overall statistics
        overall_stats = self._calculate_overall_statistics(classification_results)
        
        # Detect deceptive branding patterns
        deceptive_analysis = self._detect_deceptive_branding(ingredients_list, classification_results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(overall_stats, deceptive_analysis)
        
        return {
            'individual_classifications': classification_results,
            'overall_statistics': overall_stats,
            'deceptive_analysis': deceptive_analysis,
            'recommendations': recommendations
        }
    
    def _classify_single_ingredient(self, ingredient: str) -> Dict:
        """Classify a single ingredient"""
        ingredient_lower = ingredient.lower().strip()
        
        # Check against natural database
        natural_match = self._check_natural_database(ingredient_lower)
        
        # Check against artificial database
        artificial_match = self._check_artificial_database(ingredient_lower)
        
        # Check chemical patterns
        chemical_match = self._check_chemical_patterns(ingredient_lower)
        
        # Determine primary classification
        classification = self._determine_classification(
            natural_match, artificial_match, chemical_match, ingredient_lower
        )
        
        # Assess processing level
        processing_level = self._assess_processing_level(ingredient_lower)
        
        # Check for E-numbers
        e_number = self._extract_e_number(ingredient_lower)
        
        return {
            'original_name': ingredient,
            'normalized_name': ingredient_lower,
            'primary_classification': classification['type'],
            'confidence': classification['confidence'],
            'natural_match': natural_match,
            'artificial_match': artificial_match,
            'chemical_indicators': chemical_match,
            'processing_level': processing_level,
            'e_number': e_number,
            'concerns': self._identify_concerns(classification, e_number)
        }
    
    def _check_natural_database(self, ingredient: str) -> Optional[Dict]:
        """Check if ingredient matches natural database"""
        for category, items in self.natural_database.items():
            for item in items:
                if item in ingredient:
                    return {
                        'category': category,
                        'matched_item': item,
                        'match_type': 'exact' if item == ingredient else 'partial'
                    }
        return None
    
    def _check_artificial_database(self, ingredient: str) -> Optional[Dict]:
        """Check if ingredient matches artificial database"""
        for category, items in self.artificial_database.items():
            for item in items:
                if item in ingredient:
                    return {
                        'category': category,
                        'matched_item': item,
                        'match_type': 'exact' if item == ingredient else 'partial'
                    }
        return None
    
    def _check_chemical_patterns(self, ingredient: str) -> List[str]:
        """Check for chemical pattern matches"""
        matches = []
        for pattern in self.chemical_patterns:
            if re.search(pattern, ingredient):
                matches.append(pattern)
        return matches
    
    def _determine_classification(self, natural_match: Optional[Dict], 
                                 artificial_match: Optional[Dict],
                                 chemical_indicators: List[str],
                                 ingredient: str) -> Dict:
        """Determine primary classification with confidence"""
        if artificial_match:
            return {'type': 'artificial', 'confidence': 0.9}
        elif natural_match:
            # Check if it's modified natural
            if chemical_indicators:
                return {'type': 'modified_natural', 'confidence': 0.7}
            else:
                return {'type': 'natural', 'confidence': 0.9}
        elif chemical_indicators:
            if len(chemical_indicators) >= 2:
                return {'type': 'artificial', 'confidence': 0.8}
            else:
                return {'type': 'modified_natural', 'confidence': 0.6}
        else:
            # Unknown - use heuristics
            if self._is_likely_artificial(ingredient):
                return {'type': 'artificial', 'confidence': 0.5}
            else:
                return {'type': 'natural', 'confidence': 0.4}
    
    def _is_likely_artificial(self, ingredient: str) -> bool:
        """Use heuristics to determine if unknown ingredient is likely artificial"""
        # Long chemical-sounding names are likely artificial
        if len(ingredient) > 15:
            return True
        
        # Contains numbers
        if re.search(r'\d', ingredient):
            return True
        
        # Contains many chemical-sounding syllables
        chemical_syllables = ['hydro', 'oxy', 'prop', 'ethyl', 'methyl', 'sodium', 'potassium']
        chemical_count = sum(1 for syllable in chemical_syllables if syllable in ingredient)
        if chemical_count >= 2:
            return True
        
        return False
    
    def _assess_processing_level(self, ingredient: str) -> int:
        """Assess processing level (1-5 scale)"""
        for indicator in self.processing_indicators['highly_processed']:
            if indicator in ingredient:
                return 5
        
        for indicator in self.processing_indicators['moderately_processed']:
            if indicator in ingredient:
                return 3
        
        for indicator in self.processing_indicators['minimally_processed']:
            if indicator in ingredient:
                return 1
        
        # Default based on classification
        return 3
    
    def _extract_e_number(self, ingredient: str) -> Optional[str]:
        """Extract E-number from ingredient"""
        match = re.search(r'e\d{3,4}', ingredient)
        return match.group(0).upper() if match else None
    
    def _identify_concerns(self, classification: Dict, e_number: Optional[str]) -> List[str]:
        """Identify potential concerns with ingredient"""
        concerns = []
        
        if classification['type'] == 'artificial':
            concerns.append('artificial_ingredient')
        
        if e_number:
            concerns.append('contains_e_number')
        
        if classification['confidence'] < 0.6:
            concerns.append('uncertain_classification')
        
        return concerns
    
    def _calculate_overall_statistics(self, classifications: List[Dict]) -> Dict:
        """Calculate overall classification statistics"""
        total = len(classifications)
        
        # Count classifications
        natural_count = sum(1 for c in classifications if c['primary_classification'] == 'natural')
        artificial_count = sum(1 for c in classifications if c['primary_classification'] == 'artificial')
        modified_natural_count = sum(1 for c in classifications if c['primary_classification'] == 'modified_natural')
        
        # Calculate percentages
        natural_percentage = (natural_count / total) * 100 if total > 0 else 0
        artificial_percentage = (artificial_count / total) * 100 if total > 0 else 0
        modified_natural_percentage = (modified_natural_count / total) * 100 if total > 0 else 0
        
        # Calculate average processing level
        avg_processing = sum(c['processing_level'] for c in classifications) / total if total > 0 else 0
        
        # Count E-numbers
        e_number_count = sum(1 for c in classifications if c['e_number'])
        
        return {
            'total_ingredients': total,
            'natural_count': natural_count,
            'artificial_count': artificial_count,
            'modified_natural_count': modified_natural_count,
            'natural_percentage': natural_percentage,
            'artificial_percentage': artificial_percentage,
            'modified_natural_percentage': modified_natural_percentage,
            'average_processing_level': avg_processing,
            'e_number_count': e_number_count,
            'overall_purity_score': max(0, natural_percentage - artificial_percentage)
        }
    
    def _detect_deceptive_branding(self, ingredients_list: List[str], 
                                 classifications: List[Dict]) -> Dict:
        """Detect deceptive branding patterns"""
        deceptive_indicators = []
        full_text = ' '.join(ingredients_list).lower()
        
        # Check for flavor simulants
        for pattern in self.deceptive_patterns['flavor_simulants']:
            matches = re.finditer(pattern, full_text)
            for match in matches:
                flavor_name = match.group(1) if match.groups() else 'unknown'
                
                # Check if the actual flavor ingredient is present
                flavor_present = any(flavor_name in ing.lower() for ing in ingredients_list)
                
                if not flavor_present:
                    deceptive_indicators.append({
                        'type': 'flavor_simulant',
                        'description': f'"{match.group(0)}" but no {flavor_name} found in ingredients',
                        'severity': 'high',
                        'pattern': pattern
                    })
        
        # Check for misleading health claims
        for pattern in self.deceptive_patterns['misleading_health_claims']:
            matches = re.finditer(pattern, full_text)
            for match in matches:
                deceptive_indicators.append({
                    'type': 'misleading_health_claim',
                    'description': f'Potential misleading claim: "{match.group(0)}"',
                    'severity': 'medium',
                    'pattern': pattern
                })
        
        # Check for hidden artificial ingredients
        artificial_count = sum(1 for c in classifications if c['primary_classification'] == 'artificial')
        if artificial_count > 0:
            for pattern in self.deceptive_patterns['hidden_artificial']:
                if re.search(pattern, full_text):
                    deceptive_indicators.append({
                        'type': 'hidden_artificial',
                        'description': f'Claims natural but contains {artificial_count} artificial ingredients',
                        'severity': 'high',
                        'pattern': pattern
                    })
        
        # Check for "natural flavors" that might be artificial
        if 'natural flavor' in full_text:
            deceptive_indicators.append({
                'type': 'questionable_natural_flavor',
                'description': '"Natural flavor" may contain artificial components',
                'severity': 'medium',
                'pattern': 'natural flavor'
            })
        
        return {
            'indicators': deceptive_indicators,
            'total_indicators': len(deceptive_indicators),
            'has_deceptive_branding': len(deceptive_indicators) > 0,
            'overall_risk': self._assess_deceptive_risk(deceptive_indicators)
        }
    
    def _assess_deceptive_risk(self, indicators: List[Dict]) -> str:
        """Assess overall deceptive branding risk"""
        if not indicators:
            return 'low'
        
        high_severity_count = sum(1 for i in indicators if i['severity'] == 'high')
        medium_severity_count = sum(1 for i in indicators if i['severity'] == 'medium')
        
        if high_severity_count >= 2:
            return 'high'
        elif high_severity_count >= 1 or medium_severity_count >= 3:
            return 'medium'
        else:
            return 'low'
    
    def _generate_recommendations(self, stats: Dict, deceptive: Dict) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Natural vs artificial recommendations
        if stats['artificial_percentage'] > 30:
            recommendations.append("Consider choosing products with fewer artificial ingredients")
        elif stats['artificial_percentage'] > 50:
            recommendations.append("This product contains a high percentage of artificial ingredients")
        
        # Processing level recommendations
        if stats['average_processing_level'] >= 4:
            recommendations.append("Product is highly processed - consider less processed alternatives")
        
        # E-number recommendations
        if stats['e_number_count'] > 3:
            recommendations.append("Product contains many E-number additives - check for necessity")
        
        # Deceptive branding recommendations
        if deceptive['has_deceptive_branding']:
            if deceptive['overall_risk'] == 'high':
                recommendations.append("WARNING: Product may have deceptive labeling practices")
            else:
                recommendations.append("Be aware of potentially misleading claims on this product")
        
        # Positive recommendations
        if stats['natural_percentage'] > 80:
            recommendations.append("Product contains mostly natural ingredients")
        
        if stats['overall_purity_score'] > 50:
            recommendations.append("Good balance of natural vs artificial ingredients")
        
        return recommendations
    
    def analyze_flavor_simulant_compliance(self, product_name: str, ingredients_list: List[str]) -> Dict:
        """
        Analyze compliance with proposed flavor simulant labeling
        
        Args:
            product_name: Name of the product
            ingredients_list: List of ingredients
            
        Returns:
            Flavor simulant compliance analysis
        """
        # Extract flavor claims from product name
        flavor_claims = self._extract_flavor_claims(product_name)
        
        # Analyze each claim
        compliance_results = []
        
        for claim in flavor_claims:
            compliance = self._check_flavor_claim_compliance(claim, ingredients_list)
            compliance_results.append(compliance)
        
        # Overall compliance assessment
        overall_compliance = self._assess_overall_compliance(compliance_results)
        
        return {
            'product_name': product_name,
            'flavor_claims': flavor_claims,
            'compliance_results': compliance_results,
            'overall_compliance': overall_compliance,
            'recommended_labeling': self._generate_recommended_labeling(compliance_results)
        }
    
    def _extract_flavor_claims(self, product_name: str) -> List[Dict]:
        """Extract flavor claims from product name"""
        claims = []
        name_lower = product_name.lower()
        
        # Look for flavor patterns
        flavor_patterns = [
            r'(\w+)\s+(?:flavor|flavour)',
            r'(\w+)\s+(?:flavored|flavoured)',
            r'(\w+)\s+(?:taste|tasting)',
            r'(\w+)\s+(?:style|styled)'
        ]
        
        for pattern in flavor_patterns:
            matches = re.finditer(pattern, name_lower)
            for match in matches:
                flavor_name = match.group(1)
                claims.append({
                    'flavor': flavor_name,
                    'claim_text': match.group(0),
                    'position': match.start()
                })
        
        return claims
    
    def _check_flavor_claim_compliance(self, claim: Dict, ingredients_list: List[str]) -> Dict:
        """Check if a flavor claim complies with proposed regulations"""
        flavor_name = claim['flavor']
        ingredients_lower = [ing.lower() for ing in ingredients_list]
        
        # Check for actual flavor ingredient
        flavor_ingredients = self._get_flavor_ingredients(flavor_name)
        found_flavors = [ing for ing in ingredients_lower if any(flavor in ing for flavor in flavor_ingredients)]
        
        # Calculate flavor contribution percentage
        flavor_percentage = self._estimate_flavor_percentage(found_flavors, ingredients_list)
        
        # Determine compliance
        if flavor_percentage >= 50:
            compliance_status = 'compliant'
            requires_simulant_label = False
        elif flavor_percentage > 0:
            compliance_status = 'non_compliant'
            requires_simulant_label = True
        else:
            compliance_status = 'non_compliant'
            requires_simulant_label = True
        
        return {
            'claim': claim,
            'flavor_ingredients_found': found_flavors,
            'estimated_flavor_percentage': flavor_percentage,
            'compliance_status': compliance_status,
            'requires_simulant_label': requires_simulant_label,
            'recommended_action': self._get_recommended_action(compliance_status, flavor_percentage)
        }
    
    def _get_flavor_ingredients(self, flavor_name: str) -> List[str]:
        """Get list of ingredients that would provide the specified flavor"""
        flavor_map = {
            'strawberry': ['strawberry', 'strawberries', 'strawberry puree', 'strawberry concentrate'],
            'banana': ['banana', 'bananas', 'banana puree', 'banana concentrate'],
            'orange': ['orange', 'oranges', 'orange juice', 'orange concentrate', 'orange oil'],
            'apple': ['apple', 'apples', 'apple juice', 'apple concentrate', 'apple puree'],
            'grape': ['grape', 'grapes', 'grape juice', 'grape concentrate'],
            'cherry': ['cherry', 'cherries', 'cherry juice', 'cherry concentrate'],
            'peach': ['peach', 'peaches', 'peach juice', 'peach concentrate', 'peach puree'],
            'lemon': ['lemon', 'lemons', 'lemon juice', 'lemon concentrate', 'lemon oil'],
            'lime': ['lime', 'limes', 'lime juice', 'lime concentrate', 'lime oil'],
            'vanilla': ['vanilla', 'vanilla extract', 'vanilla bean', 'vanilla powder'],
            'chocolate': ['cocoa', 'chocolate', 'cocoa powder', 'chocolate liquor'],
            'coffee': ['coffee', 'coffee extract', 'coffee powder'],
            'caramel': ['caramel', 'caramel color', 'burnt sugar']
        }
        
        return flavor_map.get(flavor_name, [flavor_name])
    
    def _estimate_flavor_percentage(self, flavor_ingredients: List[str], 
                                 total_ingredients: List[str]) -> float:
        """Estimate percentage of flavor-contributing ingredients"""
        if not flavor_ingredients:
            return 0
        
        # Simple estimation based on ingredient position
        # (earlier in list = higher percentage)
        total_count = len(total_ingredients)
        flavor_positions = []
        
        for flavor_ing in flavor_ingredients:
            for i, ing in enumerate(total_ingredients):
                if flavor_ing in ing.lower():
                    flavor_positions.append(i + 1)  # 1-based position
                    break
        
        if not flavor_positions:
            return 0
        
        # Average position of flavor ingredients
        avg_position = sum(flavor_positions) / len(flavor_positions)
        
        # Estimate percentage based on position
        # Earlier positions contribute more
        estimated_percentage = max(0, (total_count - avg_position + 1) / total_count * 100)
        
        return estimated_percentage
    
    def _get_recommended_action(self, compliance_status: str, flavor_percentage: float) -> str:
        """Get recommended action for non-compliant claims"""
        if compliance_status == 'compliant':
            return "No action required"
        elif flavor_percentage == 0:
            return "Remove flavor claim or add actual flavor ingredient"
        elif flavor_percentage < 50:
            return "Label as 'flavor simulant' per proposed regulations"
        else:
            return "Verify flavor ingredient percentages"
    
    def _assess_overall_compliance(self, compliance_results: List[Dict]) -> Dict:
        """Assess overall compliance for all flavor claims"""
        if not compliance_results:
            return {'status': 'no_claims', 'compliant': True}
        
        compliant_count = sum(1 for r in compliance_results if r['compliance_status'] == 'compliant')
        total_count = len(compliance_results)
        
        if compliant_count == total_count:
            status = 'fully_compliant'
        elif compliant_count > 0:
            status = 'partially_compliant'
        else:
            status = 'non_compliant'
        
        return {
            'status': status,
            'compliant': compliant_count == total_count,
            'compliance_percentage': (compliant_count / total_count) * 100,
            'requires_labeling_changes': any(r['requires_simulant_label'] for r in compliance_results)
        }
    
    def _generate_recommended_labeling(self, compliance_results: List[Dict]) -> List[str]:
        """Generate recommended labeling changes"""
        recommendations = []
        
        for result in compliance_results:
            if result['requires_simulant_label']:
                claim = result['claim']['claim_text']
                recommendations.append(f'Label "{claim}" as "FLAVOR SIMULANT" in bold, size 12 font, yellow color')
        
        if recommendations:
            recommendations.append("Add disclaimer: 'This product contains artificial flavors'")
        
        return recommendations
