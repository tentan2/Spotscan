"""
Health & Safety Warnings Module
Flags high sodium, sugar, artificial ingredients, and other health concerns
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import re
from pathlib import Path
import json

class SafetyChecker:
    """Analyzes food for health and safety warnings"""
    
    def __init__(self):
        """Initialize safety checker"""
        # Health thresholds and limits
        self.health_thresholds = {
            'sodium': {
                'daily_limit': 2300,  # mg per day
                'high_threshold': 600,  # mg per serving
                'very_high_threshold': 1000,  # mg per serving
                'warnings': {
                    'high': 'High sodium content - may contribute to high blood pressure',
                    'very_high': 'Very high sodium - exceeds daily recommended intake'
                }
            },
            'sugar': {
                'daily_limit': 50,  # g per day
                'high_threshold': 15,  # g per serving
                'very_high_threshold': 25,  # g per serving
                'warnings': {
                    'high': 'High sugar content - may contribute to weight gain and diabetes',
                    'very_high': 'Very high sugar - exceeds daily recommended intake'
                }
            },
            'saturated_fat': {
                'daily_limit': 20,  # g per day
                'high_threshold': 5,  # g per serving
                'very_high_threshold': 10,  # g per serving
                'warnings': {
                    'high': 'High saturated fat - may contribute to heart disease',
                    'very_high': 'Very high saturated fat - exceeds daily recommended intake'
                }
            },
            'calories': {
                'daily_limit': 2000,  # calories per day
                'high_threshold': 400,  # calories per serving
                'very_high_threshold': 600,  # calories per serving
                'warnings': {
                    'high': 'High calorie content - may contribute to weight gain',
                    'very_high': 'Very high calorie - exceeds recommended meal portion'
                }
            }
        }
        
        # Artificial ingredients to flag
        self.artificial_ingredients = {
            'artificial_colors': [
                'red 40', 'yellow 5', 'blue 1', 'tartrazine', 'sunset yellow',
                'carmine', 'annatto', 'beta carotene (synthetic)', 'allura red',
                'brilliant blue', 'indigo carmine', 'citrus red'
            ],
            'artificial_preservatives': [
                'sodium benzoate', 'potassium sorbate', 'calcium propionate',
                'sodium nitrite', 'sodium nitrate', 'bha', 'bht', 'propyl gallate',
                'tbhq', 'sodium bisulfite', 'potassium bisulfite'
            ],
            'artificial_sweeteners': [
                'aspartame', 'sucralose', 'acesulfame potassium', 'saccharin',
                'neotame', 'stevia (extract)', 'xylitol', 'sorbitol', 'mannitol'
            ],
            'artificial_flavors': [
                'artificial flavor', 'natural flavor (artificial source)', 'vanillin',
                'ethyl vanillin', 'maltol', 'ethyl maltol', 'dihydroxyacetophenone'
            ]
        }
        
        # Allergen warnings
        self.major_allergens = [
            'milk', 'egg', 'fish', 'shellfish', 'tree nuts', 'peanuts',
            'wheat', 'soybean', 'sesame'
        ]
        
        # Safety concerns by category
        self.safety_concerns = {
            'highly_processed': [
                'Contains ultra-processed ingredients',
                'May contain artificial additives',
                'Low nutritional value relative to calories'
            ],
            'high_sodium': [
                'May contribute to high blood pressure',
                'Risk for cardiovascular disease',
                'Not suitable for low-sodium diets'
            ],
            'high_sugar': [
                'May contribute to weight gain',
                'Risk for type 2 diabetes',
                'Can cause blood sugar spikes'
            ],
            'artificial_additives': [
                'May cause allergic reactions',
                'Potential behavioral effects in children',
                'Long-term health effects uncertain'
            ],
            'allergens': [
                'May cause severe allergic reactions',
                'Risk of anaphylaxis for sensitive individuals',
                'Cross-contamination possible'
            ]
        }
        
        # Warning levels
        self.warning_levels = {
            'info': {
                'color': 'blue',
                'severity': 'low',
                'action': 'informational'
            },
            'caution': {
                'color': 'yellow',
                'severity': 'medium',
                'action': 'moderate_concern'
            },
            'warning': {
                'color': 'orange',
                'severity': 'high',
                'action': 'significant_concern'
            },
            'danger': {
                'color': 'red',
                'severity': 'critical',
                'action': 'avoid_or_limit'
            }
        }
    
    def analyze_safety(self, nutrition_data: Dict, ingredients_list: Optional[List[str]] = None,
                      food_type: Optional[str] = None) -> Dict:
        """
        Analyze food for health and safety warnings
        
        Args:
            nutrition_data: Nutritional information
            ingredients_list: List of ingredients
            food_type: Type of food
            
        Returns:
            Comprehensive safety analysis
        """
        warnings = []
        
        # Analyze nutritional concerns
        nutrition_warnings = self._analyze_nutritional_concerns(nutrition_data)
        warnings.extend(nutrition_warnings)
        
        # Analyze ingredient concerns
        if ingredients_list:
            ingredient_warnings = self._analyze_ingredient_concerns(ingredients_list)
            warnings.extend(ingredient_warnings)
        
        # Analyze allergen concerns
        if ingredients_list:
            allergen_warnings = self._analyze_allergen_concerns(ingredients_list)
            warnings.extend(allergen_warnings)
        
        # Analyze processing level concerns
        processing_warnings = self._analyze_processing_concerns(ingredients_list, food_type)
        warnings.extend(processing_warnings)
        
        # Analyze food type specific concerns
        if food_type:
            food_type_warnings = self._analyze_food_type_concerns(food_type)
            warnings.extend(food_type_warnings)
        
        # Categorize warnings by severity
        categorized_warnings = self._categorize_warnings(warnings)
        
        # Generate overall safety assessment
        safety_assessment = self._generate_safety_assessment(categorized_warnings)
        
        # Generate recommendations
        recommendations = self._generate_safety_recommendations(categorized_warnings, food_type)
        
        return {
            'food_type': food_type,
            'warnings': warnings,
            'categorized_warnings': categorized_warnings,
            'safety_assessment': safety_assessment,
            'recommendations': recommendations,
            'summary': self._generate_safety_summary(categorized_warnings)
        }
    
    def _analyze_nutritional_concerns(self, nutrition_data: Dict) -> List[Dict]:
        """Analyze nutritional data for health concerns"""
        warnings = []
        
        for nutrient, threshold_info in self.health_thresholds.items():
            if nutrient in nutrition_data:
                amount = nutrition_data[nutrient]
                
                # Check against thresholds
                if amount >= threshold_info['very_high_threshold']:
                    level = 'danger'
                    message = threshold_info['warnings']['very_high']
                elif amount >= threshold_info['high_threshold']:
                    level = 'warning'
                    message = threshold_info['warnings']['high']
                else:
                    level = 'info'
                    message = f"Moderate {nutrient} content"
                
                # Calculate percentage of daily limit
                daily_percentage = (amount / threshold_info['daily_limit']) * 100
                
                warnings.append({
                    'type': 'nutritional',
                    'category': nutrient,
                    'level': level,
                    'amount': amount,
                    'daily_percentage': daily_percentage,
                    'message': message,
                    'threshold': threshold_info['high_threshold'],
                    'daily_limit': threshold_info['daily_limit']
                })
        
        return warnings
    
    def _analyze_ingredient_concerns(self, ingredients_list: List[str]) -> List[Dict]:
        """Analyze ingredients for artificial additives and concerns"""
        warnings = []
        ingredients_text = ' '.join(ingredients_list).lower()
        
        for category, additives in self.artificial_ingredients.items():
            detected_additives = []
            
            for additive in additives:
                if additive in ingredients_text:
                    detected_additives.append(additive)
            
            if detected_additives:
                # Determine warning level based on category and count
                if category == 'artificial_colors':
                    level = 'warning'  # Artificial colors are concerning
                    message = f"Contains artificial colors: {', '.join(detected_additives)}"
                elif category == 'artificial_preservatives':
                    level = 'caution'  # Preservatives are moderately concerning
                    message = f"Contains artificial preservatives: {', '.join(detected_additives)}"
                elif category == 'artificial_sweeteners':
                    level = 'caution'  # Sweeteners are moderately concerning
                    message = f"Contains artificial sweeteners: {', '.join(detected_additives)}"
                elif category == 'artificial_flavors':
                    level = 'info'  # Flavors are less concerning
                    message = f"Contains artificial flavors: {', '.join(detected_additives)}"
                else:
                    level = 'info'
                    message = f"Contains additives: {', '.join(detected_additives)}"
                
                warnings.append({
                    'type': 'ingredient',
                    'category': category,
                    'level': level,
                    'detected_additives': detected_additives,
                    'count': len(detected_additives),
                    'message': message
                })
        
        return warnings
    
    def _analyze_allergen_concerns(self, ingredients_list: List[str]) -> List[Dict]:
        """Analyze ingredients for allergen concerns"""
        warnings = []
        ingredients_text = ' '.join(ingredients_list).lower()
        
        detected_allergens = []
        
        for allergen in self.major_allergens:
            if allergen in ingredients_text:
                detected_allergens.append(allergen)
        
        if detected_allergens:
            # Allergens are always high priority warnings
            level = 'danger'
            message = f"Contains major allergens: {', '.join(detected_allergens)}"
            
            warnings.append({
                'type': 'allergen',
                'category': 'allergens',
                'level': level,
                'detected_allergens': detected_allergens,
                'count': len(detected_allergens),
                'message': message,
                'requires_labeling': True
            })
        
        return warnings
    
    def _analyze_processing_concerns(self, ingredients_list: Optional[List[str]], 
                                   food_type: Optional[str]) -> List[Dict]:
        """Analyze processing level concerns"""
        warnings = []
        
        if ingredients_list:
            # Count artificial vs natural ingredients
            artificial_count = 0
            total_count = len(ingredients_list)
            
            for ingredient in ingredients_list:
                ing_lower = ingredient.lower()
                
                # Check if ingredient is artificial
                is_artificial = False
                for category, additives in self.artificial_ingredients.items():
                    for additive in additives:
                        if additive in ing_lower:
                            artificial_count += 1
                            is_artificial = True
                            break
                    if is_artificial:
                        break
            
            artificial_percentage = (artificial_count / total_count) * 100 if total_count > 0 else 0
            
            # Processing level warnings
            if artificial_percentage > 50:
                level = 'warning'
                message = f"Highly processed food ({artificial_percentage:.1f}% artificial ingredients)"
            elif artificial_percentage > 25:
                level = 'caution'
                message = f"Moderately processed food ({artificial_percentage:.1f}% artificial ingredients)"
            elif artificial_percentage > 10:
                level = 'info'
                message = f"Some processing detected ({artificial_percentage:.1f}% artificial ingredients)"
            else:
                level = 'info'
                message = "Minimally processed food"
            
            warnings.append({
                'type': 'processing',
                'category': 'processing_level',
                'level': level,
                'artificial_percentage': artificial_percentage,
                'artificial_count': artificial_count,
                'total_ingredients': total_count,
                'message': message
            })
        
        return warnings
    
    def _analyze_food_type_concerns(self, food_type: str) -> List[Dict]:
        """Analyze food type specific concerns"""
        warnings = []
        food_type_lower = food_type.lower()
        
        # Food type specific warnings
        food_concerns = {
            'energy_drink': {
                'level': 'warning',
                'message': 'High caffeine content - not recommended for children or pregnant women',
                'concerns': ['caffeine', 'sugar', 'stimulants']
            },
            'diet_soda': {
                'level': 'caution',
                'message': 'Contains artificial sweeteners - long-term health effects debated',
                'concerns': ['artificial_sweeteners', 'chemical_additives']
            },
            'processed_meat': {
                'level': 'warning',
                'message': 'Processed meat linked to increased cancer risk',
                'concerns': ['nitrates', 'preservatives', 'cancer_risk']
            },
            'microwave_meal': {
                'level': 'caution',
                'message': 'Highly processed with high sodium content',
                'concerns': ['high_sodium', 'preservatives', 'processing']
            },
            'candy': {
                'level': 'warning',
                'message': 'High sugar content with minimal nutritional value',
                'concerns': ['high_sugar', 'artificial_colors', 'empty_calories']
            },
            'fast_food': {
                'level': 'caution',
                'message': 'High in calories, sodium, and unhealthy fats',
                'concerns': ['high_calories', 'high_sodium', 'unhealthy_fats']
            }
        }
        
        for food_key, concern_info in food_concerns.items():
            if food_key in food_type_lower:
                warnings.append({
                    'type': 'food_type',
                    'category': food_key,
                    'level': concern_info['level'],
                    'message': concern_info['message'],
                    'concerns': concern_info['concerns']
                })
        
        return warnings
    
    def _categorize_warnings(self, warnings: List[Dict]) -> Dict:
        """Categorize warnings by severity level"""
        categorized = {
            'info': [],
            'caution': [],
            'warning': [],
            'danger': []
        }
        
        for warning in warnings:
            level = warning.get('level', 'info')
            categorized[level].append(warning)
        
        return categorized
    
    def _generate_safety_assessment(self, categorized_warnings: Dict) -> Dict:
        """Generate overall safety assessment"""
        total_warnings = sum(len(warnings) for warnings in categorized_warnings.values())
        danger_count = len(categorized_warnings['danger'])
        warning_count = len(categorized_warnings['warning'])
        caution_count = len(categorized_warnings['caution'])
        
        # Determine overall safety level
        if danger_count > 0:
            overall_level = 'danger'
            safety_score = 0.0
            message = 'Significant health and safety concerns detected'
        elif warning_count > 2:
            overall_level = 'warning'
            safety_score = 0.3
            message = 'Multiple health concerns - consume with caution'
        elif warning_count > 0 or caution_count > 3:
            overall_level = 'caution'
            safety_score = 0.6
            message = 'Some health concerns - moderate consumption recommended'
        elif caution_count > 0:
            overall_level = 'info'
            safety_score = 0.8
            message = 'Minor health concerns - generally safe in moderation'
        else:
            overall_level = 'safe'
            safety_score = 1.0
            message = 'No significant health concerns detected'
        
        return {
            'overall_level': overall_level,
            'safety_score': safety_score,
            'message': message,
            'total_warnings': total_warnings,
            'warning_counts': {
                'danger': danger_count,
                'warning': warning_count,
                'caution': caution_count,
                'info': len(categorized_warnings['info'])
            }
        }
    
    def _generate_safety_recommendations(self, categorized_warnings: Dict, 
                                       food_type: Optional[str]) -> List[str]:
        """Generate safety recommendations based on warnings"""
        recommendations = []
        
        # General recommendations based on warning levels
        if categorized_warnings['danger']:
            recommendations.append("Avoid consumption or consult healthcare provider")
            recommendations.append("Check for allergen labeling requirements")
        
        if categorized_warnings['warning']:
            recommendations.append("Limit consumption to occasional use")
            recommendations.append("Consider healthier alternatives")
        
        if categorized_warnings['caution']:
            recommendations.append("Consume in moderation")
            recommendations.append("Monitor portion sizes")
        
        # Specific recommendations based on warning types
        all_warnings = []
        for warnings in categorized_warnings.values():
            all_warnings.extend(warnings)
        
        # Nutritional recommendations
        nutritional_warnings = [w for w in all_warnings if w.get('type') == 'nutritional']
        for warning in nutritional_warnings:
            category = warning.get('category', '')
            if 'sodium' in category:
                recommendations.append("Choose low-sodium alternatives")
                recommendations.append("Drink plenty of water")
            elif 'sugar' in category:
                recommendations.append("Limit added sugars")
                recommendations.append("Choose whole fruit alternatives")
            elif 'saturated_fat' in category:
                recommendations.append("Choose unsaturated fats")
                recommendations.append("Limit saturated fat intake")
            elif 'calories' in category:
                recommendations.append("Watch portion sizes")
                recommendations.append("Balance with physical activity")
        
        # Ingredient recommendations
        ingredient_warnings = [w for w in all_warnings if w.get('type') == 'ingredient']
        for warning in ingredient_warnings:
            category = warning.get('category', '')
            if 'artificial_colors' in category:
                recommendations.append("Look for natural color alternatives")
                recommendations.append("May affect sensitive individuals")
            elif 'artificial_sweeteners' in category:
                recommendations.append("Consider natural sweeteners")
                recommendations.append("Monitor for adverse reactions")
            elif 'artificial_preservatives' in category:
                recommendations.append("Choose fresh alternatives when possible")
                recommendations.append("Look for preservative-free options")
        
        # Allergen recommendations
        allergen_warnings = [w for w in all_warnings if w.get('type') == 'allergen']
        if allergen_warnings:
            recommendations.append("Check for allergen labeling")
            recommendations.append("Avoid if allergic to any detected allergens")
            recommendations.append("Be aware of cross-contamination risks")
        
        # Processing recommendations
        processing_warnings = [w for w in all_warnings if w.get('type') == 'processing']
        if processing_warnings:
            recommendations.append("Choose minimally processed alternatives")
            recommendations.append("Read ingredient labels carefully")
            recommendations.append("Consider whole food options")
        
        # Remove duplicates
        recommendations = list(set(recommendations))
        
        return recommendations
    
    def _generate_safety_summary(self, categorized_warnings: Dict) -> Dict:
        """Generate safety summary"""
        total_warnings = sum(len(warnings) for warnings in categorized_warnings.values())
        
        summary = {
            'total_warnings': total_warnings,
            'most_concerning_level': None,
            'key_concerns': [],
            'action_required': False
        }
        
        # Find most concerning level
        for level in ['danger', 'warning', 'caution', 'info']:
            if categorized_warnings[level]:
                summary['most_concerning_level'] = level
                break
        
        # Extract key concerns
        all_warnings = []
        for warnings in categorized_warnings.values():
            all_warnings.extend(warnings)
        
        # Group concerns by type
        concern_types = {}
        for warning in all_warnings:
            warning_type = warning.get('type', 'unknown')
            if warning_type not in concern_types:
                concern_types[warning_type] = []
            concern_types[warning_type].append(warning['message'])
        
        summary['key_concerns'] = concern_types
        
        # Determine if action is required
        summary['action_required'] = (
            len(categorized_warnings['danger']) > 0 or
            len(categorized_warnings['warning']) > 1
        )
        
        return summary
    
    def check_compliance(self, nutrition_data: Dict, ingredients_list: List[str]) -> Dict:
        """Check compliance with dietary restrictions and regulations"""
        compliance = {
            'vegetarian': self._check_vegetarian_compliance(ingredients_list),
            'vegan': self._check_vegan_compliance(ingredients_list),
            'gluten_free': self._check_gluten_free_compliance(ingredients_list),
            'keto': self._check_keto_compliance(nutrition_data),
            'paleo': self._check_paleo_compliance(ingredients_list, nutrition_data),
            'low_sodium': self._check_low_sodium_compliance(nutrition_data),
            'low_sugar': self._check_low_sugar_compliance(nutrition_data)
        }
        
        return compliance
    
    def _check_vegetarian_compliance(self, ingredients_list: List[str]) -> Dict:
        """Check if food is vegetarian"""
        non_vegetarian_ingredients = [
            'meat', 'pork', 'beef', 'chicken', 'fish', 'seafood', 'gelatin',
            'lard', 'rennet', 'carmine', 'cochineal'
        ]
        
        ingredients_text = ' '.join(ingredients_list).lower()
        violations = []
        
        for ingredient in non_vegetarian_ingredients:
            if ingredient in ingredients_text:
                violations.append(ingredient)
        
        return {
            'compliant': len(violations) == 0,
            'violations': violations,
            'message': 'Vegetarian' if len(violations) == 0 else f'Not vegetarian - contains: {", ".join(violations)}'
        }
    
    def _check_vegan_compliance(self, ingredients_list: List[str]) -> Dict:
        """Check if food is vegan"""
        non_vegan_ingredients = [
            'meat', 'pork', 'beef', 'chicken', 'fish', 'seafood', 'gelatin',
            'lard', 'rennet', 'carmine', 'cochineal', 'milk', 'cheese',
            'butter', 'cream', 'yogurt', 'egg', 'honey', 'whey', 'casein'
        ]
        
        ingredients_text = ' '.join(ingredients_list).lower()
        violations = []
        
        for ingredient in non_vegan_ingredients:
            if ingredient in ingredients_text:
                violations.append(ingredient)
        
        return {
            'compliant': len(violations) == 0,
            'violations': violations,
            'message': 'Vegan' if len(violations) == 0 else f'Not vegan - contains: {", ".join(violations)}'
        }
    
    def _check_gluten_free_compliance(self, ingredients_list: List[str]) -> Dict:
        """Check if food is gluten-free"""
        gluten_ingredients = [
            'wheat', 'barley', 'rye', 'triticale', 'malt', 'brewer\'s yeast',
            'oat'  # Oats are often contaminated with gluten
        ]
        
        ingredients_text = ' '.join(ingredients_list).lower()
        violations = []
        
        for ingredient in gluten_ingredients:
            if ingredient in ingredients_text:
                violations.append(ingredient)
        
        return {
            'compliant': len(violations) == 0,
            'violations': violations,
            'message': 'Gluten-free' if len(violations) == 0 else f'Contains gluten: {", ".join(violations)}'
        }
    
    def _check_keto_compliance(self, nutrition_data: Dict) -> Dict:
        """Check if food is keto-friendly"""
        # Keto typically: <20g net carbs, high fat, moderate protein
        
        carbs = nutrition_data.get('carbohydrates', 0)
        fiber = nutrition_data.get('fiber', 0)
        net_carbs = carbs - fiber
        
        fat = nutrition_data.get('fat', 0)
        protein = nutrition_data.get('protein', 0)
        
        # Keto criteria
        carb_compliant = net_carbs < 20
        fat_adequate = fat > 10  # Should have significant fat
        
        compliance = carb_compliant and fat_adequate
        
        return {
            'compliant': compliance,
            'net_carbs': net_carbs,
            'fat': fat,
            'protein': protein,
            'violations': [] if compliance else [f'Too many carbs: {net_carbs}g'],
            'message': 'Keto-friendly' if compliance else f'Not keto - {net_carbs}g net carbs'
        }
    
    def _check_paleo_compliance(self, ingredients_list: List[str], nutrition_data: Dict) -> Dict:
        """Check if food is paleo-friendly"""
        # Paleo excludes: grains, legumes, dairy, processed foods, refined sugar
        
        non_paleo_ingredients = [
            'wheat', 'grain', 'rice', 'corn', 'legume', 'bean', 'peanut',
            'dairy', 'milk', 'cheese', 'butter', 'cream', 'sugar', 'high_fructose',
            'processed', 'artificial', 'preservative'
        ]
        
        ingredients_text = ' '.join(ingredients_list).lower()
        violations = []
        
        for ingredient in non_paleo_ingredients:
            if ingredient in ingredients_text:
                violations.append(ingredient)
        
        return {
            'compliant': len(violations) == 0,
            'violations': violations,
            'message': 'Paleo-friendly' if len(violations) == 0 else f'Not paleo - contains: {", ".join(violations)}'
        }
    
    def _check_low_sodium_compliance(self, nutrition_data: Dict) -> Dict:
        """Check if food is low sodium (<140mg per serving)"""
        sodium = nutrition_data.get('sodium', 0)
        
        compliant = sodium < 140
        
        return {
            'compliant': compliant,
            'sodium': sodium,
            'violations': [] if compliant else [f'High sodium: {sodium}mg'],
            'message': 'Low sodium' if compliant else f'Not low sodium - {sodium}mg'
        }
    
    def _check_low_sugar_compliance(self, nutrition_data: Dict) -> Dict:
        """Check if food is low sugar (<5g per serving)"""
        sugar = nutrition_data.get('sugar', 0)
        
        compliant = sugar < 5
        
        return {
            'compliant': compliant,
            'sugar': sugar,
            'violations': [] if compliant else [f'High sugar: {sugar}g'],
            'message': 'Low sugar' if compliant else f'Not low sugar - {sugar}g'
        }
