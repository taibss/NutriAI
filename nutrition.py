"""
nutrition.py
Nutritional requirement calculation module.

This module calculates estimated daily nutritional requirements using
a simplified BMR (Basal Metabolic Rate) and TDEE (Total Daily Energy
Expenditure) approach.

Formulas used:
- Mifflin-St Jeor BMR for males:   10 * weight(kg) + 6.25 * height(cm) - 5 * age - 5
- Mifflin-St Jeor BMR for females: 10 * weight(kg) + 6.25 * height(cm) - 5 * age - 161

TDEE = BMR * Activity Multiplier

Activity Multipliers:
- Sedentary: 1.2
- Light:     1.375
- Moderate:  1.55
- Active:    1.725

Goal Adjustments (applied to TDEE):
- Maintain:     no change
- Weight Loss:  -500 kcal/day
- Weight Gain:  +300 kcal/day

Macronutrient Distribution:
- Protein:   ~25% of calories (4 kcal per gram)
- Carbs:     ~50% of calories (4 kcal per gram)
- Fat:       ~25% of calories (9 kcal per gram)

IMPORTANT: This is an educational prototype and NOT medical advice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


# Literal type for gender
Gender = Literal["Male", "Female", "Prefer not to say"]
ActivityLevel = Literal["Sedentary", "Light", "Moderate", "Active"]
Goal = Literal["Maintain Weight", "Weight Loss", "Weight Gain"]


@dataclass
class NutritionalTarget:
    """Represents the estimated daily nutritional targets for a user."""
    calories: float
    protein: float
    carbs: float
    fat: float


# Activity multipliers used in TDEE calculation
ACTIVITY_MULTIPLIERS: dict[ActivityLevel, float] = {
    "Sedentary": 1.2,
    "Light": 1.375,
    "Moderate": 1.55,
    "Active": 1.725,
}

# Goal-based calorie adjustments (kcal per day)
GOAL_ADJUSTMENTS: dict[Goal, float] = {
    "Maintain Weight": 0,
    "Weight Loss": -500,
    "Weight Gain": 300,
}


def calculate_bmr(weight_kg: float, height_cm: float, age: int, gender: Gender) -> float:
    """
    Calculate Basal Metabolic Rate using the Mifflin-St Jeor equation.

    For males:   BMR = 10*weight + 6.25*height - 5*age + 5
    For females: BMR = 10*weight + 6.25*height - 5*age - 161
    For unknown: average of male and female values

    Parameters:
        weight_kg: Body weight in kilograms
        height_cm: Height in centimeters
        age: Age in years
        gender: "Male", "Female", or "Prefer not to say"

    Returns:
        Estimated BMR in kcal/day
    """
    bmr_male = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    bmr_female = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161

    if gender == "Male":
        return bmr_male
    elif gender == "Female":
        return bmr_female
    else:
        # Average of male and female when gender is unknown
        return (bmr_male + bmr_female) / 2


def calculate_tdee(bmr: float, activity_level: ActivityLevel) -> float:
    """
    Calculate Total Daily Energy Expenditure.

    TDEE = BMR * activity_multiplier

    Parameters:
        bmr: Basal Metabolic Rate
        activity_level: Physical activity level

    Returns:
        Estimated TDEE in kcal/day
    """
    multiplier = ACTIVITY_MULTIPLIERS.get(activity_level, 1.2)
    return bmr * multiplier


def calculate_nutritional_target(
    weight_kg: float,
    height_cm: float,
    age: int,
    gender: Gender,
    activity_level: ActivityLevel,
    goal: Goal,
) -> NutritionalTarget:
    """
    Calculate the full nutritional target for a user.

    Steps:
        1. Compute BMR using Mifflin-St Jeor equation
        2. Multiply by activity factor to get TDEE
        3. Apply goal-based calorie adjustment
        4. Distribute calories into protein, carbs, and fat

    Parameters:
        weight_kg: Body weight in kilograms
        height_cm: Height in centimeters
        age: Age in years
        gender: "Male", "Female", or "Prefer not to say"
        activity_level: "Sedentary", "Light", "Moderate", or "Active"
        goal: "Maintain Weight", "Weight Loss", or "Weight Gain"

    Returns:
        NutritionalTarget with estimated daily calories, protein, carbs, and fat
    """
    bmr = calculate_bmr(weight_kg, height_cm, age, gender)
    tdee = calculate_tdee(bmr, activity_level)

    # Apply goal adjustment
    adjustment = GOAL_ADJUSTMENTS.get(goal, 0)
    target_calories = tdee + adjustment

    # Ensure minimum calorie intake for safety
    if target_calories < 1000:
        target_calories = 1000

    # Macronutrient distribution
    # Protein: 25% of calories, 4 kcal per gram
    protein_grams = (target_calories * 0.25) / 4
    # Carbs: 50% of calories, 4 kcal per gram
    carbs_grams = (target_calories * 0.50) / 4
    # Fat: 25% of calories, 9 kcal per gram
    fat_grams = (target_calories * 0.25) / 9

    return NutritionalTarget(
        calories=round(target_calories, 1),
        protein=round(protein_grams, 1),
        carbs=round(carbs_grams, 1),
        fat=round(fat_grams, 1),
    )
