"""
test_astar.py
Unit tests for the A* search and nutrition modules.

Run with:
    python -m pytest tests/test_astar.py -v
"""

import os
import sys

import pytest

# Ensure the project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from astar import (
    astar_search,
    compute_meal_nutrition,
    filter_foods,
    heuristic,
    is_goal_state,
    load_food_data,
)
from nutrition import calculate_bmr, calculate_nutritional_target, calculate_tdee


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def food_df():
    """Load the food dataset."""
    csv_path = os.path.join(os.path.dirname(__file__), "..", "food_data.csv")
    return load_food_data(csv_path)


@pytest.fixture
def target_maintenance():
    """A typical nutritional target for a moderate-activity vegetarian."""
    return calculate_nutritional_target(
        weight_kg=60,
        height_cm=165,
        age=21,
        gender="Prefer not to say",
        activity_level="Moderate",
        goal="Maintain Weight",
    )


# ---------------------------------------------------------------------------
# Nutrition Tests
# ---------------------------------------------------------------------------

class TestNutrition:
    """Tests for the nutrition calculation module."""

    def test_bmr_male(self):
        bmr = calculate_bmr(70, 175, 25, "Male")
        assert bmr > 1400
        assert bmr < 2200

    def test_bmr_female(self):
        bmr = calculate_bmr(60, 165, 21, "Female")
        assert bmr > 1200
        assert bmr < 2000

    def test_tdee_increases_with_activity(self):
        bmr = 1500
        sedentary = calculate_tdee(bmr, "Sedentary")
        active = calculate_tdee(bmr, "Active")
        assert active > sedentary

    def test_weight_loss_reduces_calories(self):
        target_maintain = calculate_nutritional_target(60, 165, 21, "Female", "Moderate", "Maintain Weight")
        target_loss = calculate_nutritional_target(60, 165, 21, "Female", "Moderate", "Weight Loss")
        assert target_loss.calories < target_maintain.calories

    def test_weight_gain_increases_calories(self):
        target_maintain = calculate_nutritional_target(60, 165, 21, "Female", "Moderate", "Maintain Weight")
        target_gain = calculate_nutritional_target(60, 165, 21, "Female", "Moderate", "Weight Gain")
        assert target_gain.calories > target_maintain.calories

    def test_nutritional_target_has_all_macros(self):
        target = calculate_nutritional_target(60, 165, 21, "Male", "Light", "Maintain Weight")
        assert target.calories > 0
        assert target.protein > 0
        assert target.carbs > 0
        assert target.fat > 0


# ---------------------------------------------------------------------------
# Food Filtering Tests
# ---------------------------------------------------------------------------

class TestFoodFiltering:
    """Tests for dietary restriction filtering."""

    def test_food_data_loads(self, food_df):
        assert len(food_df) >= 25

    def test_vegetarian_filter(self, food_df):
        veg_df = filter_foods(food_df, vegetarian=True, restrictions=[])
        for _, row in veg_df.iterrows():
            assert row["vegetarian"] == "yes"

    def test_non_vegetarian_includes_meat(self, food_df):
        all_df = filter_foods(food_df, vegetarian=False, restrictions=[])
        non_veg_names = all_df["food_name"].tolist()
        assert "Chicken" in non_veg_names or "Egg" in non_veg_names

    def test_dairy_restriction(self, food_df):
        df = filter_foods(food_df, vegetarian=False, restrictions=["Dairy"])
        for _, row in df.iterrows():
            assert row["contains_dairy"] == "no"

    def test_nut_restriction(self, food_df):
        df = filter_foods(food_df, vegetarian=False, restrictions=["Nuts"])
        for _, row in df.iterrows():
            assert row["contains_nuts"] == "no"

    def test_gluten_restriction(self, food_df):
        df = filter_foods(food_df, vegetarian=False, restrictions=["Gluten"])
        for _, row in df.iterrows():
            assert row["contains_gluten"] == "no"

    def test_vegan_dairy_gluten_restriction(self, food_df):
        df = filter_foods(food_df, vegetarian=True, restrictions=["Dairy", "Gluten"])
        for _, row in df.iterrows():
            assert row["vegetarian"] == "yes"
            assert row["contains_dairy"] == "no"
            assert row["contains_gluten"] == "no"


# ---------------------------------------------------------------------------
# A* Search Tests
# ---------------------------------------------------------------------------

class TestAStarSearch:
    """Tests for the A* search implementation."""

    def test_returns_valid_meal(self, food_df, target_maintenance):
        filtered = filter_foods(food_df, vegetarian=True, restrictions=[])
        result = astar_search(filtered, target_maintenance)
        assert len(result.meal) > 0

    def test_meal_contains_valid_foods(self, food_df, target_maintenance):
        filtered = filter_foods(food_df, vegetarian=True, restrictions=[])
        result = astar_search(filtered, target_maintenance)
        valid_names = set(filtered["food_name"].tolist())
        for food_name, _ in result.meal:
            assert food_name in valid_names

    def test_nutrition_totals_match_meal(self, food_df, target_maintenance):
        filtered = filter_foods(food_df, vegetarian=True, restrictions=[])
        result = astar_search(filtered, target_maintenance)
        # Convert food names back to DataFrame indices for compute_meal_nutrition
        state_from_names = []
        for food_name, qty in result.meal:
            matches = filtered[filtered["food_name"] == food_name]
            if not matches.empty:
                idx = matches.index[0]
                state_from_names.append((idx, qty))
        cal, prot, carb, fat = compute_meal_nutrition(
            tuple(state_from_names), filtered
        ) if state_from_names else (0, 0, 0, 0)
        # Allow small rounding differences
        assert abs(cal - result.total_calories) < 1
        assert abs(prot - result.total_protein) < 1

    def test_search_result_has_costs(self, food_df, target_maintenance):
        filtered = filter_foods(food_df, vegetarian=True, restrictions=[])
        result = astar_search(filtered, target_maintenance)
        assert result.g_cost >= 0
        assert result.h_cost >= 0
        assert result.f_cost >= 0

    def test_search_respects_vegetarian(self, food_df, target_maintenance):
        filtered = filter_foods(food_df, vegetarian=True, restrictions=[])
        result = astar_search(filtered, target_maintenance)
        for food_name, _ in result.meal:
            food_row = food_df[food_df["food_name"] == food_name].iloc[0]
            assert food_row["vegetarian"] == "yes"

    def test_search_respects_dairy_restriction(self, food_df, target_maintenance):
        filtered = filter_foods(food_df, vegetarian=False, restrictions=["Dairy"])
        result = astar_search(filtered, target_maintenance)
        for food_name, _ in result.meal:
            food_row = food_df[food_df["food_name"] == food_name].iloc[0]
            assert food_row["contains_dairy"] == "no"

    def test_search_respects_nut_restriction(self, food_df, target_maintenance):
        filtered = filter_foods(food_df, vegetarian=False, restrictions=["Nuts"])
        result = astar_search(filtered, target_maintenance)
        for food_name, _ in result.meal:
            food_row = food_df[food_df["food_name"] == food_name].iloc[0]
            assert food_row["contains_nuts"] == "no"

    def test_nodes_explored_positive(self, food_df, target_maintenance):
        filtered = filter_foods(food_df, vegetarian=True, restrictions=[])
        result = astar_search(filtered, target_maintenance)
        assert result.nodes_explored > 0

    def test_meal_quantity_reasonable(self, food_df, target_maintenance):
        filtered = filter_foods(food_df, vegetarian=True, restrictions=[])
        result = astar_search(filtered, target_maintenance)
        for _, qty in result.meal:
            assert 50 <= qty <= 300


# ---------------------------------------------------------------------------
# Heuristic & State Tests
# ---------------------------------------------------------------------------

class TestHeuristic:
    """Tests for heuristic and goal state functions."""

    def test_heuristic_empty_meal(self, food_df, target_maintenance):
        h = heuristic((), food_df, target_maintenance)
        assert h > 0

    def test_goal_state_empty_meal_is_false(self, food_df, target_maintenance):
        assert not is_goal_state((), food_df, target_maintenance)

    def test_meal_nutrition_empty(self, food_df):
        cal, prot, carb, fat = compute_meal_nutrition((), food_df)
        assert cal == 0
        assert prot == 0
        assert carb == 0
        assert fat == 0
