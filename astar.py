"""
astar.py
A* Search implementation for meal planning.

This module implements a genuine A* search algorithm to find a meal
combination that best matches a user's nutritional requirements.

IMPORTANT NOTE:
The selected research paper ("An AI Based Approach for Personalized
Nutrition and Food Menu Planning", IEEE ICECOCS 2022) does NOT
explicitly specify A* search. A* is OUR PROPOSED SEARCH METHOD for
the meal-selection component of this prototype.

A* SEARCH COMPONENTS IN OUR PROBLEM:

State:
    A meal represented as a tuple of (food_index, quantity_grams) pairs.
    Example: ((0, 150), (3, 100), (7, 80)) means food_0 at 150g,
    food_3 at 100g, food_7 at 80g.

Initial State:
    Empty meal: ()  -- no foods selected.

Actions:
    1. Add a new food with a chosen quantity.
    2. Increase the quantity of an existing food.
    3. Decrease the quantity of an existing food.
    4. Remove a food from the meal.

Transition:
    Applying an action to a state produces a new state (new meal).

g(n):
    The cost accumulated so far -- sum of squared nutrient deviations
    from the target. This represents how far the current meal is from
    meeting all nutritional goals.

h(n):
    Heuristic -- estimated remaining cost to reach the target. Uses
    normalized differences between current and target nutrients:
    calories, protein, carbs, and fat.

f(n) = g(n) + h(n):
    Total estimated cost. A* expands the state with lowest f(n).

Goal State:
    The meal's nutritional values are within an acceptable tolerance
    of the target, or the search limit is reached.

Priority Queue:
    Python's heapq is used to efficiently retrieve the state with
    the lowest f(n).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from nutrition import NutritionalTarget


# --- Constants ---

MAX_FOODS_IN_MEAL = 5
MAX_QUANTITY_G = 300
MIN_QUANTITY_G = 50
QUANTITY_STEP = 25
MAX_SEARCH_NODES = 5000

# Normalization factors for heuristic (per 100g average scale)
NORM_CALORIES = 2000.0
NORM_PROTEIN = 60.0
NORM_CARB = 250.0
NORM_FAT = 70.0

# Tolerance for goal check (percentage of target)
GOAL_TOLERANCE = 0.15  # 15% tolerance


@dataclass(order=True)
class SearchNode:
    """A node in the A* search graph."""
    f_cost: float
    state: tuple = field(compare=False)
    g_cost: float = field(compare=False)
    h_cost: float = field(compare=False)
    actions_taken: list = field(compare=False)


@dataclass
class AStarResult:
    """Result returned by the A* search."""
    meal: list[tuple[str, float]]
    total_calories: float
    total_protein: float
    total_carbs: float
    total_fat: float
    g_cost: float
    h_cost: float
    f_cost: float
    nodes_explored: int
    search_limit_reached: bool


def load_food_data(csv_path: str = "food_data.csv") -> pd.DataFrame:
    """Load the food dataset from CSV."""
    df = pd.read_csv(csv_path)
    return df


def filter_foods(
    df: pd.DataFrame,
    vegetarian: bool,
    restrictions: list[str],
) -> pd.DataFrame:
    """
    Filter foods based on dietary preference and restrictions.

    Parameters:
        df: Food dataframe
        vegetarian: True if user is vegetarian
        restrictions: List of restrictions like ["Dairy", "Nuts", "Gluten"]

    Returns:
        Filtered dataframe of allowed foods
    """
    filtered = df.copy()

    if vegetarian:
        filtered = filtered[filtered["vegetarian"] == "yes"]

    if "Dairy" in restrictions:
        filtered = filtered[filtered["contains_dairy"] == "no"]

    if "Nuts" in restrictions:
        filtered = filtered[filtered["contains_nuts"] == "no"]

    if "Gluten" in restrictions:
        filtered = filtered[filtered["contains_gluten"] == "no"]

    return filtered


def compute_meal_nutrition(
    state: tuple,
    food_df: pd.DataFrame,
) -> tuple[float, float, float, float]:
    """
    Compute total nutritional values for a meal state.

    Parameters:
        state: Tuple of (food_index, quantity_grams) pairs.
               food_index refers to the row index in food_df.
        food_df: The filtered food dataframe.

    Returns:
        (calories, protein, carbs, fat) totals
    """
    total_cal = 0.0
    total_protein = 0.0
    total_carbs = 0.0
    total_fat = 0.0

    for food_idx, qty in state:
        if food_idx in food_df.index:
            row = food_df.loc[food_idx]
            factor = qty / 100.0
            total_cal += row["calories_per_100g"] * factor
            total_protein += row["protein_per_100g"] * factor
            total_carbs += row["carbs_per_100g"] * factor
            total_fat += row["fat_per_100g"] * factor

    return total_cal, total_protein, total_carbs, total_fat


def heuristic(
    state: tuple,
    food_df: pd.DataFrame,
    target: NutritionalTarget,
) -> float:
    """
    Calculate the heuristic h(n) -- estimated remaining nutritional deviation.

    Uses normalized sum of absolute differences between current meal
    nutrients and target nutrients. Normalization ensures all nutrients
    contribute proportionally to the heuristic.

    h(n) = |cal_cur - cal_tgt|/NORM_CAL
         + |prot_cur - prot_tgt|/NORM_PROT
         + |carb_cur - carb_tgt|/NORM_CARB
         + |fat_cur - fat_tgt|/NORM_FAT

    This is admissible (never overestimates) because it measures the
    raw gap without assuming any single food can perfectly fill it.
    """
    cal, prot, carb, fat = compute_meal_nutrition(state, food_df)

    h = (
        abs(cal - target.calories) / NORM_CALORIES
        + abs(prot - target.protein) / NORM_PROTEIN
        + abs(carb - target.carbs) / NORM_CARB
        + abs(fat - target.fat) / NORM_FAT
    )
    return h


def cost_fn(
    state: tuple,
    food_df: pd.DataFrame,
    target: NutritionalTarget,
) -> float:
    """
    Calculate g(n) -- the actual cost of the current state.

    Measures squared normalized deviation of each nutrient from target.
    Lower values mean the meal is closer to the target.
    """
    cal, prot, carb, fat = compute_meal_nutrition(state, food_df)

    g = (
        ((cal - target.calories) / NORM_CALORIES) ** 2
        + ((prot - target.protein) / NORM_PROTEIN) ** 2
        + ((carb - target.carbs) / NORM_CARB) ** 2
        + ((fat - target.fat) / NORM_FAT) ** 2
    )
    return g


def is_goal_state(
    state: tuple,
    food_df: pd.DataFrame,
    target: NutritionalTarget,
) -> bool:
    """
    Check if the current meal state meets the nutritional target
    within an acceptable tolerance.
    """
    cal, prot, carb, fat = compute_meal_nutrition(state, food_df)

    within_cal = abs(cal - target.calories) / target.calories <= GOAL_TOLERANCE
    within_prot = abs(prot - target.protein) / target.protein <= GOAL_TOLERANCE
    within_carb = abs(carb - target.carbs) / target.carbs <= GOAL_TOLERANCE
    within_fat = abs(fat - target.fat) / target.fat <= GOAL_TOLERANCE

    return within_cal and within_prot and within_carb and within_fat


def get_actions(
    state: tuple,
    food_df: pd.DataFrame,
) -> list[tuple[str, Any]]:
    """
    Generate all valid actions from the current state.

    Actions:
        ("add", food_idx, quantity)     -- Add a new food
        ("increase", food_idx, qty)     -- Increase existing food quantity
        ("decrease", food_idx, qty)     -- Decrease existing food quantity
        ("remove", food_idx)            -- Remove a food from meal

    Returns list of (action_type, data) tuples.
    """
    actions = []
    current_foods = {idx for idx, _ in state}
    food_indices = list(food_df.index)

    # Action 1: Add a new food (if meal not full)
    if len(state) < MAX_FOODS_IN_MEAL:
        for fi in food_indices:
            if fi not in current_foods:
                for qty in range(MIN_QUANTITY_G, MAX_QUANTITY_G + 1, QUANTITY_STEP):
                    actions.append(("add", fi, qty))

    # Actions 2 & 3: Increase or decrease quantity of existing foods
    for i, (fi, qty) in enumerate(state):
        if qty < MAX_QUANTITY_G:
            new_qty = min(qty + QUANTITY_STEP, MAX_QUANTITY_G)
            actions.append(("increase", i, new_qty))
        if qty > MIN_QUANTITY_G:
            new_qty = max(qty - QUANTITY_STEP, MIN_QUANTITY_G)
            actions.append(("decrease", i, new_qty))

    # Action 4: Remove a food from meal
    for i, (fi, _) in enumerate(state):
        actions.append(("remove", i))

    return actions


def apply_action(state: tuple, action: tuple) -> tuple:
    """
    Apply an action to a state and return the new state.

    Parameters:
        state: Current meal state
        action: (action_type, ...) tuple

    Returns:
        New state after applying the action
    """
    state_list = list(state)
    action_type = action[0]

    if action_type == "add":
        _, food_idx, qty = action
        state_list.append((food_idx, qty))
    elif action_type == "increase":
        _, index, new_qty = action
        fi, _ = state_list[index]
        state_list[index] = (fi, new_qty)
    elif action_type == "decrease":
        _, index, new_qty = action
        fi, _ = state_list[index]
        state_list[index] = (fi, new_qty)
    elif action_type == "remove":
        _, index = action
        state_list.pop(index)

    return tuple(state_list)


def astar_search(
    food_df: pd.DataFrame,
    target: NutritionalTarget,
) -> AStarResult:
    """
    Execute A* search to find the best meal plan.

    The search explores meal combinations starting from an empty meal,
    applying actions (add, increase, decrease, remove foods) to find
    a combination that best matches the nutritional target.

    Uses a priority queue (heapq) ordered by f(n) = g(n) + h(n).

    Parameters:
        food_df: Filtered food dataframe
        target: NutritionalTarget with calorie and macro goals

    Returns:
        AStarResult with the best meal found
    """
    # Initial state: empty meal
    initial_state: tuple = ()

    # Calculate initial costs
    g0 = cost_fn(initial_state, food_df, target)
    h0 = heuristic(initial_state, food_df, target)
    f0 = g0 + h0

    # Create initial node
    initial_node = SearchNode(
        f_cost=f0,
        state=initial_state,
        g_cost=g0,
        h_cost=h0,
        actions_taken=[],
    )

    # Priority queue (min-heap)
    open_list: list[SearchNode] = []
    heapq.heappush(open_list, initial_node)

    # Track visited states to avoid cycles
    visited: set[tuple] = set()

    # Best solution found so far
    best_node: SearchNode = initial_node
    nodes_explored = 0

    while open_list and nodes_explored < MAX_SEARCH_NODES:
        # Pop the node with lowest f(n)
        current = heapq.heappop(open_list)

        # Skip if already visited
        state_key = current.state
        if state_key in visited:
            continue
        visited.add(state_key)
        nodes_explored += 1

        # Update best node if current is better
        if current.h_cost < best_node.h_cost:
            best_node = current

        # Check if goal state reached
        if is_goal_state(current.state, food_df, target):
            best_node = current
            break

        # Expand: generate successor states
        actions = get_actions(current.state, food_df)

        for action in actions:
            new_state = apply_action(current.state, action)

            # Skip if already visited
            if new_state in visited:
                continue

            # Calculate costs for new state
            new_g = cost_fn(new_state, food_df, target)
            new_h = heuristic(new_state, food_df, target)
            new_f = new_g + new_h

            # Create new node with action history
            new_actions = current.actions_taken + [action]
            new_node = SearchNode(
                f_cost=new_f,
                state=new_state,
                g_cost=new_g,
                h_cost=new_h,
                actions_taken=new_actions,
            )
            heapq.heappush(open_list, new_node)

    # Build result
    search_limit_reached = nodes_explored >= MAX_SEARCH_NODES
    meal_foods = []
    for food_idx, qty in best_node.state:
        if food_idx in food_df.index:
            row = food_df.loc[food_idx]
            meal_foods.append((row["food_name"], qty))

    cal, prot, carb, fat = compute_meal_nutrition(best_node.state, food_df)

    return AStarResult(
        meal=meal_foods,
        total_calories=round(cal, 1),
        total_protein=round(prot, 1),
        total_carbs=round(carb, 1),
        total_fat=round(fat, 1),
        g_cost=round(best_node.g_cost, 4),
        h_cost=round(best_node.h_cost, 4),
        f_cost=round(best_node.f_cost, 4),
        nodes_explored=nodes_explored,
        search_limit_reached=search_limit_reached,
    )
