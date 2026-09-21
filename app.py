"""
app.py
Streamlit application for NutriAI – Personalized Nutrition Planner.

Run with:
    streamlit run app.py

This is an educational prototype inspired by the personalized nutrition
architecture presented in the research paper "An AI Based Approach for
Personalized Nutrition and Food Menu Planning" (IEEE ICECOCS 2022).
"""

import os
import sys

import pandas as pd
import streamlit as st

# Ensure local modules are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from astar import astar_search, filter_foods, load_food_data
from nutrition import calculate_nutritional_target


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NutriAI – Personalized Nutrition Planner",
    page_icon="🍽️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #E0F7FA;
    }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .stApp p, .stApp span, .stApp label, .stApp div[data-testid="stMarkdownContainer"] {
        color: #1A237E !important;
    }
    .stApp .stMarkdown blockquote {
        color: #0D47A1 !important;
        border-left-color: #0288D1 !important;
    }
    .stApp .stMetric label, .stApp .stMetric div[data-testid="stMetricValue"] {
        color: #1A237E !important;
    }
    .stApp .stCaption {
        color: #37474F !important;
    }
    [data-testid="stSidebar"] {
        background-color: #1A237E;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] {
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        color: #FFFFFF !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("NutriAI – Personalized Nutrition Planner")
st.subheader("An AI-based prototype for personalized meal planning")

st.markdown(
    """
> **Disclaimer:** This is an educational prototype and **not** medical or
> clinical nutritional advice. The food values used are approximate
> demonstration data, not a clinical nutrition database.
"""
)

# ---------------------------------------------------------------------------
# Sidebar – Patient Information
# ---------------------------------------------------------------------------
st.sidebar.header("Patient Information")

age = st.sidebar.number_input("Age (years)", min_value=1, max_value=120, value=21)
gender = st.sidebar.selectbox("Gender", ["Prefer not to say", "Male", "Female"])
height = st.sidebar.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=165.0, step=1.0)
weight = st.sidebar.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=60.0, step=1.0)

st.sidebar.header("Lifestyle")
activity = st.sidebar.selectbox(
    "Physical Activity",
    ["Sedentary", "Light", "Moderate", "Active"],
)

st.sidebar.header("Dietary Preference")
diet = st.sidebar.radio("Diet", ["Vegetarian", "Non-Vegetarian"])

st.sidebar.header("Restrictions / Allergies")
restriction_choice = st.sidebar.selectbox(
    "Restrictions",
    ["None", "Dairy", "Nuts", "Gluten", "Custom"],
)
custom_restriction = ""
if restriction_choice == "Custom":
    custom_restriction = st.sidebar.text_input("Enter custom restriction")

st.sidebar.header("Goal")
goal = st.sidebar.selectbox(
    "Goal",
    ["Maintain Weight", "Weight Loss", "Weight Gain"],
)

# ---------------------------------------------------------------------------
# Demo Patient Button
# ---------------------------------------------------------------------------
st.sidebar.markdown("---")
if st.sidebar.button("Load Demo Patient"):
    st.session_state["demo_loaded"] = True

# Override values if demo is loaded
if st.session_state.get("demo_loaded"):
    age = 21
    gender = "Prefer not to say"
    height = 165.0
    weight = 60.0
    activity = "Moderate"
    diet = "Vegetarian"
    restriction_choice = "None"
    custom_restriction = ""
    goal = "Maintain Weight"
    st.session_state["demo_loaded"] = False

# ---------------------------------------------------------------------------
# Main – Generate Meal Plan
# ---------------------------------------------------------------------------
st.markdown("---")

if st.button("Generate Personalized Meal Plan"):
    # Validate inputs
    errors = []
    if age < 1 or age > 120:
        errors.append("Please enter a valid age (1-120).")
    if height < 50 or height > 250:
        errors.append("Please enter a valid height (50-250 cm).")
    if weight < 20 or weight > 300:
        errors.append("Please enter a valid weight (20-300 kg).")
    if restriction_choice == "Custom" and not custom_restriction.strip():
        errors.append("Please enter a custom restriction or select 'None'.")

    if errors:
        for e in errors:
            st.error(e)
    else:
        # ------- Nutritional Target Calculation -------
        st.header("Personalized Nutritional Target")

        target = calculate_nutritional_target(
            weight_kg=weight,
            height_cm=height,
            age=age,
            gender=gender,
            activity_level=activity,
            goal=goal,
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Calories", f"{target.calories:.0f} kcal")
        col2.metric("Protein", f"{target.protein:.0f} g")
        col3.metric("Carbs", f"{target.carbs:.0f} g")
        col4.metric("Fat", f"{target.fat:.0f} g")

        st.caption(
            "Estimated using simplified BMR/TDEE calculation. "
            "This is for educational demonstration only."
        )

        # ------- Load & Filter Food Data -------
        csv_path = os.path.join(os.path.dirname(__file__), "food_data.csv")
        food_df = load_food_data(csv_path)

        restrictions = []
        if restriction_choice != "None":
            restrictions.append(restriction_choice)

        vegetarian = diet == "Vegetarian"
        filtered_df = filter_foods(food_df, vegetarian, restrictions)

        if filtered_df.empty:
            st.warning(
                "No foods match the selected dietary restrictions. "
                "Please try different options."
            )
        else:
            # ------- A* Search -------
            st.header("A* Recommended Meal Plan")

            with st.spinner("Running A* search to find the best meal..."):
                result = astar_search(filtered_df, target)

            # Display recommended meal table
            if result.meal:
                meal_data = []
                for food_name, qty in result.meal:
                    food_row = filtered_df[filtered_df["food_name"] == food_name].iloc[0]
                    factor = qty / 100.0
                    meal_data.append({
                        "Food": food_name,
                        "Quantity (g)": qty,
                        "Calories": round(food_row["calories_per_100g"] * factor, 1),
                        "Protein (g)": round(food_row["protein_per_100g"] * factor, 1),
                        "Carbs (g)": round(food_row["carbs_per_100g"] * factor, 1),
                        "Fat (g)": round(food_row["fat_per_100g"] * factor, 1),
                    })
                meal_df = pd.DataFrame(meal_data)
                st.dataframe(meal_df, use_container_width=True, hide_index=True)

                # Totals
                st.subheader("Meal Totals")
                tc1, tc2, tc3, tc4 = st.columns(4)
                tc1.metric("Total Calories", f"{result.total_calories:.0f} kcal")
                tc2.metric("Total Protein", f"{result.total_protein:.0f} g")
                tc3.metric("Total Carbs", f"{result.total_carbs:.0f} g")
                tc4.metric("Total Fat", f"{result.total_fat:.0f} g")

                # ------- Target vs Recommended Comparison -------
                st.header("Target vs Recommended")
                comparison = pd.DataFrame({
                    "Nutrient": ["Calories", "Protein (g)", "Carbs (g)", "Fat (g)"],
                    "Target": [
                        target.calories,
                        target.protein,
                        target.carbs,
                        target.fat,
                    ],
                    "Recommended": [
                        result.total_calories,
                        result.total_protein,
                        result.total_carbs,
                        result.total_fat,
                    ],
                    "Difference": [
                        round(result.total_calories - target.calories, 1),
                        round(result.total_protein - target.protein, 1),
                        round(result.total_carbs - target.carbs, 1),
                        round(result.total_fat - target.fat, 1),
                    ],
                })
                st.dataframe(comparison, use_container_width=True, hide_index=True)

                # ------- A* Search Info -------
                with st.expander("How did A* select this meal?"):
                    st.markdown(
                        f"""
**A* Search Explanation:**

1. **State:** Each possible meal combination is treated as a search state.
   A state is a tuple of `(food_index, quantity)` pairs.

2. **Initial State:** The search starts with an empty meal `()`.

3. **Actions:** From any state, the search can:
   - *Add* a new food with a specific quantity
   - *Increase* the quantity of an existing food
   - *Decrease* the quantity of an existing food
   - *Remove* a food from the meal

4. **Transition:** Applying an action produces a new meal state.

5. **g(n) — Actual Cost:** Measures squared normalized deviation of each
   nutrient from the target. Lower values mean the meal is closer to
   the nutritional goals.

6. **h(n) — Heuristic:** Estimates the remaining nutritional gap using
   normalized absolute differences between current and target nutrients.
   This is *admissible* (never overestimates the true remaining cost).

7. **f(n) = g(n) + h(n):** The total estimated cost. A* always expands
   the state with the lowest f(n) first.

8. **Goal State:** The meal's nutrients fall within ±15% of all targets,
   or the search limit is reached.

9. **Priority Queue:** Python's `heapq` is used to efficiently retrieve
   the state with the lowest f(n).

**Search Statistics:**
- Nodes explored: {result.nodes_explored}
- Search limit reached: {result.search_limit_reached}
- Final g(n): {result.g_cost}
- Final h(n): {result.h_cost}
- Final f(n): {result.f_cost}
"""
                    )

                # ------- Meal Consumption Simulation -------
                st.header("Meal Consumption Analysis")
                st.caption(
                    "Simplified simulation – this prototype does NOT perform "
                    "actual food-image segmentation or computer vision."
                )

                consumption_pct = st.slider(
                    "Approximate consumption percentage",
                    min_value=25,
                    max_value=100,
                    value=100,
                    step=25,
                )
                factor = consumption_pct / 100.0

                consumed_cal = round(result.total_calories * factor, 1)
                consumed_protein = round(result.total_protein * factor, 1)
                consumed_carbs = round(result.total_carbs * factor, 1)
                consumed_fat = round(result.total_fat * factor, 1)

                remaining_cal = round(result.total_calories - consumed_cal, 1)
                remaining_protein = round(result.total_protein - consumed_protein, 1)
                remaining_carbs = round(result.total_carbs - consumed_carbs, 1)
                remaining_fat = round(result.total_fat - consumed_fat, 1)

                st.subheader(f"Consumption: {consumption_pct}%")
                col_c1, col_c2, col_c3, col_c4 = st.columns(4)
                col_c1.metric("Consumed Calories", f"{consumed_cal} kcal")
                col_c2.metric("Consumed Protein", f"{consumed_protein} g")
                col_c3.metric("Consumed Carbs", f"{consumed_carbs} g")
                col_c4.metric("Consumed Fat", f"{consumed_fat} g")

                st.subheader("Estimated Nutritional Deficit")
                deficit_df = pd.DataFrame({
                    "Nutrient": ["Calories (kcal)", "Protein (g)", "Carbs (g)", "Fat (g)"],
                    "Remaining": [remaining_cal, remaining_protein, remaining_carbs, remaining_fat],
                    "Target": [target.calories, target.protein, target.carbs, target.fat],
                })
                st.dataframe(deficit_df, use_container_width=True, hide_index=True)

                # Progress bars for deficit
                st.subheader("Deficit Progress")
                for _, row in deficit_df.iterrows():
                    pct = (row["Remaining"] / row["Target"]) * 100 if row["Target"] > 0 else 0
                    pct = max(0, min(100, pct))
                    st.progress(pct / 100, text=f"{row['Nutrient']}: {pct:.1f}% remaining")

            else:
                st.warning("A* search could not find a suitable meal. Please try different settings.")

# ---------------------------------------------------------------------------
# Research Paper Connection
# ---------------------------------------------------------------------------
st.markdown("---")
with st.expander("Connection to Selected Research Paper"):
    st.markdown(
        """
This prototype is inspired by the personalized nutrition architecture
presented in:

> **"An AI Based Approach for Personalized Nutrition and Food Menu Planning"**
> Authors: Khalid Azzimani, Hayat Bihri, Asma Dahmi, Salma Azzouzi,
> My El Hassan Charaf
> IEEE ICECOCS 2022

**Workflow from the paper:**

1. Patient information and nutritional history collection
2. Nutritional requirement determination
3. Food restrictions and allergies identification
4. Food composition data lookup
5. Personalized meal planning
6. Nutritional comparison and deficit estimation
7. Redistribution of missing nutrients in subsequent meals

The paper also discusses before/after food images and segmentation
to estimate food consumption.

**What our prototype implements:**
- Patient information collection ✓
- Nutritional requirement estimation (simplified) ✓
- Food restriction filtering ✓
- Food composition data (local dataset) ✓
- Personalized meal planning using A* search ✓ (our proposed method)
- Nutritional comparison ✓
- Simplified consumption simulation ✓

**What is NOT implemented (future scope):**
- Computer vision for food image segmentation
- RGB-D camera food analysis
- Real patient clinical datasets
- Multi-meal daily planning with deficit redistribution
"""
    )

# ---------------------------------------------------------------------------
# Limitations
# ---------------------------------------------------------------------------
with st.expander("Limitations"):
    st.markdown(
        """
- The food dataset contains approximate nutritional values for
  demonstration purposes only.
- BMR/TDEE formulas are simplified estimates, not clinical measurements.
- A* search operates on a limited discrete state space.
- The consumption analysis is a mathematical simulation only.
- No real computer vision or food image processing is implemented.
- This is NOT a medical device and should not be used for clinical
  decision-making.
"""
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "NutriAI – Academic Prototype | CIA-1 Project | "
    "Inspired by IEEE ICECOCS 2022 Paper"
)
