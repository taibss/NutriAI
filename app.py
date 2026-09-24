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
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Ensure local modules are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from astar import astar_search, filter_foods, load_food_data
from nutrition import calculate_nutritional_target


# ---------------------------------------------------------------------------
# Colour constants (kept in sync with styles.css)
# ---------------------------------------------------------------------------
CAL_COLOR = "#ff6b6b"
PRO_COLOR = "#4ecdc4"
CARB_COLOR = "#ffd93d"
FAT_COLOR = "#a78bfa"
ACCENT = "#00d4aa"
TEXT_SEC = "#8b949e"
BG_CARD = "rgba(22, 27, 34, 0.85)"

# Plotly layout defaults for dark theme
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#e6edf3", size=13),
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(48,54,61,0.4)",
        borderwidth=1,
        font=dict(size=12),
    ),
)


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NutriAI – Personalized Nutrition Planner",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Load external CSS
# ---------------------------------------------------------------------------
def load_css() -> None:
    """Inject styles.css into the Streamlit page."""
    css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "styles.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css()


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def metric_card(label: str, value: str, unit: str, color: str) -> None:
    """Render a styled metric card with a coloured left border."""
    st.markdown(
        f"""
        <div class="metric-card" style="border-left-color: {color};">
            <p class="metric-label">{label}</p>
            <p class="metric-value" style="color: {color};">{value}</p>
            <p class="metric-unit">{unit}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_card(label: str, value: str) -> None:
    """Render a small info card for profile display."""
    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-label">{label}</div>
            <div class="info-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(text: str) -> None:
    """Render a styled section divider."""
    st.markdown(
        f"""<div class="section-header"><h3>{text}</h3></div>""",
        unsafe_allow_html=True,
    )


def badge(text: str, kind: str = "info") -> str:
    """Return HTML for a coloured badge. kind = success | warning | info."""
    return f'<span class="badge badge-{kind}">{text}</span>'


# ---------------------------------------------------------------------------
# Plotly chart builders
# ---------------------------------------------------------------------------
def build_target_vs_recommended_chart(
    target, result,
) -> go.Figure:
    """Grouped bar chart: Target vs Recommended for all four nutrients."""
    nutrients = ["Calories (kcal)", "Protein (g)", "Carbs (g)", "Fat (g)"]
    target_vals = [target.calories, target.protein, target.carbs, target.fat]
    rec_vals = [
        result.total_calories,
        result.total_protein,
        result.total_carbs,
        result.total_fat,
    ]
    colors_target = [CAL_COLOR, PRO_COLOR, CARB_COLOR, FAT_COLOR]
    colors_rec = [
        "rgba(255,107,107,0.45)",
        "rgba(78,205,196,0.45)",
        "rgba(255,217,61,0.45)",
        "rgba(167,139,250,0.45)",
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Target",
            x=nutrients,
            y=target_vals,
            marker_color=colors_target,
            marker_line=dict(width=0),
            opacity=0.9,
        )
    )
    fig.add_trace(
        go.Bar(
            name="Recommended",
            x=nutrients,
            y=rec_vals,
            marker_color=colors_rec,
            marker_line=dict(color=colors_target, width=2),
        )
    )
    fig.update_layout(
        barmode="group",
        title="Target vs Recommended",
        yaxis_title="Value",
        **PLOTLY_LAYOUT,
    )
    fig.update_xaxes(gridcolor="rgba(48,54,61,0.3)")
    fig.update_yaxes(gridcolor="rgba(48,54,61,0.3)")
    return fig


def build_macro_donut(result) -> go.Figure:
    """Donut chart showing protein / carbs / fat split in grams."""
    labels = ["Protein", "Carbs", "Fat"]
    values = [result.total_protein, result.total_carbs, result.total_fat]
    colors = [PRO_COLOR, CARB_COLOR, FAT_COLOR]

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.55,
            marker=dict(colors=colors, line=dict(color="#0e1117", width=2)),
            textinfo="label+percent",
            textfont=dict(size=13, color="#e6edf3"),
            hovertemplate="%{label}: %{value:.1f} g<br>%{percent}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Macro Distribution",
        annotations=[
            dict(
                text=f"{result.total_calories:.0f}<br>kcal",
                x=0.5,
                y=0.5,
                font=dict(size=18, color="#e6edf3", family="Inter"),
                showarrow=False,
            )
        ],
        **PLOTLY_LAYOUT,
    )
    return fig


def build_convergence_chart(f_cost_history: list[float]) -> go.Figure:
    """Line chart showing the best f-cost across search iterations."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(f_cost_history) + 1)),
            y=f_cost_history,
            mode="lines+markers",
            line=dict(color=ACCENT, width=2),
            marker=dict(size=6, color=ACCENT),
            name="Best f(n)",
            hovertemplate="Node %{x}<br>f(n) = %{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Search Convergence (best f-cost)",
        xaxis_title="Nodes Explored",
        yaxis_title="f(n) = g(n) + h(n)",
        height=300,
        **PLOTLY_LAYOUT,
    )
    fig.update_xaxes(gridcolor="rgba(48,54,61,0.3)", dtick=1)
    fig.update_yaxes(gridcolor="rgba(48,54,61,0.3)")
    return fig


# ---------------------------------------------------------------------------
# Sidebar – Patient inputs
# ---------------------------------------------------------------------------
def load_demo() -> None:
    """Callback to populate demo patient values in session state."""
    st.session_state["input_age"] = 21
    st.session_state["input_gender"] = "Prefer not to say"
    st.session_state["input_height"] = 165.0
    st.session_state["input_weight"] = 60.0
    st.session_state["input_activity"] = "Moderate"
    st.session_state["input_diet"] = "Vegetarian"
    st.session_state["input_restriction"] = "None"
    st.session_state["input_goal"] = "Maintain Weight"


with st.sidebar:
    st.markdown("### Patient Information")

    age = st.number_input(
        "Age (years)", min_value=1, max_value=120, value=21, key="input_age"
    )
    gender = st.selectbox(
        "Gender",
        ["Prefer not to say", "Male", "Female"],
        key="input_gender",
    )
    height_cm = st.number_input(
        "Height (cm)",
        min_value=50.0,
        max_value=250.0,
        value=165.0,
        step=1.0,
        key="input_height",
    )
    weight_kg = st.number_input(
        "Weight (kg)",
        min_value=20.0,
        max_value=300.0,
        value=60.0,
        step=1.0,
        key="input_weight",
    )

    st.markdown("---")
    st.markdown("### Lifestyle")
    activity = st.selectbox(
        "Physical Activity",
        ["Sedentary", "Light", "Moderate", "Active"],
        key="input_activity",
    )

    st.markdown("---")
    st.markdown("### Dietary Preference")
    diet = st.radio(
        "Diet",
        ["Vegetarian", "Non-Vegetarian"],
        key="input_diet",
        horizontal=True,
    )

    st.markdown("---")
    st.markdown("### Restrictions / Allergies")
    restriction_choice = st.selectbox(
        "Restrictions",
        ["None", "Dairy", "Nuts", "Gluten"],
        key="input_restriction",
    )

    st.markdown("---")
    st.markdown("### Goal")
    goal = st.selectbox(
        "Goal",
        ["Maintain Weight", "Weight Loss", "Weight Gain"],
        key="input_goal",
    )

    st.markdown("---")
    st.button(
        "Load Demo Patient",
        on_click=load_demo,
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <h1>NutriAI</h1>
        <p>Personalized Nutrition Planner &mdash; A* Search Optimization</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="disclaimer">'
    "<strong>Disclaimer:</strong> This is an educational prototype and "
    "<strong>not</strong> medical or clinical nutritional advice. "
    "Food values are approximate demonstration data."
    "</div>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Generate button
# ---------------------------------------------------------------------------
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    generate = st.button(
        "Generate Personalized Meal Plan",
        use_container_width=True,
        type="primary",
    )


# ---------------------------------------------------------------------------
# Run the pipeline when button is clicked
# ---------------------------------------------------------------------------
if generate:
    # Validate
    errors: list[str] = []
    if age < 1 or age > 120:
        errors.append("Please enter a valid age (1–120).")
    if height_cm < 50 or height_cm > 250:
        errors.append("Please enter a valid height (50–250 cm).")
    if weight_kg < 20 or weight_kg > 300:
        errors.append("Please enter a valid weight (20–300 kg).")

    if errors:
        for e in errors:
            st.error(e)
    else:
        # ---- Nutritional target ----
        target = calculate_nutritional_target(
            weight_kg=weight_kg,
            height_cm=height_cm,
            age=age,
            gender=gender,
            activity_level=activity,
            goal=goal,
        )

        # ---- Load & filter food data ----
        csv_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "food_data.csv"
        )
        food_df = load_food_data(csv_path)

        restrictions: list[str] = []
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
            # ---- A* Search ----
            with st.spinner("Running A* search …"):
                t0 = time.perf_counter()
                result = astar_search(filtered_df, target)
                search_time = time.perf_counter() - t0

            # Persist everything needed for the tabs
            st.session_state["target"] = target
            st.session_state["result"] = result
            st.session_state["search_time"] = search_time
            st.session_state["filtered_df"] = filtered_df
            st.session_state["patient"] = {
                "age": age,
                "gender": gender,
                "height_cm": height_cm,
                "weight_kg": weight_kg,
                "activity": activity,
                "diet": diet,
                "restriction": restriction_choice,
                "goal": goal,
            }


# ---------------------------------------------------------------------------
# Results tabs (only shown when a meal plan has been generated)
# ---------------------------------------------------------------------------
if "result" not in st.session_state:
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-icon">&#9881;</div>
            <p>Configure a patient in the sidebar and click
            <strong>Generate Personalized Meal Plan</strong> to begin.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# Retrieve persisted state
target = st.session_state["target"]
result = st.session_state["result"]
search_time: float = st.session_state["search_time"]
filtered_df: pd.DataFrame = st.session_state["filtered_df"]
patient: dict = st.session_state["patient"]

tab_profile, tab_targets, tab_meal, tab_intake, tab_summary = st.tabs(
    [
        "Patient Profile",
        "Nutritional Targets",
        "A* Meal Plan",
        "Meal Intake Analysis",
        "Summary",
    ]
)


# ======================== TAB 1 – PATIENT PROFILE =========================
with tab_profile:
    section_header("Patient Overview")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        info_card("Age", f"{patient['age']} years")
    with c2:
        info_card("Gender", patient["gender"])
    with c3:
        info_card("Height", f"{patient['height_cm']:.0f} cm")
    with c4:
        info_card("Weight", f"{patient['weight_kg']:.0f} kg")

    st.markdown("<br>", unsafe_allow_html=True)

    c5, c6, c7 = st.columns(3)
    with c5:
        info_card("Activity Level", patient["activity"])
    with c6:
        info_card("Diet", patient["diet"])
    with c7:
        info_card("Goal", patient["goal"])

    if patient["restriction"] != "None":
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"**Restrictions:** {badge(patient['restriction'], 'warning')}",
            unsafe_allow_html=True,
        )


# ===================== TAB 2 – NUTRITIONAL TARGETS ========================
with tab_targets:
    section_header("Daily Nutritional Targets")

    st.caption(
        "Estimated using the Mifflin-St Jeor BMR equation and TDEE "
        "activity multipliers. This is for educational demonstration only."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Calories", f"{target.calories:,.0f}", "kcal / day", CAL_COLOR)
    with c2:
        metric_card("Protein", f"{target.protein:,.0f}", "grams (25%)", PRO_COLOR)
    with c3:
        metric_card("Carbs", f"{target.carbs:,.0f}", "grams (50%)", CARB_COLOR)
    with c4:
        metric_card("Fat", f"{target.fat:,.0f}", "grams (25%)", FAT_COLOR)

    # Target macro donut
    st.markdown("<br>", unsafe_allow_html=True)
    target_donut = go.Figure(
        go.Pie(
            labels=["Protein", "Carbs", "Fat"],
            values=[target.protein, target.carbs, target.fat],
            hole=0.55,
            marker=dict(
                colors=[PRO_COLOR, CARB_COLOR, FAT_COLOR],
                line=dict(color="#0e1117", width=2),
            ),
            textinfo="label+percent",
            textfont=dict(size=13, color="#e6edf3"),
        )
    )
    target_donut.update_layout(
        title="Target Macro Split",
        height=370,
        annotations=[
            dict(
                text=f"{target.calories:,.0f}<br>kcal",
                x=0.5,
                y=0.5,
                font=dict(size=18, color="#e6edf3", family="Inter"),
                showarrow=False,
            )
        ],
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(target_donut, use_container_width=True)


# ======================== TAB 3 – A* MEAL PLAN ============================
with tab_meal:
    if not result.meal:
        st.warning(
            "A* search could not find a suitable meal. "
            "Please try different settings."
        )
    else:
        section_header("Recommended Meal")

        # Build the meal dataframe
        meal_data = []
        for food_name, qty in result.meal:
            food_row = filtered_df[filtered_df["food_name"] == food_name].iloc[0]
            factor = qty / 100.0
            meal_data.append(
                {
                    "Food": food_name,
                    "Qty (g)": int(qty),
                    "Calories": round(food_row["calories_per_100g"] * factor, 1),
                    "Protein (g)": round(food_row["protein_per_100g"] * factor, 1),
                    "Carbs (g)": round(food_row["carbs_per_100g"] * factor, 1),
                    "Fat (g)": round(food_row["fat_per_100g"] * factor, 1),
                }
            )
        meal_df = pd.DataFrame(meal_data)
        st.dataframe(meal_df, use_container_width=True, hide_index=True)

        # Meal totals
        section_header("Meal Totals")
        tc1, tc2, tc3, tc4 = st.columns(4)
        with tc1:
            metric_card(
                "Total Calories",
                f"{result.total_calories:,.0f}",
                "kcal",
                CAL_COLOR,
            )
        with tc2:
            metric_card(
                "Total Protein",
                f"{result.total_protein:,.0f}",
                "grams",
                PRO_COLOR,
            )
        with tc3:
            metric_card(
                "Total Carbs",
                f"{result.total_carbs:,.0f}",
                "grams",
                CARB_COLOR,
            )
        with tc4:
            metric_card(
                "Total Fat", f"{result.total_fat:,.0f}", "grams", FAT_COLOR
            )

        # Charts
        st.markdown("<br>", unsafe_allow_html=True)
        chart_left, chart_right = st.columns(2)
        with chart_left:
            st.plotly_chart(
                build_target_vs_recommended_chart(target, result),
                use_container_width=True,
            )
        with chart_right:
            st.plotly_chart(
                build_macro_donut(result), use_container_width=True
            )

        # Target vs Recommended comparison table
        section_header("Detailed Comparison")
        comparison = pd.DataFrame(
            {
                "Nutrient": ["Calories (kcal)", "Protein (g)", "Carbs (g)", "Fat (g)"],
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
                "Deviation %": [
                    f"{abs(result.total_calories - target.calories) / target.calories * 100:.1f}%"
                    if target.calories
                    else "–",
                    f"{abs(result.total_protein - target.protein) / target.protein * 100:.1f}%"
                    if target.protein
                    else "–",
                    f"{abs(result.total_carbs - target.carbs) / target.carbs * 100:.1f}%"
                    if target.carbs
                    else "–",
                    f"{abs(result.total_fat - target.fat) / target.fat * 100:.1f}%"
                    if target.fat
                    else "–",
                ],
            }
        )
        st.dataframe(comparison, use_container_width=True, hide_index=True)

        # ---- A* Search Explorer ----
        with st.expander("A* Search Explorer"):
            # Status badge
            if result.goal_reached:
                status_html = badge("Goal Reached", "success")
            elif result.search_limit_reached:
                status_html = badge("Search Limit Reached", "warning")
            else:
                status_html = badge("Best Effort", "info")

            st.markdown(f"**Status:** {status_html}", unsafe_allow_html=True)

            # Stats grid
            st.markdown(
                f"""
                <div class="stat-grid">
                    <div class="stat-item">
                        <div class="stat-label">Nodes Explored</div>
                        <div class="stat-value">{result.nodes_explored}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Search Time</div>
                        <div class="stat-value">{search_time:.3f}s</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">g(n) Actual Cost</div>
                        <div class="stat-value">{result.g_cost:.4f}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">h(n) Heuristic</div>
                        <div class="stat-value">{result.h_cost:.4f}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">f(n) Total</div>
                        <div class="stat-value">{result.f_cost:.4f}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Goal Tolerance</div>
                        <div class="stat-value">&pm;15%</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Convergence chart
            if result.f_cost_history:
                st.plotly_chart(
                    build_convergence_chart(result.f_cost_history),
                    use_container_width=True,
                )

            # Explanation
            st.markdown(
                """
**How A* selects this meal:**

1. **State:** Each meal combination is a search state — a tuple of
   *(food\_index, quantity)* pairs.
2. **Initial state:** Empty meal `()`.
3. **Actions:** Add a food, increase/decrease quantity, or remove a food.
4. **g(n) — State cost:** Sum of squared normalized nutrient deviations
   from the target. Lower = closer to goals.
5. **h(n) — Heuristic:** Sum of absolute normalized nutrient differences.
   Guides the search toward the target but is *not strictly admissible*
   (h can overestimate g when deviations < 1), so the search does not
   guarantee optimality.
6. **f(n) = g(n) + h(n):** The search always expands the state with the
   lowest f(n), behaving as a heuristic best-first search.
7. **Goal:** All nutrients within ±15% of their targets.
8. **Priority queue:** Python `heapq` for efficient min-f(n) retrieval.
                """
            )


# ==================== TAB 4 – MEAL INTAKE ANALYSIS =======================
with tab_intake:
    section_header("Consumption Simulation")

    st.caption(
        "Simplified mathematical simulation. This prototype does NOT "
        "perform food-image segmentation or computer vision. "
        "See the Summary tab for how the paper's real pipeline differs."
    )

    if not result.meal:
        st.info("Generate a meal plan first.")
    else:
        consumption_pct = st.slider(
            "Approximate consumption percentage",
            min_value=0,
            max_value=100,
            value=100,
            step=5,
            key="consumption_slider",
        )
        cfactor = consumption_pct / 100.0

        consumed_cal = round(result.total_calories * cfactor, 1)
        consumed_pro = round(result.total_protein * cfactor, 1)
        consumed_carb = round(result.total_carbs * cfactor, 1)
        consumed_fat = round(result.total_fat * cfactor, 1)

        remaining_cal = round(target.calories - consumed_cal, 1)
        remaining_pro = round(target.protein - consumed_pro, 1)
        remaining_carb = round(target.carbs - consumed_carb, 1)
        remaining_fat = round(target.fat - consumed_fat, 1)

        section_header(f"Consumed ({consumption_pct}%)")
        cc1, cc2, cc3, cc4 = st.columns(4)
        with cc1:
            metric_card("Calories", f"{consumed_cal:,.0f}", "kcal", CAL_COLOR)
        with cc2:
            metric_card("Protein", f"{consumed_pro:,.0f}", "grams", PRO_COLOR)
        with cc3:
            metric_card("Carbs", f"{consumed_carb:,.0f}", "grams", CARB_COLOR)
        with cc4:
            metric_card("Fat", f"{consumed_fat:,.0f}", "grams", FAT_COLOR)

        section_header("Remaining vs Daily Target")
        deficit_df = pd.DataFrame(
            {
                "Nutrient": [
                    "Calories (kcal)",
                    "Protein (g)",
                    "Carbs (g)",
                    "Fat (g)",
                ],
                "Daily Target": [
                    target.calories,
                    target.protein,
                    target.carbs,
                    target.fat,
                ],
                "Consumed": [consumed_cal, consumed_pro, consumed_carb, consumed_fat],
                "Remaining": [
                    remaining_cal,
                    remaining_pro,
                    remaining_carb,
                    remaining_fat,
                ],
            }
        )
        st.dataframe(deficit_df, use_container_width=True, hide_index=True)

        # Progress bars
        section_header("Fulfilment")
        for nutrient_label, consumed_val, target_val in [
            ("Calories", consumed_cal, target.calories),
            ("Protein", consumed_pro, target.protein),
            ("Carbs", consumed_carb, target.carbs),
            ("Fat", consumed_fat, target.fat),
        ]:
            pct = (consumed_val / target_val * 100) if target_val > 0 else 0
            pct_clamped = max(0.0, min(100.0, pct))
            st.progress(
                pct_clamped / 100,
                text=f"{nutrient_label}: {pct:.0f}% of daily target",
            )


# ========================== TAB 5 – SUMMARY ==============================
with tab_summary:
    section_header("Pipeline Overview")

    # Visual pipeline
    p1, a1, p2, a2, p3, a3, p4 = st.columns([2, 1, 2, 1, 2, 1, 2])
    with p1:
        st.markdown(
            '<div class="pipeline-step">Patient Input</div>',
            unsafe_allow_html=True,
        )
    with a1:
        st.markdown(
            '<div class="pipeline-arrow">&rarr;</div>', unsafe_allow_html=True
        )
    with p2:
        st.markdown(
            '<div class="pipeline-step">BMR / TDEE</div>',
            unsafe_allow_html=True,
        )
    with a2:
        st.markdown(
            '<div class="pipeline-arrow">&rarr;</div>', unsafe_allow_html=True
        )
    with p3:
        st.markdown(
            '<div class="pipeline-step">A* Search</div>',
            unsafe_allow_html=True,
        )
    with a3:
        st.markdown(
            '<div class="pipeline-arrow">&rarr;</div>', unsafe_allow_html=True
        )
    with p4:
        st.markdown(
            '<div class="pipeline-step">Meal Plan</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Research paper connection
    with st.expander("Connection to Research Paper"):
        st.markdown(
            """
This prototype is inspired by the personalized nutrition architecture
presented in:

> **"An AI Based Approach for Personalized Nutrition and Food Menu Planning"**
> Authors: Khalid Azzimani, Hayat Bihri, Asma Dahmi, Salma Azzouzi,
> My El Hassan Charaf — IEEE ICECOCS 2022

| Paper's Architecture | Our Implementation |
|---|---|
| Patient information collection | User input form |
| Nutritional requirement estimation | Simplified BMR/TDEE calculation |
| Food restrictions filtering | Restriction-based food filtering |
| Food composition data | Local CSV dataset (30 foods) |
| Personalized meal planning | A* heuristic best-first search |
| Before/after food analysis | Simplified consumption simulation |
| Computer vision / segmentation | Not implemented (future scope) |
            """
        )

    # Limitations
    with st.expander("Limitations"):
        st.markdown(
            """
- The food dataset contains approximate nutritional values for
  demonstration purposes only.
- BMR/TDEE formulas are simplified estimates, not clinical measurements.
- The A* heuristic is not strictly admissible; the search does not
  guarantee an optimal solution but consistently finds meals within
  ±15% of all nutrient targets.
- The consumption analysis is a mathematical simulation only —
  no real computer vision or food image processing is implemented.
- This is **NOT** a medical device and should not be used for clinical
  decision-making.
            """
        )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "NutriAI — Academic Prototype | "
    "Inspired by IEEE ICECOCS 2022 Paper | "
    "A* Heuristic Best-First Search"
)
