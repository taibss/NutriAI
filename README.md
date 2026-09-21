# NutriAI – Personalized Nutrition Planner

An AI-based prototype for personalized meal planning using A* search.

## 1. Project Overview

NutriAI is a small academic prototype that demonstrates how AI search techniques can be applied to personalized nutrition and meal planning. The system collects user information, estimates nutritional requirements, filters foods based on dietary restrictions, and uses A* search to find a meal combination that best matches the user's nutritional needs.

## 2. Problem Statement

Traditional dietary assessment methods can be time-consuming, dependent on patient memory and interviews, and may require skilled clinical assessment. Automated systems can help streamline the process of analyzing nutritional requirements and generating personalized meal recommendations.

The research paper "An AI Based Approach for Personalized Nutrition and Food Menu Planning" (IEEE ICECOCS 2022) proposes an AI-based architecture to address this problem.

## 3. Objective

Build a small AI prototype that demonstrates:
- Personalized nutritional requirement estimation
- Dietary restriction filtering
- Meal selection using A* search
- Nutritional comparison (target vs. recommended)
- Simplified meal consumption simulation

## 4. Research Paper

**Title:** "An AI Based Approach for Personalized Nutrition and Food Menu Planning"

**Authors:** Khalid Azzimani, Hayat Bihri, Asma Dahmi, Salma Azzouzi, My El Hassan Charaf

**Published in:** IEEE ICECOCS 2022

The paper proposes an AI-based personalized nutrition system that:
- Collects patient information and nutritional history
- Determines nutritional requirements
- Considers foods that are not allowed/restricted
- Uses food composition data
- Analyzes food before and after consumption
- Estimates consumed food/nutrients
- Identifies nutritional deficits
- Redistributes missing nutrients in subsequent meals

## 5. Important Methodology Note

> **The selected research paper does not explicitly specify A* search. In this academic prototype, A* is proposed as a search technique for the meal-selection component.**

A* is our proposed implementation for solving the meal planning problem. The paper describes the overall architecture and workflow, but does not prescribe a specific search algorithm.

## 6. System Architecture

```
User Input (Age, Weight, Height, Activity, Diet, Restrictions, Goal)
    ↓
Nutritional Requirement Calculation (BMR/TDEE)
    ↓
Food Filtering (Dietary Restrictions)
    ↓
A* Search (Meal Selection)
    ↓
Personalized Meal Recommendation
    ↓
Nutritional Comparison (Target vs. Recommended)
    ↓
Consumption Simulation (Optional)
```

## 7. Why A*?

The meal planning problem can be viewed as a search problem:
- There are many possible food combinations
- Each combination has different nutritional values
- We want to find the combination closest to our target

A* is suitable because:
- It combines actual cost g(n) with estimated future cost h(n)
- It is complete and optimal (within the search space)
- It efficiently explores the most promising states first
- It uses a priority queue for efficient state retrieval

## 8. A* Components

| Component | Description |
|-----------|-------------|
| **State** | A meal represented as a tuple of (food_index, quantity_grams) pairs |
| **Initial State** | Empty meal: `()` |
| **Actions** | Add a food, increase quantity, decrease quantity, remove a food |
| **Transition** | Applying an action produces a new meal state |
| **g(n)** | Squared normalized nutrient deviation from target (actual cost) |
| **h(n)** | Normalized absolute nutrient differences (heuristic estimate) |
| **f(n)** | `f(n) = g(n) + h(n)` (total estimated cost) |
| **Goal State** | Nutrients within ±15% of all targets, or search limit reached |
| **Priority Queue** | Python's `heapq` for efficient min-f(n) retrieval |

## 9. Technologies

- **Python 3.10+**
- **Streamlit** – Web-based UI framework
- **Pandas** – Data manipulation and CSV handling
- **NumPy** – Numerical operations
- **pytest** – Unit testing

No paid APIs or external AI services are used.

## 10. Project Structure

```
NutriAI/
│
├── app.py                  # Main Streamlit application
├── astar.py                # A* search implementation
├── nutrition.py            # BMR/TDEE nutritional calculations
├── food_data.csv           # Local food dataset (30 foods)
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── .gitignore              # Git ignore rules
│
├── assets/
│   └── README.md           # Assets directory placeholder
│
└── tests/
    └── test_astar.py       # Unit tests
```

## 11. Installation

### Prerequisites
- Python 3.10 or higher
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/NutriAI.git
cd NutriAI

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# macOS/Linux:
source .venv/bin/activate

# Windows:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 12. Run

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

## 13. Example Demo

1. Click **"Load Demo Patient"** in the sidebar to fill in sample values
2. Click **"Generate Personalized Meal Plan"**
3. View the recommended meal, nutritional comparison, and consumption simulation
4. Expand "How did A* select this meal?" to see the A* explanation

## 14. Running Tests

```bash
python -m pytest tests/test_astar.py -v
```

## 15. Limitations

- The food dataset contains approximate nutritional values for demonstration only
- BMR/TDEE formulas are simplified estimates, not clinical measurements
- A* search operates on a limited discrete state space (max 5 foods, 25g steps)
- The consumption analysis is a mathematical simulation only
- No real computer vision or food image processing is implemented
- This is NOT a medical device and should not be used for clinical decision-making

## 16. Future Scope

- Food image recognition using computer vision
- RGB-D camera integration for food analysis
- Food segmentation using deep learning
- Better portion estimation
- Larger regional food composition databases
- More advanced nutritional recommendation models
- Real patient datasets for validation
- Integration with the complete architecture described in the paper
- Multi-meal daily planning with deficit redistribution

## 17. Connection to the Research Paper

This prototype implements a subset of the architecture described in the paper:

| Paper's Architecture | Our Implementation |
|---------------------|-------------------|
| Patient information collection | User input form ✓ |
| Nutritional requirement estimation | Simplified BMR/TDEE calculation ✓ |
| Food restrictions filtering | Restriction-based food filtering ✓ |
| Food composition data | Local CSV dataset ✓ |
| Personalized meal planning | A* search (our proposed method) ✓ |
| Before/after food analysis | Simplified consumption simulation ✓ |
| Computer vision / food segmentation | Not implemented (future scope) |

## 18. Disclaimer

This project is an academic prototype and is not intended to provide medical or clinical nutritional advice. The nutritional calculations are simplified estimates for educational demonstration purposes only.

## 19. License

This project is for academic/educational purposes.

## 20. Acknowledgments

Based on the research paper: "An AI Based Approach for Personalized Nutrition and Food Menu Planning" by Azzimani et al., IEEE ICECOCS 2022.
# NutriAI
