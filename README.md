# NBA Spatial Shot Quality Model
👉 **[Live Interactive Dashboard: Click Here to Try the App!](https://abhinav-nba-shot-quality.streamlit.app)**
## Executive Summary
This repository demonstrates an Expected Points (xPoints) model built from NBA shot data to support better offensive decision-making. xPoints estimates the expected scoring value of each shot opportunity, helping teams optimize shot selection, allocate player roles more effectively, and identify market inefficiencies in shot creation and defensive coverage.

## Domain Expertise
The modeling approach reflects a blend of domain knowledge and analytical rigor. My background includes national-level basketball competition, selection to the NBA Academy, and collegiate basketball experience in India. That competitive insight, paired with my transition from an English Honours undergraduate degree to an MSBA, allows me to translate technical analytics into clear, actionable sports strategy.

## The Dataset
The model uses 219,000+ live shot events from the 2025-26 NBA regular season, sourced directly through the official NBA API. The dataset includes spatial coordinates, shot type, game context, and outcome information.

## The Tech Stack
- Python
- XGBoost
- Scikit-Learn
- Matplotlib
- Seaborn
- Streamlit

## Feature Engineering
Key engineered features include:
- geometric shot angles calculated using trigonometry (`np.arctan2`)
- converted game clock values into continuous time variables
- spatial shot locations combined with contextual game-state data

## Model Performance
- Accuracy: 63%
- ROC-AUC: 0.65

In the context of human sports performance, this level of accuracy is a meaningful baseline. Basketball outcomes are influenced by high-variance events, defensive actions, and player decision-making, so a model that consistently improves expected shot quality provides useful insight.

## How to Run
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Open the notebook:
```bash
jupyter notebook data_collection.ipynb
```
3. Run the notebook cells in order to fetch data, clean it, train the model, and generate visualizations.
