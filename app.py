import pickle

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, Rectangle

st.set_page_config(page_title='NBA Shot Quality Dashboard', layout='wide')


@st.cache_data
def load_shot_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_model(path: str):
    with open(path, 'rb') as file:
        return pickle.load(file)


def prepare_features(df: pd.DataFrame, model) -> pd.DataFrame:
    feature_columns = [
        'SHOT_DISTANCE',
        'SHOT_ANGLE',
        'TIME_REMAINING_QUARTER',
        'PERIOD',
        'ACTION_TYPE',
        'SHOT_TYPE',
        'SHOT_ZONE_BASIC',
        'SHOT_ZONE_AREA'
    ]

    X = df[feature_columns].copy()
    X = pd.get_dummies(
        X,
        columns=[
            'ACTION_TYPE',
            'SHOT_TYPE',
            'SHOT_ZONE_BASIC',
            'SHOT_ZONE_AREA'
        ],
        drop_first=True
    )

    if hasattr(model, 'feature_names_in_'):
        expected_columns = list(model.feature_names_in_)
    else:
        booster = model.get_booster()
        expected_columns = list(booster.feature_names)

    X = X.reindex(columns=expected_columns, fill_value=0)
    return X


def draw_half_court(ax):
    hoop = Circle((0, 0), radius=7.5, linewidth=2, fill=False, color='black', zorder=2)
    backboard = Rectangle((-30, -7.5), 60, 1, linewidth=2, color='black', zorder=2)
    paint = Rectangle((-80, 0), 160, 190, linewidth=2, fill=False, color='black', zorder=2)
    free_throw = Arc((0, 190), width=120, height=120, theta1=0, theta2=180, linewidth=2, color='black', zorder=2)
    three_point = Arc((0, 0), width=475, height=475, theta1=0, theta2=180, linewidth=2, color='black', zorder=2)

    court_elements = [hoop, backboard, paint, free_throw, three_point]
    for element in court_elements:
        ax.add_patch(element)

    ax.set_xlim(-250, 250)
    ax.set_ylim(-50, 470)
    ax.set_aspect('equal')
    ax.axis('off')


def plot_player_hexbin(player_shots: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 11))

    hb = ax.hexbin(
        player_shots['LOC_X'],
        player_shots['LOC_Y'],
        C=player_shots['xPoints_Probability'],
        gridsize=35,
        reduce_C_function=np.mean,
        cmap='YlOrRd',
        mincnt=1,
        alpha=0.7,
        zorder=0
    )

    draw_half_court(ax)

    cbar = fig.colorbar(hb, ax=ax, shrink=0.8)
    cbar.set_label('Average Predicted xPoints Probability', fontsize=12)

    ax.set_title('Hexbin Shot Quality Map — Predicted xPoints Probability', fontsize=16, weight='bold')
    plt.tight_layout()
    return fig


def main():
    st.title('NBA Shot Quality Dashboard')
    st.markdown('Select a player to review their shot distribution and model-predicted shot quality.')

    shots = load_shot_data('nba_cleaned_shots.csv')
    model = load_model('nba_shot_quality_model.pkl')

    X_full = prepare_features(shots, model)
    shots['xPoints_Probability'] = model.predict_proba(X_full)[:, 1]

    players = sorted(shots['PLAYER_NAME'].unique())
    selected_player = st.sidebar.selectbox('Choose a player', players)

    player_shots = shots[shots['PLAYER_NAME'] == selected_player].copy()

    if player_shots.empty:
        st.warning('No shots found for the selected player.')
        return

    total_shots = len(player_shots)
    actual_fg = player_shots['SHOT_MADE_FLAG'].mean() * 100
    expected_fg = player_shots['xPoints_Probability'].mean() * 100

    c1, c2, c3 = st.columns(3)
    c1.metric('Total Shots Taken', f'{total_shots}')
    c2.metric('Actual FG%', f'{actual_fg:.1f}%')
    c3.metric('Expected FG%', f'{expected_fg:.1f}%')

    st.subheader(f'Shot Quality Heatmap for {selected_player}')
    fig = plot_player_hexbin(player_shots)
    st.pyplot(fig)


if __name__ == '__main__':
    main()
