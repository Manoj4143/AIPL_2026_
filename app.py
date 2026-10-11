"""
IPL Match Outcome Prediction System
Interactive Machine Learning & Analytics Dashboard (Phase 2)
"""

import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.prediction import (
    predict_match,
    get_available_teams,
    get_available_venues,
    get_venue_city_map,
    get_venue_details,
    get_team_comparison,
    get_prediction_explanation,
    manage_prediction_history,
    load_model_and_artifacts
)

# Page configuration
st.set_page_config(
    page_title="IPL Match Outcome Prediction System",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for professional analytics UI
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.1rem;
        letter-spacing: -0.02em;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .outcome-card {
        padding: 1.25rem 1.5rem;
        border-radius: 0.75rem;
        border-left: 6px solid #10B981;
        background: #F0FDF4;
        margin-bottom: 1.2rem;
    }
    .outcome-header {
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        color: #047857;
        letter-spacing: 0.06em;
    }
    .outcome-value {
        font-size: 2rem;
        font-weight: 800;
        color: #064E3B;
    }
    .metric-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-high {
        background: #FEF3C7;
        color: #92400E;
    }
    .badge-normal {
        background: #E0E7FF;
        color: #3730A3;
    }
    .disclaimer-box {
        background: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 0.9rem 1.2rem;
        font-size: 0.88rem;
        color: #92400E;
        border-radius: 0.375rem;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


def check_model_availability():
    """Checks if the trained model artifact is available."""
    try:
        load_model_and_artifacts()
        return True, ""
    except FileNotFoundError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Unexpected error while loading model: {str(e)}"


def format_class_description(cls_name, team_a, team_b):
    """Returns human-readable explanation for each outcome class."""
    if cls_name == 'A_small':
        return f"{team_a} wins by small margin (≤ 20 runs or ≤ 5 wkts)"
    elif cls_name == 'A_big':
        return f"{team_a} wins by large margin (> 20 runs or ≥ 6 wkts)"
    elif cls_name == 'B_small':
        return f"{team_b} wins by small margin (≤ 20 runs or ≤ 5 wkts)"
    elif cls_name == 'B_big':
        return f"{team_b} wins by large margin (> 20 runs or ≥ 6 wkts)"
    return cls_name


def init_session_state():
    """Initializes session state tracking for predictions and explanations."""
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    if 'latest_result' not in st.session_state:
        st.session_state.latest_result = None
    if 'latest_inputs' not in st.session_state:
        st.session_state.latest_inputs = None


def main():
    init_session_state()

    # Sidebar: Model Reference & System Info
    with st.sidebar:
        st.markdown("### ⚙️ System Info")
        model_ready, model_error = check_model_availability()
        if model_ready:
            st.success("● CatBoost Engine Online (200 Iterations)")
        else:
            st.error("● Model Artifact Offline")

        st.markdown("---")
        st.markdown("### 📖 Outcome Classes")
        st.markdown(r"""
        - **`A_small`**: Team A wins by $\le$ 20 runs or $\le$ 5 wkts
        - **`A_big`**: Team A wins by $>$ 20 runs or $\ge$ 6 wkts
        - **`B_small`**: Team B wins by $\le$ 20 runs or $\le$ 5 wkts
        - **`B_big`**: Team B wins by $>$ 20 runs or $\ge$ 6 wkts
        """)
        st.markdown("---")
        st.markdown("### 📊 Metric Definition")
        st.markdown("""
        Evaluated on multiclass column-wise log loss. Probabilities are strictly normalized to sum to $1.0$.
        """)

    # 1. Header and Project Description
    st.markdown('<div class="main-title">🏏 IPL Match Outcome Prediction System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">'
        'Probabilistic machine-learning analytics dashboard for IPL match outcomes '
        '(<code>A_small</code>, <code>A_big</code>, <code>B_small</code>, <code>B_big</code>), '
        'powered by historical ball-by-ball performance, toss dynamics, and CatBoost.'
        '</div>',
        unsafe_allow_html=True
    )

    if not model_ready:
        st.error(f"⚠️ **Model File Missing or Inaccessible**\n\n{model_error}")
        st.info("💡 **Resolution**: Run `python -m src.preprocessing` in your terminal to train and serialize the model to `models/catboost_model.cbm`.")
        return

    # Load metadata
    try:
        available_teams = get_available_teams()
        available_venues = get_available_venues()
        venue_city_map = get_venue_city_map()
    except Exception as e:
        st.error(f"Error loading system metadata: {e}")
        return

    # 2. Main Dashboard Layout (Tabs)
    tab_pred, tab_explain, tab_teams, tab_venue, tab_history = st.tabs([
        "🔮 Match Prediction",
        "🔍 Model Explanation",
        "⚔️ Team Comparison",
        "🏟️ Venue Insights",
        "📜 Session History"
    ])

    # ----------------------------------------------------
    # TAB 1: MATCH PREDICTION
    # ----------------------------------------------------
    with tab_pred:
        st.subheader("Match Configuration")

        input_col1, input_col2 = st.columns(2)

        with input_col1:
            st.markdown("##### 👥 Team Selection")
            team_a = st.selectbox(
                "Team A",
                options=available_teams,
                index=0,
                key="select_team_a",
                help="Select the first participating team."
            )

            team_b = st.selectbox(
                "Team B",
                options=available_teams,
                index=min(1, len(available_teams) - 1),
                key="select_team_b",
                help="Select the second participating team."
            )

            same_team_selected = (team_a == team_b)
            if same_team_selected:
                st.warning("⚠️ **Invalid Selection**: Team A and Team B cannot be the same team. Please select two distinct teams.")

        with input_col2:
            st.markdown("##### 🏟️ Venue & Conditions")
            venue = st.selectbox(
                "Venue",
                options=available_venues,
                index=0,
                key="select_venue",
                help="Select the match stadium."
            )

            suggested_city = venue_city_map.get(venue, "")
            all_cities = sorted(list(set(c for c in venue_city_map.values() if c)))
            city_options = [""] + all_cities if all_cities else [""]
            city_idx = city_options.index(suggested_city) if suggested_city in city_options else 0

            city = st.selectbox(
                "City (optional)",
                options=city_options,
                index=city_idx,
                key="select_city",
                help="Host city for the venue."
            )

        st.markdown("##### 🪙 Toss Setup")
        toss_c1, toss_c2 = st.columns(2)

        with toss_c1:
            toss_winner_options = [team_a] if same_team_selected else [team_a, team_b]
            toss_winner = st.selectbox(
                "Toss Winner",
                options=toss_winner_options,
                index=0,
                key="select_toss_winner",
                help="Team that won the coin toss."
            )

        with toss_c2:
            toss_decision = st.selectbox(
                "Toss Decision",
                options=["field", "bat"],
                index=0,
                key="select_toss_decision",
                help="Decision made by toss winner: 'field' (chase) or 'bat' (set target)."
            )

        st.write("")
        predict_clicked = st.button(
            "🔮 Predict Match Outcome",
            type="primary",
            disabled=same_team_selected,
            use_container_width=True
        )

        if predict_clicked:
            if same_team_selected:
                st.error("Cannot predict: Please select two different teams for Team A and Team B.")
            else:
                with st.spinner("Evaluating historical profiles, venue factors, and toss dynamics..."):
                    try:
                        result = predict_match(
                            team_a=team_a,
                            team_b=team_b,
                            venue=venue,
                            city=city if city else None,
                            toss_winner=toss_winner,
                            toss_decision=toss_decision
                        )

                        # Update session state
                        st.session_state.latest_result = result
                        st.session_state.latest_inputs = {
                            'team_a': team_a,
                            'team_b': team_b,
                            'venue': venue,
                            'city': city,
                            'toss_winner': toss_winner,
                            'toss_decision': toss_decision
                        }

                        # Record to session history
                        history_entry = {
                            'Team A': team_a,
                            'Team B': team_b,
                            'Venue': venue,
                            'Toss Winner': toss_winner,
                            'Toss Decision': toss_decision,
                            'Predicted Outcome': result['most_likely_outcome'],
                            'Confidence': f"{result['most_likely_probability'] * 100:.2f}%",
                            'A_small (%)': f"{result['A_small'] * 100:.2f}%",
                            'A_big (%)': f"{result['A_big'] * 100:.2f}%",
                            'B_small (%)': f"{result['B_small'] * 100:.2f}%",
                            'B_big (%)': f"{result['B_big'] * 100:.2f}%"
                        }
                        st.session_state.prediction_history = manage_prediction_history(
                            st.session_state.prediction_history,
                            new_entry=history_entry
                        )

                    except ValueError as ve:
                        st.error(f"Input Validation Error: {ve}")
                    except Exception as ex:
                        st.error(f"Prediction Failed: {ex}")

        # Render Latest Prediction Results
        current_res = st.session_state.latest_result
        if current_res is not None:
            st.markdown("---")
            st.subheader("Match Outcome Prediction Results")

            # Match Situation Banner
            st.info(
                f"📋 **Innings Assignment**: **{current_res['batting_first']}** will bat first (set target) | "
                f"**{current_res['batting_second']}** will bat second (chase) at **{current_res['venue']}**."
            )

            # Most Likely Outcome Highlight Card
            most_likely = current_res['most_likely_outcome']
            most_likely_prob = current_res['most_likely_probability']
            most_likely_desc = format_class_description(
                most_likely, current_res['team_a'], current_res['team_b']
            )

            st.markdown(
                f"""
                <div class="outcome-card">
                    <div class="outcome-header">Most Likely Predicted Outcome</div>
                    <div class="outcome-value">{most_likely} ({most_likely_prob * 100:.2f}%)</div>
                    <div style="font-size: 1.05rem; color: #065F46; margin-top: 0.3rem;">
                        <strong>{most_likely_desc}</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # 4 Class Metric Cards
            p1, p2, p3, p4 = st.columns(4)
            with p1:
                st.metric(
                    label=f"A_small ({current_res['team_a'][:12]})",
                    value=f"{current_res['A_small'] * 100:.2f}%",
                    delta="Win ≤20r / ≤5w"
                )
            with p2:
                st.metric(
                    label=f"A_big ({current_res['team_a'][:12]})",
                    value=f"{current_res['A_big'] * 100:.2f}%",
                    delta="Win >20r / ≥6w"
                )
            with p3:
                st.metric(
                    label=f"B_small ({current_res['team_b'][:12]})",
                    value=f"{current_res['B_small'] * 100:.2f}%",
                    delta="Win ≤20r / ≤5w"
                )
            with p4:
                st.metric(
                    label=f"B_big ({current_res['team_b'][:12]})",
                    value=f"{current_res['B_big'] * 100:.2f}%",
                    delta="Win >20r / ≥6w"
                )

            st.write("")

            # Visualizations & Table
            chart_col, table_col = st.columns([3, 2])

            with chart_col:
                st.markdown("##### 📊 Probability Distribution")
                classes = ['A_small', 'A_big', 'B_small', 'B_big']
                percentages = [current_res[c] * 100 for c in classes]

                bar_colors = [
                    '#10B981' if c == most_likely else '#2563EB' if c.startswith('A') else '#7C3AED'
                    for c in classes
                ]

                fig = go.Figure(data=[
                    go.Bar(
                        x=classes,
                        y=percentages,
                        text=[f"{p:.2f}%" for p in percentages],
                        textposition='outside',
                        marker_color=bar_colors,
                        hovertemplate="<b>%{x}</b><br>Probability: %{y:.2f}%<extra></extra>"
                    )
                ])
                fig.update_layout(
                    yaxis_title="Probability (%)",
                    xaxis_title="Outcome Class",
                    yaxis_range=[0, max(percentages) * 1.25],
                    template="plotly_white",
                    height=330,
                    margin=dict(l=20, r=20, t=20, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)

            with table_col:
                st.markdown("##### 📋 Breakdown Table")
                table_data = [
                    {
                        "Class": "A_small",
                        "Meaning": f"{current_res['team_a']} Small Win",
                        "Probability": f"{current_res['A_small']:.4f}",
                        "Share": f"{current_res['A_small'] * 100:.2f}%"
                    },
                    {
                        "Class": "A_big",
                        "Meaning": f"{current_res['team_a']} Big Win",
                        "Probability": f"{current_res['A_big']:.4f}",
                        "Share": f"{current_res['A_big'] * 100:.2f}%"
                    },
                    {
                        "Class": "B_small",
                        "Meaning": f"{current_res['team_b']} Small Win",
                        "Probability": f"{current_res['B_small']:.4f}",
                        "Share": f"{current_res['B_small'] * 100:.2f}%"
                    },
                    {
                        "Class": "B_big",
                        "Meaning": f"{current_res['team_b']} Big Win",
                        "Probability": f"{current_res['B_big']:.4f}",
                        "Share": f"{current_res['B_big'] * 100:.2f}%"
                    }
                ]
                st.dataframe(pd.DataFrame(table_data), hide_index=True, use_container_width=True)

            # Validation Confirmation
            prob_sum = current_res['probability_sum']
            if current_res['is_valid_sum']:
                st.success(f"✅ **Probability Integrity Validated**: All 4 predicted probabilities sum to 100% ({prob_sum:.6f} ≈ 1.000000).")
            else:
                st.error(f"❌ **Validation Warning**: Probability sum deviation detected ({prob_sum:.6f}).")
        else:
            st.info("👋 Select match parameters above and click **Predict Match Outcome** to generate probabilistic predictions.")

    # ----------------------------------------------------
    # TAB 2: MODEL EXPLANATION
    # ----------------------------------------------------
    with tab_explain:
        st.subheader("Model Prediction Explanation")
        st.markdown(
            "Understand why the CatBoost model predicted the probabilistic outcome using exact feature inputs and "
            "native TreeSHAP attribution."
        )

        current_res = st.session_state.latest_result
        if current_res is None:
            st.info("ℹ️ Please run a prediction in the **Match Prediction** tab first to see the model explanation.")
        else:
            inputs = st.session_state.latest_inputs
            try:
                explanation = get_prediction_explanation(
                    team_a=inputs['team_a'],
                    team_b=inputs['team_b'],
                    venue=inputs['venue'],
                    toss_winner=inputs['toss_winner'],
                    toss_decision=inputs['toss_decision'],
                    result=current_res
                )

                st.markdown(f"#### 🎯 Explaining Predicted Class: `{explanation['most_likely_outcome']}`")

                # Analytical Key Factors
                st.markdown("##### 📌 Analytical Highlights")
                for point in explanation['summary_points']:
                    st.markdown(f"- {point}")

                st.write("")

                # TreeSHAP Feature Attribution Chart
                st.markdown("##### 🌲 Feature Contributions (Native TreeSHAP)")
                st.caption(
                    "Positive values (green) increased the probability of this outcome; "
                    "negative values (red) reduced the probability of this outcome relative to the baseline."
                )

                contribs = explanation['feature_contributions']
                labels = [c['label'] for c in contribs]
                shap_vals = [c['attribution'] for c in contribs]

                # Reverse order for horizontal bar chart (top impact at the top)
                labels_rev = labels[::-1]
                shap_vals_rev = shap_vals[::-1]
                colors_rev = ['#10B981' if v >= 0 else '#EF4444' for v in shap_vals_rev]

                fig_shap = go.Figure(go.Bar(
                    x=shap_vals_rev,
                    y=labels_rev,
                    orientation='h',
                    marker_color=colors_rev,
                    text=[f"{v:+.4f}" for v in shap_vals_rev],
                    textposition='outside',
                    hovertemplate="<b>%{y}</b><br>SHAP Contribution: %{x:+.4f}<extra></extra>"
                ))
                fig_shap.update_layout(
                    xaxis_title="SHAP Attribution (Log-Odds Impact)",
                    yaxis_title="Feature Factor",
                    template="plotly_white",
                    height=420,
                    margin=dict(l=20, r=40, t=20, b=20)
                )
                st.plotly_chart(fig_shap, use_container_width=True)

                # Feature Values Summary Table
                st.markdown("##### 📑 Raw Input Features Supplied to Model")
                feat_table = [
                    {
                        "Feature Column": c['feature'],
                        "Factor Description": c['label'],
                        "Supplied Value": f"{c['value']:.4f}" if isinstance(c['value'], (int, float)) else str(c['value']),
                        "SHAP Impact": f"{c['attribution']:+.4f}"
                    }
                    for c in contribs
                ]
                st.dataframe(pd.DataFrame(feat_table), hide_index=True, use_container_width=True)

            except Exception as e:
                st.error(f"Error computing model explanation: {e}")

    # ----------------------------------------------------
    # TAB 3: TEAM COMPARISON
    # ----------------------------------------------------
    with tab_teams:
        st.subheader("Historical Head-to-Head & Team Strength")
        st.markdown(
            "Compare historical performance metrics between the two teams based on training data from 2008 onwards."
        )

        curr_team_a = st.session_state.latest_inputs['team_a'] if st.session_state.latest_inputs else available_teams[0]
        curr_team_b = st.session_state.latest_inputs['team_b'] if st.session_state.latest_inputs else available_teams[1]

        comp_col1, comp_col2 = st.columns(2)
        with comp_col1:
            team_sel_a = st.selectbox("Compare Team A", options=available_teams, index=available_teams.index(curr_team_a) if curr_team_a in available_teams else 0, key="comp_t_a")
        with comp_col2:
            team_sel_b = st.selectbox("Compare Team B", options=available_teams, index=available_teams.index(curr_team_b) if curr_team_b in available_teams else 1, key="comp_t_b")

        if team_sel_a == team_sel_b:
            st.warning("Select two different teams to view head-to-head metrics.")
        else:
            team_comp = get_team_comparison(team_sel_a, team_sel_b)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"#### 🔵 {team_sel_a}")
                st.metric("Overall Win Rate", f"{team_comp['team_a_win_rate']*100:.1f}%")
                st.metric("Batting First Win Rate", f"{team_comp['team_a_bat1st_rate']*100:.1f}%")
                st.caption(f"Historical Matches: {team_comp['team_a_matches']}")

            with c2:
                st.markdown(f"#### 🟣 {team_sel_b}")
                st.metric("Overall Win Rate", f"{team_comp['team_b_win_rate']*100:.1f}%")
                st.metric("Batting First Win Rate", f"{team_comp['team_b_bat1st_rate']*100:.1f}%")
                st.caption(f"Historical Matches: {team_comp['team_b_matches']}")

            with c3:
                st.markdown("#### ⚖️ Relative Difference")
                diff = team_comp['win_rate_diff']
                favored_team = team_sel_a if diff >= 0 else team_sel_b
                st.metric(
                    "Win Rate Differential",
                    f"{abs(diff)*100:.1f}%",
                    delta=f"{favored_team} favored"
                )
                st.info(
                    f"**Historical Advantage**: **{favored_team}** carries a historical strength edge of "
                    f"**{abs(diff)*100:.1f}%** over their competitor."
                )

            # Side-by-side comparison chart
            categories = ['Overall Win Rate (%)', 'Bat-First Win Rate (%)']
            fig_comp = go.Figure(data=[
                go.Bar(
                    name=team_sel_a,
                    x=categories,
                    y=[team_comp['team_a_win_rate']*100, team_comp['team_a_bat1st_rate']*100],
                    marker_color='#2563EB',
                    text=[f"{team_comp['team_a_win_rate']*100:.1f}%", f"{team_comp['team_a_bat1st_rate']*100:.1f}%"],
                    textposition='outside'
                ),
                go.Bar(
                    name=team_sel_b,
                    x=categories,
                    y=[team_comp['team_b_win_rate']*100, team_comp['team_b_bat1st_rate']*100],
                    marker_color='#7C3AED',
                    text=[f"{team_comp['team_b_win_rate']*100:.1f}%", f"{team_comp['team_b_bat1st_rate']*100:.1f}%"],
                    textposition='outside'
                )
            ])
            fig_comp.update_layout(
                barmode='group',
                yaxis_title="Win Rate (%)",
                yaxis_range=[0, 100],
                template="plotly_white",
                height=340,
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig_comp, use_container_width=True)

    # ----------------------------------------------------
    # TAB 4: VENUE INSIGHTS
    # ----------------------------------------------------
    with tab_venue:
        st.subheader("Venue Scoring Profile & Stadium Dynamics")
        st.markdown(
            "Inspect historical ground characteristics, average match runs, and scoring patterns."
        )

        curr_venue = st.session_state.latest_inputs['venue'] if st.session_state.latest_inputs else available_venues[0]
        venue_choice = st.selectbox(
            "Select Venue to Inspect",
            options=available_venues,
            index=available_venues.index(curr_venue) if curr_venue in available_venues else 0,
            key="venue_insights_select"
        )

        v_info = get_venue_details(venue_choice)

        vc1, vc2, vc3, vc4 = st.columns(4)
        with vc1:
            st.metric("Host City", v_info['city'])
        with vc2:
            st.metric("Average Total Runs", f"{v_info['avg_score']:.1f}")
        with vc3:
            st.metric("Matches in Training Data", v_info['match_count'])
        with vc4:
            st.markdown("##### Scoring Classification")
            badge_class = "badge-high" if v_info['is_high_scoring'] else "badge-normal"
            st.markdown(
                f'<span class="metric-badge {badge_class}">{v_info["scoring_type"]}</span>',
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown("##### 🏟️ Ground Profile Analysis")
        if v_info['is_high_scoring']:
            st.write(
                f"**{v_info['venue']}** historically features higher-scoring encounters averaging "
                f"**{v_info['avg_score']:.1f}** runs per match (combined both innings). "
                "Teams defending a target typically require a higher par score to protect victory."
            )
        else:
            st.write(
                f"**{v_info['venue']}** offers a balanced contest between bat and ball, averaging "
                f"**{v_info['avg_score']:.1f}** runs per match. "
                "Spinners and seamers with good variations frequently gain purchase here."
            )

    # ----------------------------------------------------
    # TAB 5: SESSION HISTORY
    # ----------------------------------------------------
    with tab_history:
        st.subheader("Current Session Prediction History")
        st.markdown(
            "Review match scenarios evaluated during your current session. "
            "Session history is held entirely in memory and is cleared upon tab closure or reset."
        )

        hist = st.session_state.prediction_history
        if hist:
            col_h1, col_h2 = st.columns([4, 1])
            with col_h2:
                if st.button("🗑️ Clear History", use_container_width=True):
                    st.session_state.prediction_history = []
                    st.rerun()

            df_hist = pd.DataFrame(hist)
            st.dataframe(df_hist, use_container_width=True, hide_index=True)
            st.caption(f"Total session records: {len(hist)}")
        else:
            st.info("No predictions recorded in this session yet. Run a prediction to see history log entries.")

    # 3. Limitations & Sports Variance Disclaimer
    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>⚠️ Sports Variance Disclaimer:</strong>
            IPL T20 cricket possesses significant natural variance. Predictions are generated probabilistically 
            from historical ball-by-ball records, team win-rate baselines, venue scoring averages, and toss situational context. 
            Real-world outcomes depend heavily on unpredictable live variables including playing XI selections, injuries, 
            weather conditions, dew factors, and individual match-winning performances.
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == '__main__':
    main()
