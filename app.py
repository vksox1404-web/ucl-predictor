import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import requests
import os
import json
from dotenv import load_dotenv
from streamlit_option_menu import option_menu

load_dotenv()

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="UCL Predictor",
    page_icon="assets/ucl.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────
# CUSTOM CSS — Dark Premium Theme
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Poppins:wght@700;800;900&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #080c14;
    color: #e2e8f0;
}

/* Main background */
.main { background-color: #080c14; }
.block-container { padding: 1.5rem 2rem; max-width: 1400px; }

/* Hide sidebar completely */
[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }

/* Top navbar */
.nav-container {
    background: rgba(13,18,32,0.95);
    border-bottom: 1px solid rgba(59,130,246,0.15);
    padding: 0.8rem 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(10px);
    position: sticky;
    top: 0;
    z-index: 999;
}
.nav-brand {
    font-family: 'Poppins', sans-serif;
    font-weight: 800;
    font-size: 1.2rem;
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Option menu styling */
nav[data-testid="stHorizontalBlock"] { background: transparent !important; }
div[data-testid="stHorizontalBlock"] > div { background: transparent !important; }

/* Cards */
.match-card {
    background: linear-gradient(135deg, rgba(20,28,48,0.9) 0%, rgba(13,18,32,0.9) 100%);
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: 16px;
    padding: 1.4rem;
    margin-bottom: 1rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
}
.match-card:hover { border-color: rgba(59,130,246,0.5); box-shadow: 0 8px 32px rgba(59,130,246,0.15); }

.stat-card {
    background: linear-gradient(135deg, rgba(20,28,48,0.9) 0%, rgba(13,18,32,0.9) 100%);
    border: 1px solid rgba(59,130,246,0.15);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
}

.prediction-card {
    background: linear-gradient(135deg, rgba(59,130,246,0.1) 0%, rgba(139,92,246,0.1) 100%);
    border: 1px solid rgba(59,130,246,0.3);
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 8px 32px rgba(59,130,246,0.1);
}

/* Typography */
.page-title {
    font-family: 'Poppins', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 1rem;
}
.team-name {
    font-family: 'Poppins', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #f1f5f9;
}
.big-team {
    font-family: 'Poppins', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    color: #f1f5f9;
}
.vs-text {
    font-size: 1.2rem;
    font-weight: 700;
    color: #3b82f6;
}
.badge {
    display: inline-block;
    padding: 0.2rem 0.8rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}
.badge-blue  { background: rgba(59,130,246,0.2);  color: #93c5fd; border: 1px solid rgba(59,130,246,0.3); }
.badge-green { background: rgba(16,185,129,0.2);  color: #6ee7b7; border: 1px solid rgba(16,185,129,0.3); }
.badge-purple{ background: rgba(139,92,246,0.2);  color: #c4b5fd; border: 1px solid rgba(139,92,246,0.3); }

.metric-value { font-size: 2rem; font-weight: 800; color: #f1f5f9; }
.metric-label { font-size: 0.8rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; }

/* Divider */
.divider { border: none; border-top: 1px solid rgba(59,130,246,0.1); margin: 1.5rem 0; }

/* Streamlit overrides */
div[data-testid="stSelectbox"] > div { background: rgba(20,28,48,0.9) !important; border-color: rgba(59,130,246,0.3) !important; border-radius: 10px !important; }
div[data-testid="stMetric"] { background: rgba(20,28,48,0.6); border-radius: 12px; padding: 1rem; border: 1px solid rgba(59,130,246,0.1); }
div[data-testid="stMetric"] label { color: #64748b !important; }
div[data-testid="stMetric"] div { color: #f1f5f9 !important; }

.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.6rem 2rem !important;
    font-size: 1rem !important;
    transition: all 0.3s !important;
    width: 100%;
}
.stButton > button:hover { opacity: 0.9; transform: translateY(-1px); }

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# LOAD MODELS & DATA
# ─────────────────────────────────────────
BASE = os.path.dirname(__file__)

@st.cache_resource
def load_models():
    models_dir = os.path.join(BASE, "models")
    return {
        "result": joblib.load(f"{models_dir}/model_result.pkl"),
        "over25": joblib.load(f"{models_dir}/model_over25.pkl"),
        "btts":   joblib.load(f"{models_dir}/model_btts.pkl"),
        "features": joblib.load(f"{models_dir}/features.pkl"),
    }

@st.cache_data
def load_data():
    proc = os.path.join(BASE, "data/processed")
    team_stats = pd.read_csv(f"{proc}/team_stats.csv")
    ucl        = pd.read_csv(f"{proc}/ucl_processed.csv")
    domestic   = pd.read_csv(f"{proc}/domestic_processed.csv")
    with open(f"{proc}/ucl_teams.json") as f:
        teams_info = json.load(f)
    return team_stats, ucl, domestic, teams_info

@st.cache_data(ttl=3600)
def get_upcoming_matches():
    from datetime import datetime, timezone
    api_key = os.getenv("FOOTBALL_API_KEY")
    try:
        r = requests.get(
            "https://api.football-data.org/v4/competitions/CL/matches",
            headers={"X-Auth-Token": api_key},
            params={"season": 2025},
            timeout=5
        )
        if r.status_code == 200:
            now = datetime.now(timezone.utc)
            matches = r.json().get("matches", [])
            result = []
            for m in matches:
                if m.get("status") == "FINISHED":
                    continue
                match_date = datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00"))
                if match_date <= now:
                    continue
                home = m["homeTeam"].get("name")
                away = m["awayTeam"].get("name")
                if home and away:
                    result.append({
                        "home": home,
                        "away": away,
                        "date": m["utcDate"][:10],
                        "stage": m["stage"].replace("_", " ").title()
                    })
                if len(result) >= 8:
                    break
            return result
    except: pass
    return []

@st.cache_data(ttl=86400)
def get_current_ucl_teams():
    """Fetch the actual current season UCL teams from the API."""
    api_key = os.getenv("FOOTBALL_API_KEY")
    try:
        r = requests.get(
            "https://api.football-data.org/v4/competitions/CL/teams",
            headers={"X-Auth-Token": api_key},
            params={"season": 2025},
            timeout=5
        )
        if r.status_code == 200:
            teams = r.json().get("teams", [])
            return sorted([t["name"] for t in teams if t.get("name")])
    except:
        pass
    # Fallback: use teams from current season in our data
    ucl = pd.read_csv(os.path.join(BASE, "data/processed/ucl_processed.csv"))
    current = ucl[ucl['season'] == '2025-26'] if 'season' in ucl.columns else ucl
    if current.empty:
        current = ucl
    return sorted(set(current['home_team'].dropna().tolist() + current['away_team'].dropna().tolist()))

models     = load_models()
team_stats, ucl_df, domestic_df, TEAMS_INFO = load_data()
UCL_TEAMS  = sorted(TEAMS_INFO.keys())

def find_ucl_team(name):
    """Find the closest matching name in UCL_TEAMS."""
    if not name:
        return UCL_TEAMS[0]
    if name in UCL_TEAMS:
        return name
    first = name.split()[0].lower()
    for t in UCL_TEAMS:
        if first in t.lower():
            return t
    return UCL_TEAMS[0]

# Selectbox defaults (overridden when navigating from a match card)
if 'home_select' not in st.session_state:
    st.session_state['home_select'] = "Real Madrid CF" if "Real Madrid CF" in UCL_TEAMS else UCL_TEAMS[0]
if 'away_select' not in st.session_state:
    st.session_state['away_select'] = "FC Bayern München" if "FC Bayern München" in UCL_TEAMS else UCL_TEAMS[1]

def get_crest(team_name, size=40):
    """Return an HTML img tag with the team crest."""
    info = TEAMS_INFO.get(team_name, {})
    url  = info.get("crest", "")
    if url:
        return f"<img src='{url}' style='width:{size}px;height:{size}px;object-fit:contain;vertical-align:middle;border-radius:4px;' />"
    return f"<div style='width:{size}px;height:{size}px;background:rgba(59,130,246,0.2);border-radius:50%;display:inline-block;'></div>"

def get_crest_url(team_name):
    return TEAMS_INFO.get(team_name, {}).get("crest", "")

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def get_team_stats(team_name):
    """Fuzzy match team name to get Elo and form."""
    if not team_name or not isinstance(team_name, str):
        return team_stats['elo'].mean(), team_stats['form3'].mean()
    ts = team_stats.copy()
    # Exact match first
    exact = ts[ts['team'] == team_name]
    if not exact.empty:
        return exact.iloc[0]['elo'], exact.iloc[0]['form3']
    # Partial match using first word
    first_word = team_name.split()[0] if team_name.split() else team_name
    partial = ts[ts['team'].str.contains(first_word, case=False, na=False)]
    if not partial.empty:
        return partial.iloc[0]['elo'], partial.iloc[0]['form3']
    # Default (league average)
    return ts['elo'].mean(), ts['form3'].mean()

def get_team_goals(team_name):
    """Get rolling avg goals scored/conceded from UCL data."""
    team_matches = ucl_df[(ucl_df['home_team'] == team_name) | (ucl_df['away_team'] == team_name)]
    if team_matches.empty:
        return 1.5, 1.5
    last = team_matches.iloc[-1]
    if last['home_team'] == team_name:
        return last['home_scored_avg'], last['home_conceded_avg']
    return last['away_scored_avg'], last['away_conceded_avg']

def predict_match(home_team, away_team):
    home_elo, home_form = get_team_stats(home_team)
    away_elo, away_form = get_team_stats(away_team)
    home_scored, home_conceded = get_team_goals(home_team)
    away_scored, away_conceded = get_team_goals(away_team)

    elo_diff = home_elo - away_elo
    X = pd.DataFrame([[elo_diff, home_form, away_form,
                        home_scored, home_conceded,
                        away_scored, away_conceded]],
                     columns=models["features"])

    result_proba = models["result"].predict_proba(X)[0]
    result_classes = models["result"].classes_
    result_dict = dict(zip(result_classes, result_proba))

    over25_proba = models["over25"].predict_proba(X)[0][1]
    btts_proba   = models["btts"].predict_proba(X)[0][1]

    return {
        "home_win": result_dict.get("H", 0),
        "draw":     result_dict.get("D", 0),
        "away_win": result_dict.get("A", 0),
        "over25":   over25_proba,
        "btts":     btts_proba,
        "home_elo": home_elo,
        "away_elo": away_elo,
        "home_form": home_form,
        "away_form": away_form,
        "home_scored": home_scored,
        "away_scored": away_scored,
    }

def result_bar_chart(home, away, hw, d, aw):
    fig = go.Figure()
    categories = [f"🏠 {home}", "Draw", f"✈ {away}"]
    values = [hw*100, d*100, aw*100]
    colors = ["#3b82f6", "#64748b", "#8b5cf6"]
    for i, (cat, val, col) in enumerate(zip(categories, values, colors)):
        fig.add_trace(go.Bar(
            x=[val], y=[cat], orientation='h',
            marker=dict(color=col, line=dict(width=0)),
            text=f"{val:.1f}%", textposition='inside',
            textfont=dict(size=14, color='white', family='Inter'),
            name=cat, showlegend=False
        ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=200,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(range=[0,100], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color='#94a3b8', size=13)),
        barmode='overlay',
        bargap=0.4
    )
    return fig

def gauge_chart(value, title, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        number={'suffix': '%', 'font': {'color': color, 'size': 28, 'family': 'Inter'}},
        title={'text': title, 'font': {'color': '#94a3b8', 'size': 13}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#334155', 'tickwidth': 1},
            'bar': {'color': color, 'thickness': 0.25},
            'bgcolor': 'rgba(30,41,59,0.5)',
            'bordercolor': 'rgba(255,255,255,0.05)',
            'steps': [{'range': [0, 50], 'color': 'rgba(30,41,59,0.3)'},
                      {'range': [50, 100], 'color': 'rgba(30,41,59,0.1)'}],
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        height=180,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def team_radar(home, away, stats):
    categories = ['Elo Rating', 'Form', 'Goals Scored', 'Goals Conceded (inv)']
    max_elo = team_stats['elo'].max()

    def normalize(val, mn, mx): return (val - mn) / (mx - mn + 1e-9)

    home_vals = [
        normalize(stats['home_elo'], 1200, max_elo),
        normalize(stats['home_form'], 0, 9),
        normalize(stats['home_scored'], 0, 4),
        1 - normalize(stats['away_scored'], 0, 4),
    ]
    away_vals = [
        normalize(stats['away_elo'], 1200, max_elo),
        normalize(stats['away_form'], 0, 9),
        normalize(stats['away_scored'], 0, 4),
        1 - normalize(stats['home_scored'], 0, 4),
    ]

    fig = go.Figure()
    fill_colors = ['rgba(59,130,246,0.15)', 'rgba(139,92,246,0.15)']
    for (name, vals, color), fill in zip([(home, home_vals, '#3b82f6'), (away, away_vals, '#8b5cf6')], fill_colors):
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=categories + [categories[0]],
            fill='toself', name=name,
            line=dict(color=color, width=2),
            fillcolor=fill
        ))
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, range=[0,1], showticklabels=False, gridcolor='rgba(255,255,255,0.05)'),
            angularaxis=dict(tickfont=dict(color='#94a3b8', size=11), gridcolor='rgba(255,255,255,0.05)')
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(font=dict(color='#94a3b8'), bgcolor='rgba(0,0,0,0)'),
        height=320,
        margin=dict(l=40, r=40, t=20, b=20)
    )
    return fig

# ─────────────────────────────────────────
# TOP NAVBAR
# ─────────────────────────────────────────
st.markdown("""
<div class='nav-container'>
    <div class='nav-brand'>UCL Predictor &nbsp;<span style='-webkit-text-fill-color:#475569;font-size:0.75rem;font-weight:400'>AI-Powered Analytics</span></div>
    <div style='color:#475569;font-size:0.75rem'>Powered by football-data.org &nbsp;·&nbsp; Random Forest v1.0 &nbsp;·&nbsp; 31,735 matches</div>
</div>
""", unsafe_allow_html=True)

# Handle navigation from match card buttons
if st.session_state.get('_nav_to'):
    st.session_state['main_nav'] = st.session_state.pop('_nav_to')

page = option_menu(
    menu_title=None,
    options=["Dashboard", "Predictions", "Stats", "Teams"],
    icons=["grid-3x3-gap", "lightning-charge", "bar-chart-line", "shield"],
    orientation="horizontal",
    key="main_nav",
    styles={
        "container": {
            "padding": "0",
            "background-color": "rgba(13,18,32,0.6)",
            "border": "1px solid rgba(59,130,246,0.15)",
            "border-radius": "12px",
            "margin-bottom": "1.5rem",
        },
        "icon": {"color": "#64748b", "font-size": "16px"},
        "nav-link": {
            "font-family": "Inter, sans-serif",
            "font-size": "0.9rem",
            "color": "#94a3b8",
            "padding": "0.75rem 1.5rem",
            "border-radius": "10px",
        },
        "nav-link-selected": {
            "background": "linear-gradient(135deg, rgba(59,130,246,0.2), rgba(139,92,246,0.2))",
            "color": "#f1f5f9",
            "font-weight": "600",
            "border": "1px solid rgba(59,130,246,0.3)",
        },
    }
)

# ─────────────────────────────────────────
# PAGE: DASHBOARD
# ─────────────────────────────────────────
if page == "Dashboard":
    st.markdown("<div class='page-title'>Champions League Hub</div>", unsafe_allow_html=True)
    st.markdown("<div style='color:#64748b;margin-bottom:1.5rem'>Live predictions — updated automatically</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Upcoming Matches</div>", unsafe_allow_html=True)

    upcoming = get_upcoming_matches()
    st.caption(f"DEBUG — API returned {len(upcoming)} upcoming matches")
    if upcoming:
        for i in range(0, len(upcoming), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                if i + j < len(upcoming):
                    m = upcoming[i + j]
                    pred = predict_match(m['home'], m['away'])
                    winner = m['home'] if pred['home_win'] > pred['away_win'] and pred['home_win'] > pred['draw'] \
                             else (m['away'] if pred['away_win'] > pred['draw'] else "Draw")
                    conf   = max(pred['home_win'], pred['draw'], pred['away_win'])
                    with col:
                        st.markdown(f"""
                        <div class='match-card'>
                            <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:0.8rem'>
                                <span class='badge badge-blue'>{m['stage']}</span>
                                <span style='color:#475569;font-size:0.8rem'>{m['date']}</span>
                            </div>
                            <div style='display:flex;justify-content:space-between;align-items:center;margin:0.8rem 0'>
                                <div style='text-align:center;flex:1'>
                                    {get_crest(m['home'], 48)}
                                    <div class='team-name' style='margin-top:0.4rem'>{m['home'][:20]}</div>
                                    <div style='color:#3b82f6;font-weight:700;font-size:1.1rem;margin-top:0.2rem'>{pred['home_win']*100:.0f}%</div>
                                    <div style='color:#64748b;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em'>Home</div>
                                </div>
                                <div style='text-align:center;padding:0 1rem'>
                                    <div class='vs-text'>VS</div>
                                    <div style='color:#475569;font-size:0.75rem'>Draw {pred['draw']*100:.0f}%</div>
                                </div>
                                <div style='text-align:center;flex:1'>
                                    {get_crest(m['away'], 48)}
                                    <div class='team-name' style='margin-top:0.4rem'>{m['away'][:20]}</div>
                                    <div style='color:#8b5cf6;font-weight:700;font-size:1.1rem;margin-top:0.2rem'>{pred['away_win']*100:.0f}%</div>
                                    <div style='color:#64748b;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em'>Away</div>
                                </div>
                            </div>
                            <div style='border-top:1px solid rgba(255,255,255,0.05);padding-top:0.8rem;display:flex;justify-content:space-between'>
                                <span style='color:#10b981;font-size:0.85rem;font-weight:600'>Predicted: {winner[:20]}</span>
                                <span class='badge badge-green'>{conf*100:.0f}% confidence</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        if st.button("View Full Prediction", key=f"up_{i}_{j}", use_container_width=True):
                            st.session_state['_nav_to'] = 'Predictions'
                            st.session_state['home_select'] = find_ucl_team(m['home'])
                            st.session_state['away_select'] = find_ucl_team(m['away'])
                            st.rerun()
    else:
        st.markdown("<div style='color:#64748b;font-size:0.85rem;margin-bottom:1rem'>No scheduled matches at the moment — showing the most recent UCL results.</div>", unsafe_allow_html=True)
        recent_matches = ucl_df.dropna(subset=['home_team','away_team','home_goals','away_goals']).sort_values('date').tail(8)
        for i in range(0, len(recent_matches), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(recent_matches):
                    m = recent_matches.iloc[idx]
                    hg, ag = int(m['home_goals']), int(m['away_goals'])
                    if hg > ag:
                        winner_label = "Home Win"
                        badge_class  = "badge-blue"
                    elif ag > hg:
                        winner_label = "Away Win"
                        badge_class  = "badge-purple"
                    else:
                        winner_label = "Draw"
                        badge_class  = "badge-green"
                    with col:
                        st.markdown(f"""
                        <div class='match-card'>
                            <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:0.8rem'>
                                <span class='badge badge-blue'>Final Result</span>
                                <span style='color:#475569;font-size:0.8rem'>{m['date']}</span>
                            </div>
                            <div style='display:flex;justify-content:space-between;align-items:center;margin:0.8rem 0'>
                                <div style='text-align:center;flex:1'>
                                    {get_crest(m['home_team'], 48)}
                                    <div class='team-name' style='margin-top:0.4rem'>{str(m['home_team'])[:20]}</div>
                                </div>
                                <div style='text-align:center;padding:0 1rem'>
                                    <div style='font-size:2rem;font-weight:800;color:#f1f5f9'>{hg} – {ag}</div>
                                </div>
                                <div style='text-align:center;flex:1'>
                                    {get_crest(m['away_team'], 48)}
                                    <div class='team-name' style='margin-top:0.4rem'>{str(m['away_team'])[:20]}</div>
                                </div>
                            </div>
                            <div style='border-top:1px solid rgba(255,255,255,0.05);padding-top:0.8rem;text-align:center'>
                                <span class='badge {badge_class}'>{winner_label}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        if st.button("View Prediction", key=f"rec_{idx}", use_container_width=True):
                            st.session_state['_nav_to'] = 'Predictions'
                            st.session_state['home_select'] = find_ucl_team(str(m['home_team']))
                            st.session_state['away_select'] = find_ucl_team(str(m['away_team']))
                            st.rerun()

# ─────────────────────────────────────────
# PAGE: PREDICTIONS
# ─────────────────────────────────────────
elif page == "Predictions":
    st.markdown("<div class='page-title'>Match Predictor</div>", unsafe_allow_html=True)
    st.markdown("<div style='color:#64748b;margin-bottom:1.5rem'>Select two teams to generate AI predictions</div>", unsafe_allow_html=True)

    col1, col_vs, col2 = st.columns([5, 1, 5])
    with col1:
        home = st.selectbox("Home Team", UCL_TEAMS, key='home_select')
    with col_vs:
        st.markdown("<div style='text-align:center;padding-top:2.2rem;font-weight:800;color:#3b82f6;font-size:1.3rem'>VS</div>", unsafe_allow_html=True)
    with col2:
        away = st.selectbox("Away Team", UCL_TEAMS, key='away_select')

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    predict_btn = st.button("Generate Prediction")

    if predict_btn or True:
        if home == away:
            st.warning("Please select two different teams.")
        else:
            pred = predict_match(home, away)

            st.markdown("<hr class='divider'>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class='prediction-card'>
                <div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:1rem'>
                    <div style='text-align:center;flex:1'>
                        <div style='font-size:0.7rem;color:#3b82f6;font-weight:600;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:0.8rem'>Home</div>
                        {get_crest(home, 72)}
                        <div class='big-team' style='margin-top:0.6rem'>{home}</div>
                        <div style='color:#3b82f6;font-size:2rem;font-weight:800;margin-top:0.4rem'>{pred['home_win']*100:.1f}%</div>
                        <div style='color:#64748b;font-size:0.85rem'>Win probability</div>
                    </div>
                    <div style='text-align:center;padding:0 1.5rem'>
                        <div style='font-size:1rem;color:#64748b;margin-bottom:0.3rem'>Draw</div>
                        <div style='font-size:1.8rem;font-weight:800;color:#f1f5f9'>{pred['draw']*100:.1f}%</div>
                    </div>
                    <div style='text-align:center;flex:1'>
                        <div style='font-size:0.7rem;color:#8b5cf6;font-weight:600;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:0.8rem'>Away</div>
                        {get_crest(away, 72)}
                        <div class='big-team' style='margin-top:0.6rem'>{away}</div>
                        <div style='color:#8b5cf6;font-size:2rem;font-weight:800;margin-top:0.4rem'>{pred['away_win']*100:.1f}%</div>
                        <div style='color:#64748b;font-size:0.85rem'>Win probability</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            st.plotly_chart(result_bar_chart(home, away, pred['home_win'], pred['draw'], pred['away_win']),
                            use_container_width=True, config={'displayModeBar': False})

            # Over/Under & BTTS gauges
            st.markdown("<div class='section-title'>Additional Predictions</div>", unsafe_allow_html=True)
            g1, g2 = st.columns(2)
            with g1:
                st.plotly_chart(gauge_chart(pred['over25'], "Over 2.5 Goals", "#10b981"),
                                use_container_width=True, config={'displayModeBar': False})
                label_color = "#10b981" if pred['over25'] >= 0.5 else "#ef4444"
                label_text  = "Likely OVER 2.5" if pred['over25'] >= 0.5 else "Likely UNDER 2.5"
                st.markdown(f"<div style='text-align:center;color:{label_color};font-weight:600'>{label_text}</div>", unsafe_allow_html=True)
            with g2:
                st.plotly_chart(gauge_chart(pred['btts'], "Both Teams to Score", "#f59e0b"),
                                use_container_width=True, config={'displayModeBar': False})
                label_color = "#10b981" if pred['btts'] >= 0.5 else "#ef4444"
                label_text  = "Likely YES" if pred['btts'] >= 0.5 else "Likely NO"
                st.markdown(f"<div style='text-align:center;color:{label_color};font-weight:600'>{label_text}</div>", unsafe_allow_html=True)

            # Radar chart
            st.markdown("<hr class='divider'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Team Comparison</div>", unsafe_allow_html=True)
            r1, r2 = st.columns([3, 2])
            with r1:
                st.plotly_chart(team_radar(home, away, pred), use_container_width=True, config={'displayModeBar': False})
            with r2:
                st.markdown(f"""
                <div class='stat-card' style='margin-bottom:0.8rem'>
                    <div class='metric-label'>Elo Difference</div>
                    <div class='metric-value' style='color:{"#3b82f6" if pred["home_elo"]>pred["away_elo"] else "#8b5cf6"}'>
                        {pred["home_elo"]-pred["away_elo"]:+.0f}
                    </div>
                    <div style='color:#64748b;font-size:0.8rem'>{"Home advantage" if pred["home_elo"]>pred["away_elo"] else "Away advantage"}</div>
                </div>
                <div class='stat-card' style='margin-bottom:0.8rem'>
                    <div class='metric-label'>Home Elo</div>
                    <div class='metric-value'>{pred["home_elo"]:.0f}</div>
                </div>
                <div class='stat-card'>
                    <div class='metric-label'>Away Elo</div>
                    <div class='metric-value'>{pred["away_elo"]:.0f}</div>
                </div>
                """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# PAGE: STATS
# ─────────────────────────────────────────
elif page == "Stats":
    st.markdown("<div class='page-title'>Model Performance</div>", unsafe_allow_html=True)
    st.markdown("<div style='color:#64748b;margin-bottom:1.5rem'>Random Forest · Trained on 31,735 European matches</div>", unsafe_allow_html=True)

    # Dataset KPIs
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Training Matches", "31,735", "Domestic + UCL")
    with kpi2:
        st.metric("UCL Matches", f"{len(ucl_df)}", "2023–2026")
    with kpi3:
        st.metric("W/D/L Accuracy", "49.3%", "+16% vs random")
    with kpi4:
        st.metric("Teams Tracked", f"{len(UCL_TEAMS)}", "Active UCL clubs")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Accuracy cards
    c1, c2, c3 = st.columns(3)
    metrics = [
        ("Match Result (W/D/L)", "49.3%", "55.4%", "#3b82f6", "vs 33% random baseline"),
        ("Over/Under 2.5",       "53.8%", "55.4%", "#10b981", "vs 50% random baseline"),
        ("Both Teams to Score",  "53.5%", "52.9%", "#f59e0b", "vs 50% random baseline"),
    ]
    for col, (name, test, cv, color, note) in zip([c1,c2,c3], metrics):
        with col:
            st.markdown(f"""
            <div class='stat-card'>
                <div style='color:{color};font-size:0.8rem;font-weight:600;text-transform:uppercase;letter-spacing:0.1em'>{name}</div>
                <div style='font-size:2.5rem;font-weight:800;color:#f1f5f9;margin:0.5rem 0'>{test}</div>
                <div style='color:#64748b;font-size:0.8rem'>Test Accuracy</div>
                <div style='color:#94a3b8;font-size:0.85rem;margin-top:0.5rem'>CV: {cv}</div>
                <div style='color:#475569;font-size:0.75rem;margin-top:0.3rem'>{note}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Feature Importances — W/D/L Model</div>", unsafe_allow_html=True)

    features = ['EloDiff', 'away_scored_avg', 'home_scored_avg', 'away_conceded_avg', 'home_conceded_avg', 'Form3Home', 'Form3Away']
    importances = [0.589, 0.098, 0.092, 0.060, 0.059, 0.052, 0.050]
    colors_feat = ['#3b82f6','#8b5cf6','#8b5cf6','#10b981','#10b981','#f59e0b','#f59e0b']

    fig = go.Figure(go.Bar(
        x=importances, y=features, orientation='h',
        marker=dict(color=colors_feat, line=dict(width=0)),
        text=[f"{v*100:.1f}%" for v in importances],
        textposition='outside', textfont=dict(color='#94a3b8', size=12)
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=300, margin=dict(l=10,r=60,t=10,b=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(tickfont=dict(color='#94a3b8', size=13), gridcolor='rgba(255,255,255,0.03)')
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


# ─────────────────────────────────────────
# PAGE: TEAMS
# ─────────────────────────────────────────
elif page == "Teams":
    st.markdown("<div class='page-title'>Team Profiles</div>", unsafe_allow_html=True)
    st.markdown("<div style='color:#64748b;margin-bottom:1.5rem'>Browse UCL team statistics</div>", unsafe_allow_html=True)

    selected = st.selectbox("Select a team", UCL_TEAMS)

    # Team header with logo
    crest_url = get_crest_url(selected)
    country   = TEAMS_INFO.get(selected, {}).get("country", "")
    st.markdown(f"""
    <div class='match-card' style='display:flex;align-items:center;gap:1.5rem;margin-bottom:1.5rem'>
        {'<img src="' + crest_url + '" style="width:80px;height:80px;object-fit:contain;" />' if crest_url else ''}
        <div>
            <div style='font-family:Poppins;font-size:1.8rem;font-weight:800;color:#f1f5f9'>{selected}</div>
            <div style='color:#64748b;font-size:0.9rem;margin-top:0.2rem'>{country} &nbsp;·&nbsp; UEFA Champions League 2025/26</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    elo, form = get_team_stats(selected)
    scored, conceded = get_team_goals(selected)

    team_matches = ucl_df[(ucl_df['home_team'] == selected) | (ucl_df['away_team'] == selected)].copy()
    team_matches['result_label'] = team_matches.apply(
        lambda r: ('Win' if (r['home_team']==selected and r['home_goals']>r['away_goals']) or
                           (r['away_team']==selected and r['away_goals']>r['home_goals'])
                   else ('Loss' if (r['home_team']==selected and r['home_goals']<r['away_goals']) or
                                  (r['away_team']==selected and r['away_goals']<r['home_goals'])
                   else 'Draw')), axis=1
    )

    wins   = (team_matches['result_label'] == 'Win').sum()
    draws  = (team_matches['result_label'] == 'Draw').sum()
    losses = (team_matches['result_label'] == 'Loss').sum()
    total  = len(team_matches)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    for col, label, val, color in [
        (c1, "Elo Rating",     f"{elo:.0f}",          "#3b82f6"),
        (c2, "Recent Form",    f"{form:.0f}/9",        "#10b981"),
        (c3, "Avg Goals",      f"{scored:.2f}",        "#f59e0b"),
        (c4, "UCL Matches",    str(total),             "#8b5cf6"),
    ]:
        with col:
            st.markdown(f"""
            <div class='stat-card'>
                <div class='metric-label'>{label}</div>
                <div class='metric-value' style='color:{color}'>{val}</div>
            </div>
            """, unsafe_allow_html=True)

    if total > 0:
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>UCL Record</div>", unsafe_allow_html=True)
        fig = go.Figure(go.Bar(
            x=['Wins', 'Draws', 'Losses'],
            y=[wins, draws, losses],
            marker=dict(color=['#3b82f6','#64748b','#8b5cf6'], line=dict(width=0)),
            text=[wins, draws, losses], textposition='outside',
            textfont=dict(color='#94a3b8', size=14)
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=250, margin=dict(l=10,r=10,t=20,b=10),
            xaxis=dict(tickfont=dict(color='#94a3b8'), gridcolor='rgba(0,0,0,0)'),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        st.markdown("<div class='section-title'>Recent UCL Matches</div>", unsafe_allow_html=True)
        recent = team_matches.tail(5)[['date','home_team','away_team','home_goals','away_goals','result_label']].copy()
        recent['date'] = pd.to_datetime(recent['date']).dt.strftime('%d %b %Y')
        recent.columns = ['Date','Home','Away','HG','AG','Result']
        st.dataframe(recent, use_container_width=True, hide_index=True)
