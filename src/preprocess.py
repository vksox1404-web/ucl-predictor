import pandas as pd
import numpy as np
import os

RAW = os.path.join(os.path.dirname(__file__), "../data/raw/")
OUT = os.path.join(os.path.dirname(__file__), "../data/processed/")
os.makedirs(OUT, exist_ok=True)

# ─────────────────────────────────────────
# ROLLING GOALS HELPER
# ─────────────────────────────────────────
def compute_rolling_goals(df, window=5):
    """Compute rolling avg goals scored/conceded per team before each match."""
    df = df.sort_values('date').reset_index(drop=True)
    team_history = {}

    home_scored, home_conceded = [], []
    away_scored, away_conceded = [], []

    for _, row in df.iterrows():
        ht, at = row['home_team'], row['away_team']
        hg, ag = row['home_goals'], row['away_goals']

        h_hist = team_history.get(ht, [])
        a_hist = team_history.get(at, [])

        h_sc = round(np.mean([x[0] for x in h_hist[-window:]]), 3) if h_hist else 1.5
        h_co = round(np.mean([x[1] for x in h_hist[-window:]]), 3) if h_hist else 1.5
        a_sc = round(np.mean([x[0] for x in a_hist[-window:]]), 3) if a_hist else 1.5
        a_co = round(np.mean([x[1] for x in a_hist[-window:]]), 3) if a_hist else 1.5

        home_scored.append(h_sc)
        home_conceded.append(h_co)
        away_scored.append(a_sc)
        away_conceded.append(a_co)

        if ht not in team_history: team_history[ht] = []
        if at not in team_history: team_history[at] = []
        team_history[ht].append((hg, ag))
        team_history[at].append((ag, hg))

    df['home_scored_avg']   = home_scored
    df['home_conceded_avg'] = home_conceded
    df['away_scored_avg']   = away_scored
    df['away_conceded_avg'] = away_conceded
    return df

# ─────────────────────────────────────────
# PART 1: Domestic leagues (Matches.csv)
# ─────────────────────────────────────────
print("Processing Matches.csv...")

df = pd.read_csv(RAW + "Matches.csv", low_memory=False)

EU_LEAGUES = ['E0', 'SP1', 'D1', 'I1', 'F1', 'N1', 'P1', 'B1', 'G1', 'SC0']
df = df[df['Division'].isin(EU_LEAGUES)].copy()
print(f"  After EU filter: {len(df)} rows")

df = df[['Division', 'MatchDate', 'HomeTeam', 'AwayTeam',
         'HomeElo', 'AwayElo', 'Form3Home', 'Form3Away',
         'FTHome', 'FTAway', 'FTResult']].copy()

df.dropna(subset=['HomeElo', 'AwayElo', 'FTResult'], inplace=True)
df['Form3Home'] = df['Form3Home'].fillna(0)
df['Form3Away'] = df['Form3Away'].fillna(0)
df['EloDiff'] = df['HomeElo'] - df['AwayElo']

df['MatchDate'] = pd.to_datetime(df['MatchDate'], dayfirst=False, errors='coerce')
df = df[df['MatchDate'] >= '2015-01-01']
print(f"  After 2015+ filter: {len(df)} rows")

df.rename(columns={
    'MatchDate': 'date', 'HomeTeam': 'home_team', 'AwayTeam': 'away_team',
    'FTHome': 'home_goals', 'FTAway': 'away_goals', 'FTResult': 'result'
}, inplace=True)

df['total_goals'] = df['home_goals'] + df['away_goals']
df['over25'] = (df['total_goals'] > 2.5).astype(int)
df['btts']   = ((df['home_goals'] > 0) & (df['away_goals'] > 0)).astype(int)

# Compute rolling goal features
print("  Computing rolling goal averages...")
df = compute_rolling_goals(df)

df['source'] = 'domestic'
print(f"  Final rows: {len(df)}")
print(f"  Result distribution:\n{df['result'].value_counts()}")

# ─────────────────────────────────────────
# PART 2: UCL Live Data
# ─────────────────────────────────────────
print("\nProcessing ucl_matches.csv...")

ucl = pd.read_csv(RAW + "ucl_matches.csv")
ucl['date'] = pd.to_datetime(ucl['date'])

ucl['result'] = ucl.apply(
    lambda r: 'H' if r['home_goals'] > r['away_goals']
    else ('A' if r['away_goals'] > r['home_goals'] else 'D'), axis=1
)
ucl['total_goals'] = ucl['home_goals'] + ucl['away_goals']
ucl['over25'] = (ucl['total_goals'] > 2.5).astype(int)
ucl['btts']   = ((ucl['home_goals'] > 0) & (ucl['away_goals'] > 0)).astype(int)

ucl = compute_rolling_goals(ucl)

# Elo not available in UCL data — will use team_stats lookup at prediction time
ucl['HomeElo']   = np.nan
ucl['AwayElo']   = np.nan
ucl['EloDiff']   = np.nan
ucl['Form3Home'] = np.nan
ucl['Form3Away'] = np.nan
ucl['source']    = 'ucl_live'

print(f"  UCL rows: {len(ucl)}")

# ─────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────
COLS = ['date', 'home_team', 'away_team', 'home_goals', 'away_goals',
        'HomeElo', 'AwayElo', 'EloDiff', 'Form3Home', 'Form3Away',
        'home_scored_avg', 'home_conceded_avg', 'away_scored_avg', 'away_conceded_avg',
        'total_goals', 'result', 'over25', 'btts', 'source']

df[COLS].to_csv(OUT + "domestic_processed.csv", index=False)
ucl[COLS].to_csv(OUT + "ucl_processed.csv", index=False)

# Save latest team Elo + form for Streamlit lookup
team_stats = df.sort_values('date').groupby('home_team').last()[['HomeElo', 'Form3Home']].reset_index()
team_stats.columns = ['team', 'elo', 'form3']
away_stats = df.sort_values('date').groupby('away_team').last()[['AwayElo', 'Form3Away']].reset_index()
away_stats.columns = ['team', 'elo', 'form3']
all_stats = pd.concat([team_stats, away_stats]).groupby('team').mean().reset_index()
all_stats.to_csv(OUT + "team_stats.csv", index=False)

print(f"\nSaved domestic_processed.csv → {len(df)} rows")
print(f"Saved ucl_processed.csv     → {len(ucl)} rows")
print(f"Saved team_stats.csv        → {len(all_stats)} teams")
print("\n✅ Preprocessing complete!")
