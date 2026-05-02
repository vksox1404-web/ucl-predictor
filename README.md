# UCL Match Predictor

An end-to-end machine learning project that predicts UEFA Champions League match outcomes using live data from the football-data.org API and a Random Forest classifier trained on 31,735 European league matches.

**Live predictions for three markets:**
- Match Result — Win / Draw / Loss
- Over / Under 2.5 Goals
- Both Teams to Score (BTTS)

---

## Demo

> Deploy link goes here after Streamlit Cloud setup

---

## Features

- Live upcoming UCL fixtures pulled from the football-data.org API, updated every hour
- AI predictions with win probabilities, gauge charts, and team comparison radar
- Club crests for all 36 current UCL teams sourced from the official API
- Recent match results shown automatically when no fixtures are scheduled
- Team profiles with UCL record, Elo rating, form, and goal averages
- Model performance page showing accuracy, cross-validation scores, and feature importances
- Dark premium UI with top navigation bar — no sidebar

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.11+ |
| ML | scikit-learn — Random Forest Classifier |
| Data | pandas, NumPy |
| API | football-data.org (free tier) |
| App | Streamlit, Plotly, streamlit-option-menu |
| Serialization | joblib |
| Config | python-dotenv |

---

## Project Structure

```
ucl-predictor/
│
├── app.py                   # Streamlit app (4 pages)
│
├── src/
│   ├── fetch_data.py        # Pull UCL match results from API
│   ├── fetch_teams.py       # Pull current UCL team metadata + crests
│   ├── preprocess.py        # Feature engineering pipeline
│   └── train.py             # Train and save 3 Random Forest models
│
├── data/
│   ├── raw/
│   │   ├── Matches.csv          # 230k domestic league matches (Kaggle)
│   │   └── ucl_matches.csv      # Live UCL results (API)
│   └── processed/
│       ├── domestic_processed.csv
│       ├── ucl_processed.csv
│       ├── team_stats.csv
│       └── ucl_teams.json
│
├── models/
│   ├── model_result.pkl     # W/D/L classifier
│   ├── model_over25.pkl     # Over/Under 2.5 classifier
│   ├── model_btts.pkl       # BTTS classifier
│   └── features.pkl         # Feature list
│
├── assets/
│   └── ucl.png
│
├── .env                     # API key (not committed)
├── requirements.txt
└── README.md
```

---

## Data Pipeline

**Training data (31,735 matches):**
Domestic European league matches from the Premier League, La Liga, Bundesliga, Serie A, Ligue 1, and 5 other top leagues. Sourced from a Kaggle dataset that includes pre-computed Elo ratings and form scores. Filtered to 2015–present.

**Live data (500 UCL matches):**
Fetched directly from the football-data.org API covering the 2023-24, 2024-25, and 2025-26 Champions League seasons. Updated anytime by re-running `src/fetch_data.py`.

**Features engineered per match:**

| Feature | Description |
|---|---|
| EloDiff | Home Elo minus Away Elo — captures relative team strength |
| Form3Home | Home team's form points over last 3 matches |
| Form3Away | Away team's form points over last 3 matches |
| home_scored_avg | Rolling 5-match avg goals scored by home team |
| home_conceded_avg | Rolling 5-match avg goals conceded by home team |
| away_scored_avg | Rolling 5-match avg goals scored by away team |
| away_conceded_avg | Rolling 5-match avg goals conceded by away team |

---

## Model Performance

| Model | Test Accuracy | CV Accuracy | Baseline |
|---|---|---|---|
| Match Result (W/D/L) | 49.3% | 49.0% | 33.3% |
| Over/Under 2.5 Goals | 53.8% | 55.4% | 50.0% |
| Both Teams to Score | 53.5% | 52.9% | 50.0% |

All three models use `RandomForestClassifier` with `class_weight='balanced'` to handle the class imbalance in draw predictions, and are evaluated with 5-fold stratified cross-validation.

**Most important feature:** EloDiff accounts for 58.9% of feature importance in the W/D/L model — confirming that relative team strength is the dominant predictor of match outcomes in football.

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/your-username/ucl-predictor.git
cd ucl-predictor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your API key

Create a `.env` file in the root:

```
FOOTBALL_API_KEY=your_key_here
```

Get a free key at [football-data.org](https://www.football-data.org/client/register).

### 4. Run the data pipeline

```bash
# Fetch live UCL match data
python src/fetch_data.py

# Fetch current UCL team metadata + crests
python src/fetch_teams.py

# Preprocess and engineer features
python src/preprocess.py

# Train and save the 3 models
python src/train.py
```

### 5. Launch the app

```bash
streamlit run app.py
```

---

## Limitations

This project is built as a portfolio demonstration of an end-to-end ML pipeline, not a production betting model.

- The models are trained on domestic league data and applied to UCL — different competition dynamics affect accuracy
- No injury, suspension, or lineup data is included
- Draw prediction recall is ~35% — draws remain the hardest outcome to predict in football
- The free API tier only provides data from 2023 onwards for UCL

---

## Author

**Ali Mohamed** — CS Graduate, Qatar

[LinkedIn](https://linkedin.com) · [GitHub](https://github.com)
