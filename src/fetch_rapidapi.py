import requests
import pandas as pd
import time
import os

# Paste your RapidAPI key here
RAPIDAPI_KEY = "YOUR_RAPIDAPI_KEY_HERE"

HEADERS = {
    "X-RapidAPI-Key": RAPIDAPI_KEY,
    "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
}

BASE_URL = "https://api-football-v1.p.rapidapi.com/v3"
UCL_LEAGUE_ID = 2  # Champions League ID in API-Football

# Seasons to fetch (year = start of season)
SEASONS = [2018, 2019, 2020, 2021, 2022, 2023, 2024]

def get_fixtures(season_year):
    url = f"{BASE_URL}/fixtures"
    params = {"league": UCL_LEAGUE_ID, "season": season_year}
    r = requests.get(url, headers=HEADERS, params=params)
    print(f"{season_year}-{str(season_year+1)[-2:]} → Status: {r.status_code}", end="")
    if r.status_code != 200:
        print(f" | Error: {r.text[:100]}")
        return []
    data = r.json().get("response", [])
    finished = [f for f in data if f["fixture"]["status"]["short"] == "FT"]
    print(f" | Total: {len(data)} | Finished: {len(finished)}")
    return finished

def parse_fixtures(fixtures, season_year):
    rows = []
    for f in fixtures:
        rows.append({
            "date": f["fixture"]["date"][:10],
            "season": f"{season_year}-{str(season_year+1)[-2:]}",
            "home_team": f["teams"]["home"]["name"],
            "away_team": f["teams"]["away"]["name"],
            "home_goals": f["goals"]["home"],
            "away_goals": f["goals"]["away"],
            "stage": f["league"]["round"],
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    all_dfs = []

    for year in SEASONS:
        fixtures = get_fixtures(year)
        if fixtures:
            df = parse_fixtures(fixtures, year)
            all_dfs.append(df)
        time.sleep(2)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        out = os.path.join(os.path.dirname(__file__), "../data/raw/ucl_matches_full.csv")
        combined.to_csv(out, index=False)
        print(f"\nTotal: {len(combined)} matches saved to data/raw/ucl_matches_full.csv")
        print(combined.groupby("season").size())
    else:
        print("\nNo data fetched.")
