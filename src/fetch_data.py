import requests
import pandas as pd
import os
import time
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("FOOTBALL_API_KEY")
HEADERS = {"X-Auth-Token": API_KEY}
BASE_URL = "https://api.football-data.org/v4"

# Seasons accessible on free tier (2023 onwards)
# Run this script anytime to refresh with latest UCL results
SEASONS = {
    "2023-24": 2023,
    "2024-25": 2024,
    "2025-26": 2025,  # Current season - updates every matchday
}

def get_matches(season_label, season_id):
    url = f"{BASE_URL}/competitions/CL/matches?season={season_id}"
    r = requests.get(url, headers=HEADERS)
    print(f"{season_label} → Status: {r.status_code}", end="")
    if r.status_code != 200:
        print(f" | Message: {r.json().get('message', 'unknown error')}")
        return None
    matches = r.json().get("matches", [])
    finished = [m for m in matches if m["status"] == "FINISHED"]
    print(f" | Total: {len(matches)} | Finished: {len(finished)}")
    return finished

def parse(matches, season_label):
    rows = []
    for m in matches:
        rows.append({
            "date": m["utcDate"][:10],
            "season": season_label,
            "home_team": m["homeTeam"]["name"],
            "away_team": m["awayTeam"]["name"],
            "home_goals": m["score"]["fullTime"]["home"],
            "away_goals": m["score"]["fullTime"]["away"],
            "stage": m["stage"],
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    all_dfs = []
    for label, sid in SEASONS.items():
        matches = get_matches(label, sid)
        if matches:
            all_dfs.append(parse(matches, label))
        time.sleep(7)

    if all_dfs:
        df = pd.concat(all_dfs, ignore_index=True)
        out = os.path.join(os.path.dirname(__file__), "../data/raw/ucl_matches.csv")
        df.to_csv(out, index=False)
        print(f"\nTotal: {len(df)} matches saved to data/raw/ucl_matches.csv")
        print(df.groupby("season").size())
    else:
        print("\nNo data fetched from any season.")
