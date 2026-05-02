import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("FOOTBALL_API_KEY")
HEADERS = {"X-Auth-Token": API_KEY}
OUT = os.path.join(os.path.dirname(__file__), "../data/processed/ucl_teams.json")

print("Fetching 2025-26 UCL teams...")

r = requests.get(
    "https://api.football-data.org/v4/competitions/CL/teams",
    headers=HEADERS,
    params={"season": 2025}
)

print(f"Status: {r.status_code}")

if r.status_code == 200:
    teams = r.json().get("teams", [])
    team_data = {}
    for t in teams:
        team_data[t["name"]] = {
            "id":        t["id"],
            "shortName": t.get("shortName", t["name"]),
            "tla":       t.get("tla", ""),
            "crest":     t.get("crest", ""),
            "country":   t.get("area", {}).get("name", ""),
        }
    with open(OUT, "w") as f:
        json.dump(team_data, f, indent=2)
    print(f"\nSaved {len(team_data)} teams to data/processed/ucl_teams.json")
    for name, info in team_data.items():
        print(f"  {info['tla']:4s} | {name:35s} | {info['crest'][:50]}")
else:
    print(f"Error: {r.json()}")
    print("\nFree tier may not support this endpoint.")
    print("We'll use a hardcoded list instead.")
