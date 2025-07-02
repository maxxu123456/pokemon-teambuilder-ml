"""
showdown_scraper.py: Download Showdown replay JSONs for a list of IDs, parse out
winning and losing teams, and dump the results to a JSON file.
"""

import requests
import time
import json
import argparse
from typing import List, Tuple, Optional

HOST = "https://replay.pokemonshowdown.com"
SLEEP_BETWEEN_REQUESTS = 1


def _clean_mon_name(name: str) -> str:
    """Normalize Pokémon names by stripping regional/alt suffixes."""
    name = (
        name.replace("-East", "")
        .replace("-West", "")
        .replace("-*", "")
        .replace("'", "'")
    )
    if name.startswith("Alcremie"):
        return "Alcremie"
    return name


def _download_json(replay_id: str) -> dict:
    """Fetch and return the JSON object for a given replay id."""
    time.sleep(SLEEP_BETWEEN_REQUESTS)
    url = f"{HOST}/{replay_id}.json"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _parse_teams(
    log_text: str, p1_name: str, p2_name: str
) -> Tuple[Optional[List[str]], Optional[List[str]]]:
    """
    From the raw log text and the two player names, return
    (winning_team, losing_team) as lists of mon-names.
    """
    p1_team: List[str] = []
    p2_team: List[str] = []
    winner: Optional[str] = None

    for line in log_text.splitlines():
        if line.startswith("|poke|p1|"):
            mon = _clean_mon_name(line.split("|")[3].split(",")[0])
            p1_team.append(mon)
        elif line.startswith("|poke|p2|"):
            mon = _clean_mon_name(line.split("|")[3].split(",")[0])
            p2_team.append(mon)
        elif line.startswith("|win|"):
            winner = line.split("|")[2]

    if winner is None:
        return None, None
    if winner == p1_name:
        return p1_team, p2_team
    elif winner == p2_name:
        return p2_team, p1_team
    return None, None


def scrape_replay(replay_id: str) -> Tuple[Optional[List[str]], Optional[List[str]]]:
    """
    Download the JSON for `replay_id`, extract its log, and parse out
    (winning_team, losing_team). Returns (None, None) on failure.
    """
    data = _download_json(replay_id)
    players = data.get("players", [])
    if len(players) != 2:
        return None, None
    p1_name, p2_name = players
    log_text = data.get("log", "")
    return _parse_teams(log_text, p1_name, p2_name)


def main():
    parser = argparse.ArgumentParser(
        description="Read replay IDs from a text file, scrape each via the JSON API, and dump team results to JSON."
    )
    parser.add_argument(
        "--in-file",
        "-i",
        required=True,
        help="Path to input file (one replay ID per line).",
    )
    parser.add_argument(
        "--out-file", "-o", required=True, help="Path to output JSON file."
    )
    parser.add_argument(
        "--delay",
        "-d",
        type=float,
        default=SLEEP_BETWEEN_REQUESTS,
        help="Seconds to wait between API requests.",
    )
    args = parser.parse_args()

    with open(args.in_file, "r") as f:
        ids = [line.strip() for line in f if line.strip()]

    results = []
    for rid in ids:
        win, lose = scrape_replay(rid)
        results.append({"id": rid, "winning_team": win, "losing_team": lose})

    with open(args.out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote {len(results)} entries to {args.out_file}")


if __name__ == "__main__":
    main()
