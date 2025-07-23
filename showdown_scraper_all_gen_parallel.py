import requests
import time
import json
import argparse
from typing import List, Tuple, Optional
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed

HOST = "https://replay.pokemonshowdown.com"
SEARCH_ENDPOINT = HOST + "/search.json"


def _clean_mon_name(name: str) -> str:
    name = (
        name.replace("-East", "")
        .replace("-West", "")
        .replace("-*", "")
        .replace("'", "'")
    )
    if name.startswith("Alcremie"):
        return "Alcremie"
    return name


def _download_json(replay_id: str, delay: float) -> dict:
    time.sleep(delay)
    url = f"{HOST}/{replay_id}.json"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _parse_teams(
    log_text: str, p1_name: str, p2_name: str
) -> Tuple[Optional[List[str]], Optional[List[str]]]:
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

    if not p1_team and not p2_team:
        for line in log_text.splitlines():
            parts = line.split("|")
            if parts[1] in ("switch", "drag"):
                slot, species_field = parts[2], parts[3]
                mon = _clean_mon_name(species_field.split(",")[0])
                if slot.startswith("p1a:") and mon not in p1_team:
                    p1_team.append(mon)
                elif slot.startswith("p2a:") and mon not in p2_team:
                    p2_team.append(mon)

    if winner == p1_name:
        return p1_team, p2_team
    elif winner == p2_name:
        return p2_team, p1_team
    return None, None


def scrape_replay(
    replay_id: str, delay: float
) -> Tuple[Optional[List[str]], Optional[List[str]]]:
    data = _download_json(replay_id, delay)
    players = data.get("players", [])
    if len(players) != 2:
        return None, None
    p1_name, p2_name = players
    log_text = data.get("log", "")
    return _parse_teams(log_text, p1_name, p2_name)


def find_replay_ids(format_slug: str, max_pages: int, delay: float) -> List[str]:
    all_ids: List[str] = []
    seen = set()
    before = None
    for _ in range(max_pages):
        params = {"format": format_slug}
        if before is not None:
            params["before"] = before
        resp = requests.get(SEARCH_ENDPOINT, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        page_items = data[:50]
        for entry in page_items:
            rid = entry.get("id")
            if rid and rid not in seen:
                seen.add(rid)
                all_ids.append(rid)
        if len(data) <= 50:
            break
        before = page_items[-1]["uploadtime"]
        time.sleep(delay)
    return all_ids


def main():
    parser = argparse.ArgumentParser(
        description="Scrape replays for generations 1-9 OU and extract teams."
    )
    parser.add_argument(
        "--pages-per-gen",
        nargs=9,
        type=int,
        metavar=("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9"),
        default=[100] * 9,
        help="Number of pages to fetch for gen1 through gen9 (default 100 each).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Secons to wait between API requests (default 1.0).",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=8,
        help="Number of parallel worker threads (default 8).",
    )
    parser.add_argument("--out-file", required=True, help="Path to output JSON file.")
    args = parser.parse_args()

    results = []
    for gen in range(1, 10):
        format_slug = f"gen{gen}ou"
        pages = args.pages_per_gen[gen - 1]
        print(f"Fetching up to {pages} pages of {format_slug}...")
        ids = find_replay_ids(format_slug, pages, args.delay)
        print(
            f"  Found {len(ids)} IDs; scraping replays with {args.workers} workers..."
        )

        # parallel scrape
        scrape_fn = partial(scrape_replay, delay=args.delay)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(scrape_fn, rid): rid for rid in ids}
            for future in as_completed(futures):
                rid = futures[future]
                try:
                    win, lose = future.result()
                except Exception as e:
                    print(f"Error scraping {rid}: {e}")
                    win, lose = None, None
                results.append(
                    {
                        "generation": gen,
                        "id": rid,
                        "winning_team": win,
                        "losing_team": lose,
                    }
                )

    with open(args.out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Done: wrote {len(results)} entries to {args.out_file}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("%s seconds to run parallel script" % (time.time() - start_time))
