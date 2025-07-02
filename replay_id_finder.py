import requests
import time
import argparse
from typing import List, Optional

HOST = "https://replay.pokemonshowdown.com"
SEARCH_ENDPOINT = HOST + "/search.json"
REQUEST_DELAY = 1


def find_replay_ids(
    format_slug: str = "gen9ou", max_pages: int = 100, delay: float = REQUEST_DELAY
) -> List[str]:
    all_ids: List[str] = []
    seen = set()
    before: Optional[int] = None

    for page in range(max_pages):
        params = {"format": format_slug}
        if before is not None:
            params["before"] = before
        resp = requests.get(SEARCH_ENDPOINT, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break

        # only take first 50
        batch = data[:50]
        for entry in batch:
            rid = entry.get("id")
            if rid and rid not in seen:
                seen.add(rid)
                all_ids.append(rid)

        # if fewer than 51 total, were donw
        if len(data) <= 50:
            break

        # else
        before = batch[-1]["uploadtime"]
        time.sleep(delay)

    return all_ids


def main():
    parser = argparse.ArgumentParser(
        description="Fetch gen9ou replay IDs from Showdown’s JSON API"
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=100,
        help="Max batches of 50 to fetch (default: 100)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=REQUEST_DELAY,
        help="Seconds to wait between requests (default: 1)",
    )
    parser.add_argument(
        "--out", type=str, required=True, help="Output file path (one ID per line)"
    )
    args = parser.parse_args()

    ids = find_replay_ids(
        format_slug="gen9ou",
        max_pages=args.pages,
        delay=args.delay,
    )

    with open(args.out, "w") as f:
        for rid in ids:
            f.write(rid + "\n")
    print(f"Wrote {len(ids)} replay IDs to {args.out}")


if __name__ == "__main__":
    main()
