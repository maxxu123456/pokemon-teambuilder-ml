# Pokémon Showdown Team-Scraper

This repository provides two Python scripts for collecting and parsing Pokémon Showdown replays:

1. **replay_id_finder.py** – Collects replay IDs for a specified format.
2. **showdown_scraper.py** – Downloads each replay’s JSON, extracts winning and losing teams, and outputs structured JSON.

## Table of Contents

- [Prerequisites](#prerequisites)  
- [Setup](#setup)  
- [Scripts](#scripts)  
  - [replay_id_finder.py](#replay_id_finderpy)  
  - [showdown_scraper.py](#showdown_scraperpy)  
- [Example Workflow](#example-workflow)  
- [Output Files](#output-files)  

## Prerequisites

- Python
- conda

## Setup

1. Create and activate a conda environment:

   ```bash
   conda create -n showdown-scraper
   conda activate showdown-scraper
   ```

2. Install required packages:

   ```bash
   pip install requests
   ```

## Scripts

### replay_id_finder.py

**Purpose:** Fetches replay IDs for a given Showdown format (e.g., `gen9ou`) in batches.

**Usage:**

```bash
python replay_id_finder.py --pages 10 --out ids.txt
```

- `--pages N` : Maximum number of pages (batches of results) to fetch.  
- `--out FILE`: Output text file to write one replay ID per line.

### showdown_scraper.py

**Purpose:** Reads replay IDs, downloads each replay’s JSON, parses winner/loser teams, and writes results to JSON.

**Usage:**

```bash
python showdown_scraper.py --in-file ids.txt --out-file results.json
```

- `--in-file FILE`  : Path to the input file containing replay IDs (one per line).  
- `--out-file FILE` : Path for the output JSON file.  
- `--delay S`       : *(Optional)* Seconds to wait between requests (default: 1).

## Example Workflow

1. **Collect replay IDs** (first 10 pages of Gen 9 OU):

   ```bash
   python replay_id_finder.py --pages 10 --out ids.txt
   ```

2. **Scrape winning and losing teams**:

   ```bash
   python showdown_scraper.py --in-file ids.txt --out-file results.json
   ```

## Output Files

- **ids.txt**  
  List of replay IDs, one per line:

  ```
  gen9ou-2395706971
  gen9ou-2395686570
  ...
  ```

- **results.json**  
  Array of objects:

  Example:

  ```
  [
    {
      "id": "gen9ou-2395706971",
      "winning_team": ["Scizor", "Great Tusk", ...],
      "losing_team": ["Vaporeon", "Ting-Lu", ...]
    },
    {
      "id": "gen9ou-2395686570",
      "winning_team": null,
      "losing_team": null
    }
    ...
  ]
  ```  
