import os
import subprocess
from pathlib import Path

base_dir = Path("data/raw/boxscores/2025-26")
for date_dir in sorted(base_dir.iterdir()):
    if not date_dir.is_dir(): continue
    if date_dir.name >= "2026-02-08":
        for file in date_dir.glob("*.txt"):
            print(f"Ingesting {file}")
            subprocess.run(["python3", "run_cli.py", "ingest-boxscore", str(file)])
