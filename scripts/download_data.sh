#!/usr/bin/env bash
# Re-download every EPL season CSV (football-data.co.uk via the datasets/football-datasets mirror),
# then fill in the season in progress from openfootball, which the mirror doesn't have until it ends.
set -euo pipefail
dir="$(dirname "$0")/../data/raw"
mkdir -p "$dir"
base=https://raw.githubusercontent.com/datasets/football-datasets/main/datasets/premier-league
for y in $(seq 0 30); do
  s=$(printf "%02d%02d" "$y" $((y + 1)))
  if curl -sfL "$base/season-$s.csv" -o "$dir/season-$s.csv"; then
    echo "season-$s.csv"
  else
    rm -f "$dir/season-$s.csv"
  fi
done
current=$(python3 -c 'import datetime as d; t=d.date.today(); y=t.year-(t.month<7); print(f"{y%100:02d}{(y+1)%100:02d}")')
if [ ! -f "$dir/season-$current.csv" ]; then
  python3 "$(dirname "$0")/fetch_current_season.py" "$current"
fi
