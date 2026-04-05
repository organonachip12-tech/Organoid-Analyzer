import csv
import glob
import os

# Find all matching annotation files
files = glob.glob("annotations_gdc_*.csv")

if not files:
    raise FileNotFoundError("No annotations_gdc_*.csv file found")

# Pick the most recently modified file
latest_file = max(files, key=os.path.getmtime)

print(f"Using file: {latest_file}")

with open(latest_file, newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print("Total rows:", len(rows))
print("First row:", rows[0])

death_counts = {"0": 0, "1": 0}
for r in rows:
    death_counts[r["death_occurred"]] += 1

print("Death counts:", death_counts)