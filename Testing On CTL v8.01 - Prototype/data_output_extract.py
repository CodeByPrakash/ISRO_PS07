import csv

csv_file = "exo_CTL_08.01.csv"

unique_categories = set()

with open(csv_file, "r", encoding="utf-8", newline="") as f:
    reader = csv.reader(f)

    for row in reader:
        if len(row) >= 3:
            unique_categories.add(row[2].strip())

print(f"Total Unique Categories: {len(unique_categories)}")

for category in sorted(unique_categories):
    print(category)