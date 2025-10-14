"""Merge new training data with existing dataset"""
import json
import sys
import os
from datetime import datetime

def merge_datasets(existing_file, new_file, output_file):
    """Merge two training datasets, combining samples for each letter"""

    # Load existing data
    if os.path.exists(existing_file):
        with open(existing_file, 'r') as f:
            existing_data = json.load(f)
        print(f"Loaded existing dataset: {existing_file}")
        print(f"  Existing samples: {sum(len(v) for v in existing_data.values())} total")
    else:
        existing_data = {}
        print(f"No existing dataset found, starting fresh")

    # Load new data
    with open(new_file, 'r') as f:
        new_data = json.load(f)
    print(f"\nLoaded new dataset: {new_file}")
    print(f"  New samples: {sum(len(v) for v in new_data.values())} total")

    # Merge data
    merged_data = existing_data.copy()
    for letter, images in new_data.items():
        if letter not in merged_data:
            merged_data[letter] = []

        # Add new images (avoiding exact duplicates)
        existing_set = set(merged_data[letter])
        new_count = 0
        for img in images:
            if img not in existing_set:
                merged_data[letter].append(img)
                existing_set.add(img)
                new_count += 1

        print(f"  {letter}: {len(existing_data.get(letter, []))} -> {len(merged_data[letter])} (+{new_count})")

    # Save merged data
    with open(output_file, 'w') as f:
        json.dump(merged_data, f)

    print(f"\nMerged dataset saved to: {output_file}")
    print(f"Total samples: {sum(len(v) for v in merged_data.values())}")

    # Create backup of old data
    if existing_file != output_file and os.path.exists(existing_file):
        backup_file = existing_file.replace('.json', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        os.rename(existing_file, backup_file)
        print(f"Backed up old dataset to: {backup_file}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python merge_training_data.py <new_data.json> [existing_data.json] [output.json]")
        print("\nExample 1: Merge into default dataset")
        print("  python merge_training_data.py new_samples.json")
        print("\nExample 2: Merge specific files")
        print("  python merge_training_data.py new_samples.json training_data_v1.json training_data_v2.json")
        sys.exit(1)

    new_file = sys.argv[1]
    existing_file = sys.argv[2] if len(sys.argv) > 2 else 'training_data_deduplicated_500.json'
    output_file = sys.argv[3] if len(sys.argv) > 3 else existing_file

    if not os.path.exists(new_file):
        print(f"Error: New data file not found: {new_file}")
        sys.exit(1)

    merge_datasets(existing_file, new_file, output_file)