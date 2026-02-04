#!/usr/bin/env python3
"""Report discrepancies between primary_label and primary_label_unoriented entries in results.yaml files."""

import argparse
from pathlib import Path

import yaml


def normalize_to_list(value):
    """Convert a single object or list to a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def get_ids(entries):
    """Extract IDs from a list of entries."""
    return {entry.get("id") for entry in entries if entry and "id" in entry}


def analyze_file(filepath: Path) -> dict | None:
    """Analyze a single results.yaml file for discrepancies.

    Returns a dict with discrepancy info if found, None otherwise.
    """
    with open(filepath) as f:
        data = yaml.safe_load(f)

    if data is None:
        return None

    unoriented_entries = normalize_to_list(data.get("primary_label_unoriented"))
    label_entries = normalize_to_list(data.get("primary_label"))

    unoriented_ids = get_ids(unoriented_entries)

    # Find orphaned primary_labels (parent is not a primary_label_unoriented ID)
    orphaned = []
    for entry in label_entries:
        if entry is None:
            continue
        parent = entry.get("parents")
        entry_id = entry.get("id", "unknown")
        if parent and parent not in unoriented_ids:
            orphaned.append({"id": entry_id, "parent": parent})

    unoriented_count = len(unoriented_entries)
    label_count = len(label_entries)

    # Report if there's a count mismatch or orphaned labels
    if orphaned or unoriented_count != label_count:
        return {
            "unoriented_count": unoriented_count,
            "label_count": label_count,
            "orphaned": orphaned,
        }

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Report primary_label/primary_label_unoriented discrepancies"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/workspace/data/train600"),
        help="Directory containing results.yaml files",
    )
    args = parser.parse_args()

    results_files = list(args.data_dir.glob("**/results.yaml"))
    results_files.sort()

    discrepancies = []

    for filepath in results_files:
        result = analyze_file(filepath)
        if result:
            # Get relative path from data_dir for cleaner output
            rel_path = filepath.relative_to(args.data_dir)
            discrepancies.append({"file": str(rel_path), **result})

    # Print report
    print("Label Discrepancy Report")
    print("=" * 24)
    print(f"Total files scanned: {len(results_files)}")
    print(f"Files with discrepancies: {len(discrepancies)}")

    if discrepancies:
        print("\nDiscrepancies:")
        print("-" * 14)
        for d in discrepancies:
            print(f"\nFile: {d['file']}")
            print(f"  primary_label_unoriented count: {d['unoriented_count']}")
            print(f"  primary_label count: {d['label_count']}")
            if d["orphaned"]:
                print("  Orphaned primary_labels:")
                for orphan in d["orphaned"]:
                    print(f"    - {orphan['id']} (parent: {orphan['parent']})")


if __name__ == "__main__":
    main()
