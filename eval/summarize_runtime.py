"""Summarize per-sample editing runtimes from attribute-edit CSVs."""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from statistics import mean, pstdev
from typing import Dict, List


def _as_float(row: Dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value else 0.0


def _fmt(values: List[float]) -> str:
    if len(values) == 1:
        return f"{values[0]:.2f}"
    return f"{mean(values):.2f} +/- {pstdev(values):.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize elapsed_sec from evaluation CSVs")
    parser.add_argument("csv_path", nargs="+")
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()

    groups: Dict[str, List[float]] = defaultdict(list)
    for csv_path in args.csv_path:
        source = os.path.basename(os.path.dirname(os.path.abspath(csv_path)))
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                method = row.get("method", "method")
                groups[f"{source}:{method}"].append(_as_float(row, "elapsed_sec"))

    headers = ["source", "method", "n", "elapsed_sec"]
    rows = []
    for key, values in sorted(groups.items()):
        source, method = key.rsplit(":", 1)
        rows.append({"source": source, "method": method, "n": str(len(values)), "elapsed_sec": _fmt(values)})

    if args.markdown:
        print("| " + " | ".join(headers) + " |")
        print("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            print("| " + " | ".join(row[h] for h in headers) + " |")
    else:
        writer = csv.DictWriter(__import__("sys").stdout, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
