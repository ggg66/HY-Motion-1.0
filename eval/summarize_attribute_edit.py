"""Summarize temporal attribute-edit evaluation CSVs for paper tables."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from statistics import mean, pstdev
from typing import Dict, Iterable, List


def _as_float(row: Dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value != "" else 0.0


def _as_bool(row: Dict[str, str], key: str) -> bool:
    return str(row.get(key, "")).lower() == "true"


def _fmt_mean_std(values: Iterable[float], digits: int = 2) -> str:
    vals = list(values)
    if not vals:
        return ""
    if len(vals) == 1:
        return f"{vals[0]:.{digits}f}"
    return f"{mean(vals):.{digits}f} +/- {pstdev(vals):.{digits}f}"


def _group_rows(rows: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    groups: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        case_id = row.get("case_id") or row.get("prompt") or "case"
        method = row.get("method", "")
        groups[f"{case_id}:{method}"].append(row)
    return dict(groups)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize attribute-edit results.csv")
    parser.add_argument("csv_path")
    parser.add_argument("--markdown", action="store_true", help="Print a Markdown table")
    args = parser.parse_args()

    with open(args.csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    groups = _group_rows(rows)

    table = []
    for key, group in sorted(groups.items()):
        case_id, method = key.rsplit(":", 1)
        achieved = [_as_float(r, "achieved_delta_m") for r in group]
        achievement = [_as_float(r, "achievement_pct") for r in group]
        jerk = [_as_float(r, "jerk_ratio") for r in group]
        foot = [_as_float(r, "foot_sliding_ratio") for r in group]
        pass_rate = 100.0 * sum(_as_bool(r, "meets_budget") for r in group) / max(len(group), 1)
        table.append(
            {
                "case_id": case_id,
                "method": method,
                "n": str(len(group)),
                "achieved_delta_m": _fmt_mean_std(achieved, 3),
                "achievement_pct": _fmt_mean_std(achievement, 1),
                "jerk_ratio": _fmt_mean_std(jerk, 3),
                "foot_sliding_ratio": _fmt_mean_std(foot, 3),
                "budget_pass_pct": f"{pass_rate:.1f}",
            }
        )

    headers = [
        "case_id",
        "method",
        "n",
        "achieved_delta_m",
        "achievement_pct",
        "jerk_ratio",
        "foot_sliding_ratio",
        "budget_pass_pct",
    ]
    if args.markdown:
        print("| " + " | ".join(headers) + " |")
        print("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in table:
            print("| " + " | ".join(row[h] for h in headers) + " |")
    else:
        writer = csv.DictWriter(__import__("sys").stdout, fieldnames=headers)
        writer.writeheader()
        writer.writerows(table)


if __name__ == "__main__":
    main()
