"""
From industryOR_final_results.json, calculate weighted score and accuracy, write to IndustryOR.json.
"""
from __future__ import annotations

import json
from pathlib import Path

WEIGHT = {"Easy": 1, "Medium": 2, "Hard": 4}


def main() -> None:
    base = Path(__file__).resolve().parent
    src = base / "industryOR_final_results.json"
    out = base / "IndustryOR.json"

    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)

    tasks: list[dict] = []
    for key, row in data.items():
        if not isinstance(row, dict) or "correct" not in row:
            continue
        tasks.append(row)

    total = len(tasks)
    correct_n = 0
    weighted_score = 0
    by_difficulty = {d: {"total": 0, "correct": 0} for d in WEIGHT}
    # by task id (field id), not top level key in JSON
    correct_ids: list[int] = []
    incorrect_ids: list[int] = []

    for row in tasks:
        tid = row.get("id")
        if tid is not None:
            tid = int(tid)
            if row.get("correct"):
                correct_ids.append(tid)
            else:
                incorrect_ids.append(tid)

        diff = row.get("difficulty")
        if diff not in WEIGHT:
            continue
        by_difficulty[diff]["total"] += 1
        if not row.get("correct"):
            continue
        correct_n += 1
        by_difficulty[diff]["correct"] += 1
        weighted_score += WEIGHT[diff]

    accuracy = (correct_n / total) if total else 0.0
    correct_ids.sort()
    incorrect_ids.sort()

    report = {
        "source": str(src.name),
        "total_tasks": total,
        "correct_count": correct_n,
        "weighted_score": weighted_score,
        "accuracy": accuracy,
        "weights": dict(WEIGHT),
        "by_difficulty": by_difficulty,
        "correct_task_ids": correct_ids,
        "incorrect_task_ids": incorrect_ids,
    }

    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Written to {out}")


if __name__ == "__main__":
    main()
