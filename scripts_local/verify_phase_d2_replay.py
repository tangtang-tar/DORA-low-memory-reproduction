"""Acceptance checks for the real-signal portion of D2."""

import csv
import json

from project_paths import ROOT


local = ROOT / "results/phase_d2/replay"
public = ROOT / "results_summary/phase_d2"
with (local / "replay_detail.csv").open(encoding="utf-8") as file:
    rows = list(csv.DictReader(file))
summary = json.loads((local / "replay_summary.json").read_text(encoding="utf-8"))

assert len(rows) == 12
assert {row["method"] for row in rows} == {
    "official_round_raw", "official_round", "largest_remainder", "cumulative_deficit"
}
assert all(int(row["raw_sum"]) == 4 for row in rows if row["method"] != "official_round_raw")
assert any(row["differs_from_largest_remainder"] == "True" for row in rows)
assert (local / "replay_detail.csv").read_bytes() == (public / "replay_detail.csv").read_bytes()
assert (local / "replay_summary.json").read_bytes() == (public / "replay_summary.json").read_bytes()
assert summary["trace_rounds"] == 3
print("D2 real-signal replay acceptance passed")
print(json.dumps(summary, ensure_ascii=False, indent=2))
