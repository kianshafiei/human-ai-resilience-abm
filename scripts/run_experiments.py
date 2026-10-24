"""Run the full Monte Carlo sweep, then write CSVs, figures and a markdown results table.

Usage:  python scripts/run_experiments.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hai_abm.experiments import SEEDS, run_sweep, summarise  # noqa: E402
from hai_abm.plotting import STYLE, make_figures  # noqa: E402

TABLE_COLS = [("missed_real", "Missed real"), ("false_alarms_acted", "False alarms acted"),
              ("overload_steps", "Overload (op-steps)"), ("recovery_time", "Recovery (steps)"),
              ("trust_recovery_time", "Trust recovery (steps)"), ("cascade_size", "Cascade (agents)")]


def results_table(summary) -> str:
    """Markdown table for the combined disruption at severity 0 and 1."""
    lines = []
    header = "| Severity | Mitigation | " + " | ".join(name for _, name in TABLE_COLS) + " |"
    lines += [header, "|" + "---|" * (len(TABLE_COLS) + 2)]
    sub = summary[summary.disruption == "combined"]
    for sev in (0.0, 0.5, 1.0):
        for mit in STYLE:
            row = sub[(sub.severity == sev) & (sub.mitigation == mit)].iloc[0]
            cells = [f"{row[f'{m}_mean']:.1f} ± {row[f'{m}_ci95']:.1f}" for m, _ in TABLE_COLS]
            lines.append(f"| {sev:g} | {STYLE[mit]['label']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    t0 = time.time()
    df = run_sweep()
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    df.to_csv(results / "runs.csv", index=False)
    summary = summarise(df)
    summary.to_csv(results / "summary.csv", index=False, float_format="%.4f")
    (results / "results_table.md").write_text(results_table(summary) + "\n")
    paths = make_figures(summary, ROOT / "figures")
    print(f"{len(df)} runs ({len(SEEDS)} seeds per setting) in {time.time() - t0:.1f}s")
    for p in paths:
        print("wrote", p.relative_to(ROOT))


if __name__ == "__main__":
    main()
