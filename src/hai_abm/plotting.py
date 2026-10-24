"""Figures from the sweep summary."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Gray = "no mitigation" reference; the four mitigations take categorical slots 1-4.
STYLE = {
    "none": dict(color="#52514e", marker="o", ls="-", label="No mitigation"),
    "cross_check": dict(color="#2a78d6", marker="s", ls="--", label="Human cross-check"),
    "redundancy": dict(color="#eb6834", marker="^", ls="-.", label="Redundant detectors"),
    "trust_calibration": dict(color="#1baf7a", marker="D", ls=":", label="Trust calibration"),
    "all": dict(color="#eda100", marker="P", ls="-", label="All three"),
}
TITLES = {"dropout": "Sensor dropout", "spoofing": "Spoofed contacts", "comm_delay": "Comm delay",
          "degradation": "AI degradation", "combined": "All four combined"}
TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def _severity_panels(summary: pd.DataFrame, metric: str, ylabel: str, title: str, path: Path):
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.9), sharey=True, facecolor=SURFACE)
    for ax, disruption in zip(axes, TITLES):
        _style_axes(ax)
        sub = summary[summary.disruption == disruption]
        for mit, st in STYLE.items():
            d = sub[sub.mitigation == mit].sort_values("severity")
            ax.errorbar(d.severity, d[f"{metric}_mean"], yerr=d[f"{metric}_ci95"], color=st["color"],
                        marker=st["marker"], ls=st["ls"], lw=2, ms=6, capsize=2.5,
                        elinewidth=1, label=st["label"], mec=SURFACE, mew=1)
        ax.set_title(TITLES[disruption], color=TEXT, fontsize=11)
        ax.set_xlabel("Disruption severity", color=MUTED, fontsize=9)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axes[0].set_ylabel(ylabel, color=TEXT, fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, color=TEXT, fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _recovery_panels(summary: pd.DataFrame, severity: float, path: Path):
    sub = summary[summary.severity == severity]
    panels = [("recovery_time", "Steps to recover team state"),
              ("trust_recovery_time", "Steps to recover operator trust"),
              ("cascade_size", "Agents affected (team: 5, or 7 with redundancy)")]
    disruptions = list(TITLES)
    x = np.arange(len(disruptions))
    width = 0.16
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), facecolor=SURFACE)
    for ax, (metric, ylabel) in zip(axes, panels):
        _style_axes(ax)
        for i, (mit, st) in enumerate(STYLE.items()):
            d = sub[sub.mitigation == mit].set_index("disruption").loc[disruptions]
            ax.bar(x + (i - 2) * width, d[f"{metric}_mean"], width * 0.88, yerr=d[f"{metric}_ci95"],
                   color=st["color"], label=st["label"], error_kw=dict(lw=1, capsize=2, ecolor=MUTED))
        ax.set_xticks(x, [TITLES[d].replace(" ", "\n", 1) for d in disruptions], fontsize=8.5)
        ax.set_title(ylabel, color=TEXT, fontsize=11)
    axes[1].axhline(120, color=MUTED, lw=1, ls="--")
    axes[1].set_ylim(0, 150)
    axes[1].text(-0.4, 122, "end of run (not recovered)", color=MUTED, fontsize=8)
    axes[1].text(0.5, 0.93, "(always 0 without trust calibration: trust is then fixed)",
                 transform=axes[1].transAxes, ha="center", color=MUTED, fontsize=8.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"After the disruption ends (severity {severity:g}): recovery and cascade",
                 color=TEXT, fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def make_figures(summary: pd.DataFrame, outdir: Path) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths = [outdir / "fig1_missed_vs_severity.png",
             outdir / "fig2_false_alarms_vs_severity.png",
             outdir / "fig3_recovery_and_cascade.png"]
    _severity_panels(summary, "missed_real", "Missed real contacts per run",
                     "Missed real vessel contacts (mean of 30 seeds, 95% CI)", paths[0])
    _severity_panels(summary, "false_alarms_acted", "False alarms acted on per run",
                     "False alarms the commander acted on (mean of 30 seeds, 95% CI)", paths[1])
    _recovery_panels(summary, 1.0, paths[2])
    return paths
