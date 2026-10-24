"""Monte Carlo sweep: disruption type x severity x mitigation x seed."""

from __future__ import annotations

import itertools
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .config import DISRUPTIONS, MITIGATIONS, Params
from .metrics import run_once

SEVERITIES = (0.0, 0.25, 0.5, 0.75, 1.0)
DISRUPTION_SETS = {name: (name,) for name in DISRUPTIONS} | {"combined": DISRUPTIONS}
MITIGATION_SETS = {"none": (), **{name: (name,) for name in MITIGATIONS}, "all": MITIGATIONS}
SEEDS = tuple(range(1000, 1030))  # 30 fixed seeds, shared by every setting

METRICS = ["missed_real", "missed_real_in_window", "false_alarms_acted", "spoofs_acted",
           "overload_steps", "recovery_time", "recovered", "trust_recovery_time",
           "trust_recovered", "cascade_size", "cascade_fraction"]


def _run(job: tuple) -> dict:
    disruption, severity, mitigation, seed = job
    params = Params().with_(disruptions=DISRUPTION_SETS[disruption], severity=severity,
                            mitigations=MITIGATION_SETS[mitigation])
    return dict(disruption=disruption, severity=severity, mitigation=mitigation, seed=seed,
                **run_once(params, seed))


def run_sweep(seeds=SEEDS, severities=SEVERITIES, workers: int | None = None) -> pd.DataFrame:
    jobs = list(itertools.product(DISRUPTION_SETS, severities, MITIGATION_SETS, seeds))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_run, jobs, chunksize=25))
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and 95% CI half-width (normal approximation) per setting."""
    g = df.groupby(["disruption", "severity", "mitigation"], sort=False)[METRICS]
    mean = g.mean().add_suffix("_mean")
    ci = (1.96 * g.std(ddof=1) / np.sqrt(g.count())).add_suffix("_ci95")
    out = pd.concat([mean, ci], axis=1)
    out.insert(0, "n_runs", g.size())
    return out.reset_index()
