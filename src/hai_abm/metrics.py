"""Per-run outcome metrics. The core calculations are pure functions for testability."""

from __future__ import annotations

import numpy as np

from .model import WatchTeamModel


def recovery_time(normal: np.ndarray, end: int, hold: int) -> tuple[int, bool]:
    """Steps after `end` until the team is in its normal state for `hold` consecutive steps.

    Returns (time, recovered). If the team never recovers before the run ends,
    the time is censored at len(normal) - end and recovered is False.
    """
    normal = np.asarray(normal, dtype=bool)
    run = 0
    for t in range(end, len(normal)):
        run = run + 1 if normal[t] else 0
        if run >= hold:
            return t - hold + 1 - end, True
    return len(normal) - end, False


def is_affected(abnormal: np.ndarray, baseline: tuple[int, int], window: tuple[int, int],
                margin: float = 0.05) -> bool:
    """An agent is affected if its abnormal-state rate in `window` exceeds its baseline rate by > margin."""
    abnormal = np.asarray(abnormal, dtype=float)
    pre = abnormal[baseline[0]:baseline[1]].mean()
    during = abnormal[window[0]:window[1]].mean()
    return bool(during - pre > margin)


def normal_state(records: dict, baseline: tuple[int, int]) -> np.ndarray:
    """Per-step boolean: is the team's operating state back to what it was before the disruption?

    Trust is deliberately left out: low trust only matters through the
    workload it creates, which the queue and overload checks already see.
    Trust recovery is reported separately (see `trust_normal`).
    """
    r = {k: np.asarray(v, dtype=float) for k, v in records.items()}
    b = slice(*baseline)
    queue_limit = max(np.percentile(r["total_queue"][b], 95), 1.0)
    flight_limit = max(np.percentile(r["in_flight"][b], 95), 1.0)
    fatigue_limit = r["mean_fatigue"][b].max() + 0.05
    return (
        (r["offline"] == 0)
        & (r["n_overloaded"] == 0)
        & (r["total_queue"] <= queue_limit)
        & (r["in_flight"] <= flight_limit)
        & (r["mean_fatigue"] <= fatigue_limit)
    )


def trust_normal(records: dict, baseline: tuple[int, int], tolerance: float = 0.1) -> np.ndarray:
    """Per-step boolean: mean operator trust within `tolerance` of its pre-disruption mean."""
    trust = np.asarray(records["mean_trust"], dtype=float)
    return trust >= trust[slice(*baseline)].mean() - tolerance


def compute_metrics(model: WatchTeamModel) -> dict:
    p = model.params
    cmd = model.commander
    start, end = p.disruption_start, p.disruption_end
    # Only real contacts whose deadline fell inside the run can be judged.
    judged = [c for c in model.contacts if c.is_real and c.arrival < p.steps - p.deadline]
    missed = [c for c in judged if not cmd.acted_in_time(c)]
    by_id = {c.id: c for c in model.contacts}
    false_acts = [by_id[i] for i in cmd.actions if not by_id[i].is_real]

    normal = normal_state(model.records, p.baseline_window)
    rec_time, recovered = recovery_time(normal, end, p.recovery_hold)
    trust_time, trust_recovered = recovery_time(
        trust_normal(model.records, p.baseline_window), end, p.recovery_hold)

    cascade_window = (start, min(end + 20, p.steps))
    affected = [aid for aid, series in model.abnormal.items()
                if is_affected(series, p.baseline_window, cascade_window)]

    return dict(
        real_contacts=len(judged),
        missed_real=len(missed),
        missed_real_in_window=sum(start <= c.arrival < end for c in missed),
        false_alarms_acted=len(false_acts),
        spoofs_acted=sum(c.kind == "spoof" for c in false_acts),
        overload_steps=int(sum(model.records["n_overloaded"])),
        recovery_time=rec_time,
        recovered=recovered,
        trust_recovery_time=trust_time,
        trust_recovered=trust_recovered,
        cascade_size=len(affected),
        n_agents=len(model.abnormal),
        cascade_fraction=len(affected) / len(model.abnormal),
        min_trust=float(min(model.records["mean_trust"])),
    )


def run_once(params, seed: int) -> dict:
    model = WatchTeamModel(params, seed=seed).run()
    return compute_metrics(model)
