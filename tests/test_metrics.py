import numpy as np
import pytest

from hai_abm import Params, WatchTeamModel, compute_metrics
from hai_abm.agents import Contact
from hai_abm.metrics import is_affected, normal_state, recovery_time, trust_normal


# --- recovery time ---------------------------------------------------------------------

def test_recovery_is_zero_when_already_normal():
    assert recovery_time(np.ones(100, bool), end=50, hold=10) == (0, True)


def test_recovery_counts_steps_after_end():
    normal = np.ones(100, bool)
    normal[:57] = False
    assert recovery_time(normal, end=50, hold=10) == (7, True)


def test_recovery_needs_a_sustained_normal_run():
    normal = np.zeros(100, bool)
    normal[55:60] = True  # brief flicker, shorter than hold
    normal[70:] = True
    assert recovery_time(normal, end=50, hold=10) == (20, True)


def test_recovery_is_censored_when_never_reached():
    assert recovery_time(np.zeros(100, bool), end=60, hold=10) == (40, False)


# --- cascade -------------------------------------------------------------------------------

def test_is_affected_compares_against_own_baseline():
    series = np.zeros(100, bool)
    assert not is_affected(series, (0, 50), (50, 100))
    series[60:70] = True  # 20% of the window abnormal
    assert is_affected(series, (0, 50), (50, 100))


def test_is_affected_ignores_agents_that_were_already_this_busy():
    series = np.zeros(100, bool)
    series[::5] = True  # 20% abnormal before and during
    assert not is_affected(series, (0, 50), (50, 100))


# --- normal state --------------------------------------------------------------------------

def _records(n=60, **overrides):
    r = dict(offline=[0] * n, n_overloaded=[0] * n, total_queue=[1] * n,
             in_flight=[2] * n, mean_fatigue=[0.0] * n, mean_trust=[0.8] * n)
    for key, (idx, value) in overrides.items():
        r[key][idx] = value
    return r


def test_normal_state_flags_each_abnormal_condition():
    assert normal_state(_records(), (0, 30)).all()
    for key, value in [("offline", 1), ("n_overloaded", 1), ("total_queue", 9),
                       ("in_flight", 9), ("mean_fatigue", 0.5)]:
        ns = normal_state(_records(**{key: (40, value)}), (0, 30))
        assert not ns[40] and ns[39], key


def test_trust_normal_uses_tolerance():
    r = _records(mean_trust=(40, 0.75))
    r["mean_trust"][41] = 0.6
    tn = trust_normal(r, (0, 30), tolerance=0.1)
    assert tn[40] and not tn[41]


# --- compute_metrics on a hand-built history ---------------------------------------------------

def test_compute_metrics_counts_misses_and_false_alarms():
    p = Params().with_(real_rate=0.0, clutter_rate=0.0, spoof_max_rate=0.0, base_fail_prob=0.0)
    m = WatchTeamModel(p, seed=0).run()  # empty run, just to fill records
    d = p.deadline
    on_time = Contact(0, "real", 0, 110)
    late = Contact(1, "real", 0, 120)
    never = Contact(2, "real", 1, 130)
    too_close_to_end = Contact(3, "real", 1, p.steps - 5)  # deadline after run end: not judged
    clutter = Contact(4, "clutter", 0, 140)
    spoof = Contact(5, "spoof", 1, 150)
    m.contacts = [on_time, late, never, too_close_to_end, clutter, spoof]
    m.commander.actions = {0: 110 + d, 1: 121 + d, 4: 142, 5: 151}

    out = compute_metrics(m)
    assert out["real_contacts"] == 3
    assert out["missed_real"] == 2
    assert out["missed_real_in_window"] == 2
    assert out["false_alarms_acted"] == 2 and out["spoofs_acted"] == 1
    assert out["overload_steps"] == 0
    assert out["cascade_size"] == 0 and out["n_agents"] == 5


def test_overload_steps_matches_operator_stats():
    m = WatchTeamModel(Params().with_(disruptions={"spoofing"}, severity=1.0,
                                      mitigations={"cross_check"}), seed=4).run()
    out = compute_metrics(m)
    assert out["overload_steps"] == sum(o.stats["overload_steps"] for o in m.operators)
    assert out["overload_steps"] > 0
