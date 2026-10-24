"""Seeded runs reproduce exactly, and averaged behaviour has the expected direction."""

import numpy as np
import pytest

from hai_abm import DISRUPTIONS, Params, WatchTeamModel, run_once

SEEDS = range(12)


def mean_metric(key, **kwargs):
    p = Params().with_(**kwargs)
    return np.mean([run_once(p, s)[key] for s in SEEDS])


def test_same_seed_same_result():
    p = Params().with_(disruptions=set(DISRUPTIONS), severity=0.7, mitigations={"trust_calibration"})
    a, b = WatchTeamModel(p, seed=42).run(), WatchTeamModel(p, seed=42).run()
    assert a.records == b.records
    assert run_once(p, 42) == run_once(p, 42)


def test_different_seed_different_result():
    p = Params().with_(disruptions={"spoofing"}, severity=0.5)
    assert WatchTeamModel(p, seed=1).run().records != WatchTeamModel(p, seed=2).run().records


def test_zero_severity_is_identical_to_no_disruption():
    a = run_once(Params().with_(disruptions=set(DISRUPTIONS), severity=0.0), 9)
    b = run_once(Params(), 9)
    assert a == b


def test_invalid_params_are_rejected():
    with pytest.raises(ValueError):
        Params().with_(disruptions={"meteor"})
    with pytest.raises(ValueError):
        Params().with_(mitigations={"prayer"})
    with pytest.raises(ValueError):
        Params().with_(severity=1.5)


# Directional sanity checks on fixed seeds (deterministic, so not flaky).

def test_dropout_increases_missed_contacts():
    assert mean_metric("missed_real", disruptions={"dropout"}, severity=1.0) > \
        2 * mean_metric("missed_real")


def test_spoofing_increases_false_alarms_and_cross_check_cuts_them():
    spoof = dict(disruptions={"spoofing"}, severity=1.0)
    none = mean_metric("false_alarms_acted", **spoof)
    assert none > 3 * mean_metric("false_alarms_acted")
    assert mean_metric("false_alarms_acted", mitigations={"cross_check"}, **spoof) < none / 2


def test_redundancy_reduces_missed_contacts_in_normal_operation():
    assert mean_metric("missed_real", mitigations={"redundancy"}) < mean_metric("missed_real") / 2
