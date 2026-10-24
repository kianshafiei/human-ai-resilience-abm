import numpy as np
import pytest

from hai_abm import Params, WatchTeamModel
from hai_abm.agents import AIDetector, Commander, Contact, HumanOperator, QueueItem


def make_model(**kwargs) -> WatchTeamModel:
    # No background traffic, so each test controls exactly what agents see.
    defaults = dict(real_rate=0.0, clutter_rate=0.0, spoof_max_rate=0.0)
    defaults.update(kwargs)
    return WatchTeamModel(Params().with_(**defaults), seed=1)


# --- team structure ---------------------------------------------------------------

def test_baseline_team_structure():
    m = make_model()
    roles = [d["role"] for _, d in m.graph.nodes(data=True)]
    assert roles.count("detector") == 2 and roles.count("operator") == 2 and roles.count("commander") == 1
    for det in m.detectors:
        (op_id,) = m.graph.successors(det.unique_id)
        assert m.graph.nodes[op_id]["sector"] == det.sector


def test_redundancy_adds_a_camera_per_sector():
    m = make_model(mitigations={"redundancy"})
    assert sorted(d.modality for d in m.detectors) == ["camera", "camera", "radar", "radar"]


def test_send_refuses_links_that_are_not_in_the_graph():
    m = make_model()
    with pytest.raises(ValueError):
        m.send(m.detectors[0], ("flag", None, 0), 1, recipients=[m.commander.unique_id])


# --- detector rules -----------------------------------------------------------------

def test_detector_flag_probabilities_match_parameters():
    m = make_model(mitigations={"redundancy"})
    radar = next(d for d in m.detectors if d.modality == "radar")
    camera = next(d for d in m.detectors if d.modality == "camera")
    p = m.params
    assert radar.flag_probability("real") == p.radar_recall
    assert radar.flag_probability("clutter") == p.radar_false_pos
    assert radar.flag_probability("spoof") == p.radar_spoof_flag
    assert camera.flag_probability("spoof") == 0.0


def test_degradation_ramps_up_during_window_only():
    m = make_model(disruptions={"degradation"}, severity=1.0)
    det = m.detectors[0]
    p = m.params
    m.t = p.disruption_start - 1
    assert det.flag_probability("real") == pytest.approx(p.radar_recall)
    m.t = p.disruption_end - 1  # last step of the window: full degradation
    assert det.flag_probability("real") == pytest.approx(p.radar_recall * (1 - p.degradation_recall_loss))
    assert det.flag_probability("clutter") == pytest.approx(p.radar_false_pos + p.degradation_fp_gain)
    m.t = p.disruption_end
    assert det.flag_probability("real") == pytest.approx(p.radar_recall)


def test_offline_detector_flags_nothing():
    m = make_model(radar_recall=1.0)
    det = m.detectors[0]
    det.online = False
    m.new_contacts = [Contact(0, "real", det.sector, 0)]
    det.step()
    assert m._bus == [] and det.abnormal


def test_online_detector_with_perfect_recall_flags_its_sector_only():
    m = make_model(radar_recall=1.0)
    det = m.detectors[0]
    m.new_contacts = [Contact(0, "real", det.sector, 0), Contact(1, "real", 1 - det.sector, 0)]
    det.step()
    assert [msg[3][1].id for msg in m._bus] == [0]


def test_dropout_takes_detectors_offline_much_more_often():
    m = WatchTeamModel(Params().with_(disruptions={"dropout"}, severity=1.0), seed=3).run()
    offline = np.array(m.records["offline"])
    p = m.params
    assert offline[p.disruption_start:p.disruption_end].mean() > 1.0  # of 2 detectors
    assert offline[:p.disruption_start].mean() < 0.1


# --- operator rules ------------------------------------------------------------------

def test_verify_probability_follows_trust_and_overload():
    m = make_model()
    op = m.operators[0]
    item = QueueItem(Contact(0, "real", 0, 0), {1})
    assert op.verify_probability(item, overloaded=False) == pytest.approx(1 - op.trust)
    assert op.verify_probability(item, overloaded=True) == pytest.approx(0.5 * (1 - op.trust))


def test_cross_check_verifies_everything_not_corroborated():
    m = make_model(mitigations={"cross_check"})
    op = m.operators[0]
    single = QueueItem(Contact(0, "real", 0, 0), {1})
    double = QueueItem(Contact(1, "real", 0, 0), {1, 2})
    assert op.verify_probability(single, overloaded=True) == 1.0
    assert op.verify_probability(double, overloaded=False) == 0.0


def test_trust_is_fixed_without_calibration():
    op = make_model().operators[0]
    op.update_trust(flag_was_correct=False)
    assert op.trust == op.model.params.base_trust


def test_trust_calibration_moves_toward_observed_precision():
    op = make_model(mitigations={"trust_calibration"}).operators[0]
    lr = op.model.params.trust_learning_rate
    op.update_trust(flag_was_correct=False)
    assert op.trust == pytest.approx(0.8 - lr * 0.8)
    before = op.trust
    op.update_trust(flag_was_correct=True)
    assert op.trust == pytest.approx(before + lr * (1 - before))
    for _ in range(200):
        op.update_trust(flag_was_correct=False)
    assert op.trust == op.model.params.min_trust


def test_overload_threshold():
    m = make_model()
    op = m.operators[0]
    for i in range(m.params.overload_threshold):
        op.receive(("flag", Contact(i, "real", 0, 0), 1))
    assert not op.is_overloaded()
    op.receive(("flag", Contact(99, "real", 0, 0), 1))
    assert op.is_overloaded()


def test_duplicate_flags_merge_into_one_corroborated_item():
    op = make_model().operators[0]
    c = Contact(0, "real", 0, 0)
    op.receive(("flag", c, 1))
    op.receive(("flag", c, 2))
    assert op.workload() == 1 and op.queue[0].corroborated


def test_stale_flags_expire_without_being_forwarded():
    m = make_model()
    op = m.operators[0]
    m.t = 50
    op.receive(("flag", Contact(0, "real", 0, arrival=50 - m.params.deadline - 1), 1))
    op.step()
    assert op.stats["expired"] == 1 and m._bus == []


def test_accepted_flag_is_forwarded_to_commander():
    m = make_model(base_trust=1.0)  # never verifies
    op = m.operators[0]
    op.receive(("flag", Contact(0, "clutter", 0, 0), 1))
    op.step()
    assert op.stats["accepted"] == 1
    (_, _, recipient, payload) = m._bus[0]
    assert recipient == m.commander.unique_id and payload[0] == "confirmed"


def test_fatigue_reduces_capacity():
    op = make_model().operators[0]
    full = op.effective_capacity()
    op.fatigue = 1.0
    assert op.effective_capacity() == pytest.approx(full * (1 - op.model.params.fatigue_capacity_loss))


# --- commander rules -----------------------------------------------------------------

def test_commander_acts_once_and_flags_false_alarms():
    m = make_model()
    cmd, op = m.commander, m.operators[0]
    clutter = Contact(0, "clutter", 0, 0)
    m.t = 3
    cmd.receive(("confirmed", clutter, op.unique_id))
    m.t = 4
    cmd.receive(("confirmed", clutter, op.unique_id))
    assert cmd.actions == {0: 3} and cmd.false_alarm_this_step
    # outcome feedback goes back to the forwarding operator only
    assert [msg[2] for msg in m._bus] == [op.unique_id]


def test_commander_deadline():
    m = make_model()
    cmd = m.commander
    c = Contact(0, "real", 0, arrival=10)
    m.t = 10 + m.params.deadline
    cmd.receive(("confirmed", c, m.operators[0].unique_id))
    assert cmd.acted_in_time(c)
    late = Contact(1, "real", 0, arrival=10)
    m.t = 11 + m.params.deadline
    cmd.receive(("confirmed", late, m.operators[0].unique_id))
    assert not cmd.acted_in_time(late)


# --- disruptions in the model -----------------------------------------------------------

def test_spoofs_only_appear_during_spoofing_window():
    m = WatchTeamModel(Params().with_(disruptions={"spoofing"}, severity=1.0), seed=2).run()
    spoofs = [c for c in m.contacts if c.kind == "spoof"]
    assert spoofs
    assert all(m.params.disruption_start <= c.arrival < m.params.disruption_end for c in spoofs)
    clean = WatchTeamModel(Params(), seed=2).run()
    assert not [c for c in clean.contacts if c.kind == "spoof"]


def test_comm_delay_slows_messages_in_window():
    m = make_model(disruptions={"comm_delay"}, severity=1.0)
    m.t = m.params.disruption_start
    for _ in range(200):
        m.send(m.detectors[0], ("flag", None, 0), 1)
    delays = [msg[0] - m.t for msg in m._bus]
    assert np.mean(delays) == pytest.approx(1 + m.params.max_extra_delay, rel=0.15)


def test_contact_stream_is_shared_across_mitigations():
    """Common random numbers: same seed -> same real/clutter traffic whatever the mitigation."""
    a = WatchTeamModel(Params(), seed=5).run()
    b = WatchTeamModel(Params().with_(mitigations={"redundancy", "cross_check"}), seed=5).run()
    assert a.contacts == b.contacts
