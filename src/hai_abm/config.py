"""Model parameters.

Every number here is an *assumption* chosen to give plausible-looking
dynamics, not a value measured from real maritime operations. One model step
is meant to represent roughly one minute of watch time.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

DISRUPTIONS = ("dropout", "spoofing", "comm_delay", "degradation")
MITIGATIONS = ("cross_check", "redundancy", "trust_calibration")


@dataclass(frozen=True)
class Params:
    # --- run timing -------------------------------------------------------
    steps: int = 300
    disruption_start: int = 100
    disruption_duration: int = 80
    baseline_window: tuple[int, int] = (20, 100)  # steps used as "normal" reference

    # --- contact stream (whole area, per step) ---------------------------
    n_sectors: int = 2
    real_rate: float = 0.15  # Poisson mean of real vessel contacts
    clutter_rate: float = 0.5  # Poisson mean of clutter (waves, birds, ...)
    spoof_max_rate: float = 0.5  # Poisson mean of spoofed tracks at severity 1
    deadline: int = 20  # a real contact must reach the commander within this

    # --- AI detectors -----------------------------------------------------
    # P(flag | real), P(flag | clutter), P(flag | spoof), latency in steps
    radar_recall: float = 0.90
    radar_false_pos: float = 0.05
    radar_spoof_flag: float = 0.95
    radar_latency: int = 1
    camera_recall: float = 0.80
    camera_false_pos: float = 0.05
    camera_spoof_flag: float = 0.0  # an RF spoof does not appear on an optical sensor
    camera_latency: int = 2
    base_fail_prob: float = 0.001  # random hardware fault per step
    base_repair_prob: float = 0.2  # per step, once offline

    # --- human operators --------------------------------------------------
    base_trust: float = 0.8  # P(accept AI flag without verifying)
    min_trust: float = 0.1
    capacity: float = 1.0  # work units per step
    verify_cost: float = 2.5  # work units to verify one flag
    accept_cost: float = 0.5  # work units to accept/forward one flag
    overload_threshold: int = 5  # queue length above which operator is overloaded
    overload_verify_factor: float = 0.5  # verification shortcut under overload
    p_confirm_real: float = 0.97  # verification correctly confirms a real contact
    p_reject_clutter: float = 0.95  # verification correctly rejects clutter
    p_reject_spoof: float = 0.80  # spoofs are built to look real
    fatigue: bool = True
    fatigue_gain: float = 0.01  # per overloaded step
    fatigue_decay: float = 0.01  # per non-overloaded step
    fatigue_capacity_loss: float = 0.3  # capacity multiplier at full fatigue = 0.7
    # Trust calibration: trust tracks the observed precision of AI flags,
    # trust += rate * (1 if flag was right else 0 - trust).
    trust_learning_rate: float = 0.1
    max_trust: float = 0.95

    # --- communication ----------------------------------------------------
    operator_to_commander_delay: int = 1
    feedback_delay: int = 5  # commander tells the operator whether an action was justified
    max_extra_delay: float = 8.0  # mean extra delay (steps) at severity 1

    # --- disruption effects at severity 1 ---------------------------------
    dropout_fail_prob: float = 0.2  # offline transitions per step, scaled by severity
    degradation_recall_loss: float = 0.6  # recall multiplier reaches 1 - 0.6*sev
    degradation_fp_gain: float = 0.3  # false-positive rate rises by 0.3*sev

    # --- scenario switches --------------------------------------------------
    disruptions: frozenset[str] = field(default_factory=frozenset)
    severity: float = 0.0
    mitigations: frozenset[str] = field(default_factory=frozenset)

    # --- metric settings ------------------------------------------------------
    recovery_hold: int = 10  # steps the team must stay normal to count as recovered

    def __post_init__(self) -> None:
        unknown = set(self.disruptions) - set(DISRUPTIONS)
        if unknown:
            raise ValueError(f"unknown disruptions: {sorted(unknown)}")
        unknown = set(self.mitigations) - set(MITIGATIONS)
        if unknown:
            raise ValueError(f"unknown mitigations: {sorted(unknown)}")
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError("severity must be in [0, 1]")

    @property
    def disruption_end(self) -> int:
        return self.disruption_start + self.disruption_duration

    def with_(self, **kwargs) -> "Params":
        for key in ("disruptions", "mitigations"):
            if key in kwargs:
                kwargs[key] = frozenset(kwargs[key])
        return replace(self, **kwargs)
