"""Agents of the maritime watch team.

Information flows one way: AIDetector -> HumanOperator -> Commander.
Messages travel through the model's message bus, so communication delay
applies to every hop.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

import mesa


@dataclass(frozen=True)
class Contact:
    id: int
    kind: str  # "real", "clutter" or "spoof"
    sector: int
    arrival: int

    @property
    def is_real(self) -> bool:
        return self.kind == "real"


@dataclass
class QueueItem:
    contact: Contact
    sources: set[int] = field(default_factory=set)  # detector ids that flagged it

    @property
    def corroborated(self) -> bool:
        return len(self.sources) >= 2


class AIDetector(mesa.Agent):
    """Camera or radar classifier that looks once at each new contact in its sector."""

    def __init__(self, model, sector: int, modality: str):
        super().__init__(model)
        p = model.params
        self.sector = sector
        self.modality = modality
        if modality == "radar":
            self.recall, self.false_pos = p.radar_recall, p.radar_false_pos
            self.spoof_flag, self.latency = p.radar_spoof_flag, p.radar_latency
        elif modality == "camera":
            self.recall, self.false_pos = p.camera_recall, p.camera_false_pos
            self.spoof_flag, self.latency = p.camera_spoof_flag, p.camera_latency
        else:
            raise ValueError(f"unknown modality {modality!r}")
        self.online = True
        self.abnormal = False

    # --- rules ------------------------------------------------------------
    def degradation_level(self) -> float:
        """0 when healthy; rises linearly to `severity` over the disruption window."""
        m = self.model
        if not m.disruption_active("degradation"):
            return 0.0
        progress = (m.t - m.params.disruption_start + 1) / m.params.disruption_duration
        return m.params.severity * min(progress, 1.0)

    def flag_probability(self, kind: str) -> float:
        p = self.model.params
        level = self.degradation_level()
        if kind == "real":
            return self.recall * (1.0 - p.degradation_recall_loss * level)
        if kind == "clutter":
            return min(1.0, self.false_pos + p.degradation_fp_gain * level)
        if kind == "spoof":
            return self.spoof_flag
        raise ValueError(kind)

    def update_status(self) -> None:
        """Two-state (online/offline) Markov chain; dropout raises failures and slows repair."""
        m, p = self.model, self.model.params
        dropout = m.disruption_active("dropout")
        sev = p.severity if dropout else 0.0
        if self.online:
            fail = p.base_fail_prob + p.dropout_fail_prob * sev
            if m.rng.random() < fail:
                self.online = False
        else:
            repair = p.base_repair_prob * (1.0 - sev) + (0.02 if dropout else 0.0)
            if m.rng.random() < repair:
                self.online = True

    def step(self) -> None:
        m = self.model
        flagged_spoof = False
        if self.online:
            for contact in m.new_contacts:
                if contact.sector != self.sector:
                    continue
                if m.rng.random() < self.flag_probability(contact.kind):
                    m.send(self, ("flag", contact, self.unique_id), self.latency)
                    flagged_spoof |= contact.kind == "spoof"
        self.abnormal = (not self.online) or self.degradation_level() > 0.1 or flagged_spoof


class HumanOperator(mesa.Agent):
    """Works a FIFO queue of AI flags: verifies some, accepts the rest on trust."""

    def __init__(self, model, sector: int):
        super().__init__(model)
        p = model.params
        self.sector = sector
        self.trust = p.base_trust
        self.fatigue = 0.0
        self.queue: OrderedDict[int, QueueItem] = OrderedDict()
        self.current: QueueItem | None = None
        self.current_mode: str | None = None
        self.remaining = 0.0
        self.done: set[int] = set()
        self.overloaded = False
        self.abnormal = False
        self.stats = dict(verified=0, accepted=0, caught_errors=0, wrongly_dismissed=0,
                          expired=0, overload_steps=0)

    # --- inbound ----------------------------------------------------------
    def receive(self, payload) -> None:
        if payload[0] == "feedback":
            self.update_trust(flag_was_correct=payload[1].is_real)
            return
        _, contact, detector_id = payload
        if contact.id in self.done:
            return
        if self.current is not None and self.current.contact.id == contact.id:
            self.current.sources.add(detector_id)
            return
        item = self.queue.setdefault(contact.id, QueueItem(contact))
        item.sources.add(detector_id)

    # --- rules ------------------------------------------------------------
    def workload(self) -> int:
        return len(self.queue) + (self.current is not None)

    def is_overloaded(self) -> bool:
        return self.workload() > self.model.params.overload_threshold

    def effective_capacity(self) -> float:
        p = self.model.params
        if not p.fatigue:
            return p.capacity
        return p.capacity * (1.0 - p.fatigue_capacity_loss * self.fatigue)

    def verify_probability(self, item: QueueItem, overloaded: bool) -> float:
        p = self.model.params
        if "cross_check" in p.mitigations:
            # Procedure: every flag is verified unless two independent detectors agree.
            return 0.0 if item.corroborated else 1.0
        prob = 1.0 - self.trust
        if overloaded:
            prob *= p.overload_verify_factor  # automation bias under pressure
        return prob

    def verification_passes(self, contact: Contact) -> bool:
        """Outcome of a human check: True means the contact is judged real."""
        p, rng = self.model.params, self.model.rng
        if contact.kind == "real":
            return rng.random() < p.p_confirm_real
        reject = p.p_reject_clutter if contact.kind == "clutter" else p.p_reject_spoof
        return rng.random() >= reject

    def update_trust(self, flag_was_correct: bool) -> None:
        """Trust calibration: move trust toward the AI's observed hit rate."""
        p = self.model.params
        if "trust_calibration" not in p.mitigations:
            return
        target = 1.0 if flag_was_correct else 0.0
        self.trust += p.trust_learning_rate * (target - self.trust)
        self.trust = min(p.max_trust, max(p.min_trust, self.trust))

    def update_fatigue(self) -> None:
        p = self.model.params
        if self.overloaded:
            self.fatigue = min(1.0, self.fatigue + p.fatigue_gain)
        else:
            self.fatigue = max(0.0, self.fatigue - p.fatigue_decay)

    # --- processing ---------------------------------------------------------
    def _start_next(self) -> bool:
        m, p = self.model, self.model.params
        while self.queue:
            _, item = self.queue.popitem(last=False)
            if m.t - item.contact.arrival > p.deadline:
                self.done.add(item.contact.id)  # too old to matter any more
                self.stats["expired"] += 1
                continue
            verify = m.rng.random() < self.verify_probability(item, self.overloaded)
            self.current = item
            self.current_mode = "verify" if verify else "accept"
            self.remaining = p.verify_cost if verify else p.accept_cost
            return True
        return False

    def _finish(self) -> None:
        contact = self.current.contact
        self.done.add(contact.id)
        if self.current_mode == "accept":
            self.stats["accepted"] += 1
            self._forward(contact)
        else:
            self.stats["verified"] += 1
            # A passed check is learned from later, via the commander's feedback.
            if self.verification_passes(contact):
                self._forward(contact)
            elif contact.is_real:
                self.stats["wrongly_dismissed"] += 1
            else:
                self.stats["caught_errors"] += 1
                self.update_trust(flag_was_correct=False)
        self.current, self.current_mode, self.remaining = None, None, 0.0

    def _forward(self, contact: Contact) -> None:
        self.model.send(self, ("confirmed", contact, self.unique_id),
                        self.model.params.operator_to_commander_delay)

    def step(self) -> None:
        p = self.model.params
        self.overloaded = self.is_overloaded()
        self.stats["overload_steps"] += self.overloaded
        self.update_fatigue()
        budget = self.effective_capacity()
        while budget > 1e-9:
            if self.current is None and not self._start_next():
                break
            used = min(budget, self.remaining)
            self.remaining -= used
            budget -= used
            if self.remaining <= 1e-9:
                self._finish()
        self.abnormal = self.overloaded or self.trust < p.base_trust - 0.15


class Commander(mesa.Agent):
    """Acts on every confirmed contact it receives (once per contact).

    After `feedback_delay` steps the outcome of each action (real vessel or
    not) is reported back to the operator who forwarded it.
    """

    def __init__(self, model):
        super().__init__(model)
        self.actions: dict[int, int] = {}  # contact id -> step acted
        self.false_alarm_this_step = False
        self.abnormal = False

    def receive(self, payload) -> None:
        _, contact, operator_id = payload
        if contact.id in self.actions:
            return
        self.actions[contact.id] = self.model.t
        if not contact.is_real:
            self.false_alarm_this_step = True
        self.model.send(self, ("feedback", contact), self.model.params.feedback_delay,
                        recipients=[operator_id])

    def acted_in_time(self, contact: Contact) -> bool:
        t = self.actions.get(contact.id)
        return t is not None and t - contact.arrival <= self.model.params.deadline

    def step(self) -> None:
        m = self.model
        # Real contacts whose deadline expires right now without a timely action.
        due = m.contacts_by_arrival.get(m.t - m.params.deadline, [])
        missed_now = any(c.is_real and not self.acted_in_time(c) for c in due)
        self.abnormal = self.false_alarm_this_step or missed_now
        self.false_alarm_this_step = False
