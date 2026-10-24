"""The watch-team model: contact stream, team network, message bus, recording."""

from __future__ import annotations

import heapq
import itertools
from collections import defaultdict

import mesa
import networkx as nx
import numpy as np

from .agents import AIDetector, Commander, Contact, HumanOperator
from .config import Params


class WatchTeamModel(mesa.Model):
    """A maritime watch team: per sector, AI detector(s) feed one operator; operators feed a commander."""

    def __init__(self, params: Params | None = None, seed: int = 0):
        super().__init__(rng=seed)
        self.params = params or Params()
        self.seed = seed
        # Separate stream for contacts, so every scenario with the same seed
        # faces the same real/clutter traffic (common random numbers).
        self.contact_rng = np.random.default_rng([seed, 7919])
        self.t = 0
        self._next_contact_id = 0
        self.contacts: list[Contact] = []
        self.contacts_by_arrival: dict[int, list[Contact]] = defaultdict(list)
        self.new_contacts: list[Contact] = []
        self._bus: list[tuple[int, int, int, tuple]] = []  # (deliver_at, seq, recipient, payload)
        self._seq = itertools.count()
        self._build_team()
        self.records: dict[str, list] = defaultdict(list)
        self.abnormal: dict[int, list[bool]] = {a.unique_id: [] for a in self.agents}

    # --- structure ----------------------------------------------------------
    def _build_team(self) -> None:
        p = self.params
        self.graph = nx.DiGraph()
        self.commander = Commander(self)
        self.graph.add_node(self.commander.unique_id, role="commander")
        self.operators: list[HumanOperator] = []
        self.detectors: list[AIDetector] = []
        modalities = ["radar", "camera"] if "redundancy" in p.mitigations else ["radar"]
        for sector in range(p.n_sectors):
            op = HumanOperator(self, sector)
            self.operators.append(op)
            self.graph.add_node(op.unique_id, role="operator", sector=sector)
            self.graph.add_edge(op.unique_id, self.commander.unique_id, kind="report")
            self.graph.add_edge(self.commander.unique_id, op.unique_id, kind="feedback")
            for modality in modalities:
                det = AIDetector(self, sector, modality)
                self.detectors.append(det)
                self.graph.add_node(det.unique_id, role="detector", sector=sector, modality=modality)
                self.graph.add_edge(det.unique_id, op.unique_id, kind="flag")
        self._by_id = {a.unique_id: a for a in self.agents}

    # --- disruption helpers -------------------------------------------------------
    def in_disruption_window(self, t: int | None = None) -> bool:
        t = self.t if t is None else t
        return self.params.disruption_start <= t < self.params.disruption_end

    def disruption_active(self, name: str) -> bool:
        p = self.params
        return name in p.disruptions and p.severity > 0 and self.in_disruption_window()

    # --- messaging --------------------------------------------------------------
    def send(self, sender: mesa.Agent, payload: tuple, delay: int,
             recipients: list[int] | None = None) -> None:
        """Deliver `payload` along team-graph edges (to all successors unless `recipients` is given)."""
        extra = 0
        if self.disruption_active("comm_delay"):
            extra = int(self.rng.poisson(self.params.severity * self.params.max_extra_delay))
        deliver_at = self.t + max(1, delay) + extra
        if recipients is None:
            recipients = list(self.graph.successors(sender.unique_id))
        for recipient in recipients:
            if not self.graph.has_edge(sender.unique_id, recipient):
                raise ValueError(f"no link {sender.unique_id} -> {recipient}")
            heapq.heappush(self._bus, (deliver_at, next(self._seq), recipient, payload))

    def _deliver(self) -> None:
        while self._bus and self._bus[0][0] <= self.t:
            _, _, recipient, payload = heapq.heappop(self._bus)
            self._by_id[recipient].receive(payload)

    # --- contacts -----------------------------------------------------------------
    def _new_contact(self, kind: str) -> Contact:
        c = Contact(self._next_contact_id, kind, int(self.contact_rng.integers(self.params.n_sectors)), self.t)
        self._next_contact_id += 1
        return c

    def _generate_contacts(self) -> None:
        p, rng = self.params, self.contact_rng
        n_real = rng.poisson(p.real_rate)
        n_clutter = rng.poisson(p.clutter_rate)
        n_spoof_candidates = rng.poisson(p.spoof_max_rate)
        spoof_keep = rng.random(n_spoof_candidates)
        new = [self._new_contact("real") for _ in range(n_real)]
        new += [self._new_contact("clutter") for _ in range(n_clutter)]
        if self.disruption_active("spoofing"):
            new += [self._new_contact("spoof") for u in spoof_keep if u < p.severity]
        self.new_contacts = new
        self.contacts.extend(new)
        self.contacts_by_arrival[self.t].extend(new)

    # --- schedule -----------------------------------------------------------------
    def step(self) -> None:
        self._generate_contacts()
        detectors = self.agents_by_type[AIDetector]
        detectors.do("update_status")
        detectors.do("step")
        self._deliver()
        self.agents_by_type[HumanOperator].do("step")
        self.commander.step()
        self._record()
        self.t += 1

    def run(self) -> "WatchTeamModel":
        for _ in range(self.params.steps):
            self.step()
        return self

    def _record(self) -> None:
        r = self.records
        r["t"].append(self.t)
        r["offline"].append(sum(not d.online for d in self.detectors))
        r["total_queue"].append(sum(o.workload() for o in self.operators))
        r["n_overloaded"].append(sum(o.overloaded for o in self.operators))
        r["mean_trust"].append(float(np.mean([o.trust for o in self.operators])))
        r["mean_fatigue"].append(float(np.mean([o.fatigue for o in self.operators])))
        r["in_flight"].append(len(self._bus))
        for a in self.agents:
            self.abnormal[a.unique_id].append(bool(a.abnormal))
