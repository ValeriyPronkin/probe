"""Scoring one session, and the rebuild invariant that keeps it honest.

The invariant: a stored score must equal the fold recomputed from the stored
admitted observations under the same parameters. Hold it, and changing the model
means rescoring history rather than re-running people through the scenario. Lose
it, and every past session becomes an unfalsifiable number.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass

from probe.inventory.constructs import CONSTRUCT_IDS, LEVEL_VALUES
from probe.score.aggregate import (
    DEFAULT_THRESHOLDS,
    NodeEvidence,
    Thresholds,
    build_evidence,
    to_evidence_map,
)
from probe.score.bayes import Network, expected_level, posterior
from probe.score.network import default_network
from probe.store import repo


def parameters_fingerprint(
    network: Network, thresholds: Thresholds = DEFAULT_THRESHOLDS
) -> str:
    """Stable digest of every number a score depends on.

    Any change to a CPT, a rollup threshold or the inventory changes this. Scores
    carry it, so a stale row is detectable instead of quietly authoritative.
    """
    nodes = [
        {
            "name": node.name,
            "states": list(node.states),
            "parents": list(node.parents),
            "cpt": {
                "|".join(key): [round(p, 12) for p in dist]
                for key, dist in sorted(node.cpt.items())
            },
        }
        for node in network.nodes
    ]
    payload = {
        "nodes": nodes,
        "thresholds": {
            name: round(value, 12) for name, value in sorted(vars(thresholds).items())
        },
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


@dataclass(frozen=True)
class SessionScore:
    session_id: str
    posteriors: dict[str, dict[str, float]]
    evidence: tuple[NodeEvidence, ...]
    fingerprint: str

    def expected(self) -> dict[str, float]:
        """Posterior mean per construct. A convenience for ordering and tests —
        the report shows the distribution, not this scalar."""
        return {
            construct: expected_level(self.posteriors[construct], LEVEL_VALUES)
            for construct in CONSTRUCT_IDS
        }


def score_session(
    connection: sqlite3.Connection,
    session_id: str,
    network: Network | None = None,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> SessionScore:
    network = network or default_network()
    observations = repo.admitted_observations(connection, session_id)
    surfaced = repo.surfaced_tasks(connection, session_id)
    evidence = build_evidence(observations, surfaced, thresholds)
    marginals = posterior(network, to_evidence_map(evidence))
    return SessionScore(
        session_id=session_id,
        posteriors={c: marginals[c] for c in CONSTRUCT_IDS},
        evidence=evidence,
        fingerprint=parameters_fingerprint(network, thresholds),
    )


def score_and_store(
    connection: sqlite3.Connection, session_id: str, network: Network | None = None
) -> SessionScore:
    score = score_session(connection, session_id, network)
    repo.save_scores(connection, session_id, score.posteriors, score.fingerprint)
    return score


def rebuild_matches_stored(
    connection: sqlite3.Connection, session_id: str, tolerance: float = 1e-9
) -> bool:
    """The invariant, as a function rather than a comment. Used by tests and by
    `scripts/replay.py --verify`."""
    stored, fingerprint = repo.load_scores(connection, session_id)
    if not stored:
        return False
    recomputed = score_session(connection, session_id)
    if fingerprint != recomputed.fingerprint:
        return False
    if set(stored) != set(recomputed.posteriors):
        return False
    for construct, marginal in recomputed.posteriors.items():
        if set(stored[construct]) != set(marginal):
            return False
        for level, probability in marginal.items():
            if abs(stored[construct][level] - probability) > tolerance:
                return False
    return True
