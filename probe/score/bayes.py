"""Exact inference over a small discrete Bayesian network. No I/O, no imports
from the rest of the package — the math is testable on its own and replaceable
by pgmpy in one function if the network ever outgrows enumeration.

The network here is tiny (one disposition + five constructs = 3^6 = 729 latent
assignments), so enumeration is exact, fast and has no convergence behaviour to
explain. Approximate inference would buy nothing and cost auditability.

Evidence is Pearl's *virtual evidence*: an observation contributes a likelihood
vector over a node's states rather than pinning it to one. An observer that is
60% sure moves the posterior less than one that is certain, and a node nobody
observed is simply absent from the evidence map and marginalises out — which is
exactly the treatment "we did not detect it" deserves (design/00 §4).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product

Distribution = tuple[float, ...]
ParentKey = tuple[str, ...]


@dataclass(frozen=True)
class Node:
    name: str
    states: tuple[str, ...]
    parents: tuple[str, ...]
    cpt: Mapping[ParentKey, Distribution]
    """Parent state tuple (in `parents` order) -> distribution over `states`.
    A root node uses the empty tuple as its only key."""

    def distribution(self, parent_states: ParentKey) -> Distribution:
        try:
            return self.cpt[parent_states]
        except KeyError as exc:  # pragma: no cover - a build-time bug, not a runtime path
            raise KeyError(f"{self.name}: no CPT row for parents {parent_states}") from exc


class Network:
    """An immutable DAG. Nodes must be given parents-before-children."""

    def __init__(self, nodes: Sequence[Node]) -> None:
        self.nodes: tuple[Node, ...] = tuple(nodes)
        self.by_name: dict[str, Node] = {}
        for node in self.nodes:
            if node.name in self.by_name:
                raise ValueError(f"duplicate node {node.name!r}")
            for parent in node.parents:
                if parent not in self.by_name:
                    raise ValueError(
                        f"{node.name}: parent {parent!r} is unknown or declared after it"
                    )
            self._check_cpt(node)
            self.by_name[node.name] = node

        has_children = {p for node in self.nodes for p in node.parents}
        # Latent = everything we enumerate over. Leaves are summed out
        # analytically inside the loop, which is what keeps this cheap.
        self.latent: tuple[Node, ...] = tuple(n for n in self.nodes if n.name in has_children)
        self.leaves: tuple[Node, ...] = tuple(n for n in self.nodes if n.name not in has_children)

    def _check_cpt(self, node: Node) -> None:
        expected = {
            tuple(combo)
            for combo in product(*(self.by_name[p].states for p in node.parents))
        }
        if set(node.cpt) != expected:
            missing = sorted(expected - set(node.cpt))
            extra = sorted(set(node.cpt) - expected)
            raise ValueError(f"{node.name}: CPT rows mismatch (missing={missing}, extra={extra})")
        for key, dist in node.cpt.items():
            if len(dist) != len(node.states):
                raise ValueError(f"{node.name}: row {key} has {len(dist)} entries")
            if any(p < 0 for p in dist):
                raise ValueError(f"{node.name}: row {key} has a negative probability")
            if abs(sum(dist) - 1.0) > 1e-9:
                raise ValueError(f"{node.name}: row {key} sums to {sum(dist)}")


Evidence = Mapping[str, Distribution]
"""node name -> likelihood of the observation under each of that node's states."""


def posterior(
    network: Network,
    evidence: Evidence,
    query: Sequence[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Marginal posteriors given virtual evidence.

    By default the query is the latent layer — every node that has children.
    Leaves are summed out analytically rather than enumerated, which is what
    keeps this cheap, so a leaf marginal has to be asked for explicitly via
    `query`; it is then enumerated like any other node.

    Raises if the evidence is impossible under the model (total mass zero), which
    in practice means a malformed likelihood vector rather than a surprising
    participant — a silent renormalise would hide that.
    """
    unknown = set(evidence) - set(network.by_name)
    if unknown:
        raise KeyError(f"evidence for unknown nodes: {sorted(unknown)}")
    for name, vector in evidence.items():
        node = network.by_name[name]
        if len(vector) != len(node.states):
            raise ValueError(f"{name}: evidence vector must cover {len(node.states)} states")
        if any(v < 0 for v in vector):
            raise ValueError(f"{name}: evidence likelihoods must be non-negative")

    requested = set(query) if query is not None else set()
    unknown_query = requested - set(network.by_name)
    if unknown_query:
        raise KeyError(f"query for unknown nodes: {sorted(unknown_query)}")

    enumerated = tuple(
        n for n in network.nodes if n.name in {m.name for m in network.latent} | requested
    )
    summed_out = tuple(n for n in network.leaves if n.name not in requested)

    latent_names = [n.name for n in enumerated]
    totals: dict[str, dict[str, float]] = {
        n.name: dict.fromkeys(n.states, 0.0) for n in enumerated
    }
    mass = 0.0

    for combo in product(*(n.states for n in enumerated)):
        assignment = dict(zip(latent_names, combo, strict=True))
        weight = 1.0

        for node in enumerated:
            parents = tuple(assignment[p] for p in node.parents)
            dist = node.distribution(parents)
            weight *= dist[node.states.index(assignment[node.name])]
            observed = evidence.get(node.name)
            if observed is not None:
                weight *= observed[node.states.index(assignment[node.name])]
            if weight == 0.0:
                break

        if weight == 0.0:
            continue

        for node in summed_out:
            observed = evidence.get(node.name)
            if observed is None:
                # Unobserved leaf: sums to 1 and cannot influence anything.
                continue
            parents = tuple(assignment[p] for p in node.parents)
            dist = node.distribution(parents)
            weight *= sum(p * l for p, l in zip(dist, observed, strict=True))
            if weight == 0.0:
                break

        if weight == 0.0:
            continue

        mass += weight
        for name, state in assignment.items():
            totals[name][state] += weight

    if mass == 0.0:
        raise ValueError("evidence has zero probability under the model")

    return {name: {s: v / mass for s, v in dist.items()} for name, dist in totals.items()}


def expected_level(marginal: Mapping[str, float], values: Mapping[str, float]) -> float:
    """Posterior mean on the [0, 1] ability scale. A reporting convenience — the
    distribution, not this scalar, is what the report shows."""
    return sum(marginal[state] * values[state] for state in marginal)


def hard_evidence(states: Sequence[str], observed: str) -> Distribution:
    return tuple(1.0 if s == observed else 0.0 for s in states)


def soft_evidence(states: Sequence[str], observed: str, confidence: float) -> Distribution:
    """Confidence `c` on `observed`, the remaining 1-c spread over the others.

    At c = 1/len(states) this is a flat vector, i.e. no information — which is
    the right behaviour for an observer that is guessing.
    """
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in [0, 1]")
    if observed not in states:
        raise ValueError(f"{observed!r} is not one of {tuple(states)}")
    others = len(states) - 1
    rest = (1.0 - confidence) / others if others else 0.0
    return tuple(confidence if s == observed else rest for s in states)
