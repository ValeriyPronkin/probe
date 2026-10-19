"""Inference checked against hand-computable cases."""

import pytest

from probe.score.bayes import (
    Network,
    Node,
    hard_evidence,
    posterior,
    soft_evidence,
)


def rain_network() -> Network:
    rain = Node("rain", ("no", "yes"), (), {(): (0.8, 0.2)})
    wet = Node("wet", ("no", "yes"), ("rain",), {("no",): (0.9, 0.1), ("yes",): (0.2, 0.8)})
    return Network([rain, wet])


def test_prior_is_returned_when_nothing_is_observed():
    assert posterior(rain_network(), {}) == {"rain": {"no": 0.8, "yes": 0.2}}


def test_hard_evidence_matches_bayes_by_hand():
    # P(rain|wet) = .2*.8 / (.2*.8 + .8*.1) = 2/3
    result = posterior(rain_network(), {"wet": hard_evidence(("no", "yes"), "yes")})
    assert result["rain"]["yes"] == pytest.approx(2 / 3)


def test_flat_soft_evidence_carries_no_information():
    """An observer who is guessing must not move the posterior at all."""
    flat = soft_evidence(("no", "yes"), "yes", 0.5)
    assert posterior(rain_network(), {"wet": flat}) == posterior(rain_network(), {})


def test_soft_evidence_moves_less_than_hard_evidence():
    net = rain_network()
    soft = posterior(net, {"wet": soft_evidence(("no", "yes"), "yes", 0.7)})
    hard = posterior(net, {"wet": hard_evidence(("no", "yes"), "yes")})
    prior = posterior(net, {})
    assert prior["rain"]["yes"] < soft["rain"]["yes"] < hard["rain"]["yes"]


def test_unobserved_leaf_marginalises_out():
    """A leaf nobody observed must be exactly equivalent to not existing — this
    is what makes 'we did not detect it' safe to treat as silence."""
    base = Network([Node("rain", ("no", "yes"), (), {(): (0.8, 0.2)})])
    assert posterior(rain_network(), {}, query=("rain",)) == posterior(
        base, {}, query=("rain",)
    )


def test_a_leaf_marginal_can_be_queried_explicitly():
    result = posterior(rain_network(), {}, query=("wet",))
    # P(wet) = .8*.1 + .2*.8 = .24
    assert result["wet"]["yes"] == pytest.approx(0.24)


def test_impossible_evidence_raises_instead_of_renormalising():
    net = rain_network()
    with pytest.raises(ValueError, match="zero probability"):
        posterior(net, {"wet": (0.0, 0.0)})


def test_parent_declared_after_child_is_rejected():
    child = Node("wet", ("no", "yes"), ("rain",), {("no",): (1.0, 0.0), ("yes",): (0.0, 1.0)})
    parent = Node("rain", ("no", "yes"), (), {(): (0.5, 0.5)})
    with pytest.raises(ValueError, match="unknown or declared after"):
        Network([child, parent])


def test_cpt_row_that_does_not_sum_to_one_is_rejected():
    with pytest.raises(ValueError, match="sums to"):
        Network([Node("a", ("x", "y"), (), {(): (0.5, 0.4)})])


def test_missing_cpt_row_is_rejected():
    parent = Node("rain", ("no", "yes"), (), {(): (0.5, 0.5)})
    child = Node("wet", ("no", "yes"), ("rain",), {("no",): (0.5, 0.5)})
    with pytest.raises(ValueError, match="CPT rows mismatch"):
        Network([parent, child])


def test_evidence_for_unknown_node_raises():
    with pytest.raises(KeyError):
        posterior(rain_network(), {"snow": (1.0, 0.0)})
