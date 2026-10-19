"""Assembles the scoring network from the inventories.

Shape (design/00 §1):

    disposition -> each of the five constructs
    construct   -> its indicators (one node per indicator, per session)
    constructs  -> each task it is a parent of

Indicator and task nodes are leaves, so a session that observed neither simply
returns the prior — which is the correct answer to "we learned nothing".
"""

from __future__ import annotations

from functools import lru_cache

from probe.inventory.constructs import (
    CONSTRUCTS,
    DISPOSITION,
    DISPOSITION_PRIOR,
    LEVELS,
)
from probe.inventory.indicators import ACTIVE_INDICATORS, INDICATOR_STATES
from probe.scenario.tasks import TASK_STATES, TASKS

from probe.score.bayes import Network, Node
from probe.score.cpt import construct_given_disposition, indicator_cpt, task_cpt


def task_node_name(task_id: str) -> str:
    # Namespaced so a task id can never collide with an indicator id.
    return f"task:{task_id}"


def build_network() -> Network:
    nodes: list[Node] = [
        Node(name=DISPOSITION, states=LEVELS, parents=(), cpt={(): DISPOSITION_PRIOR})
    ]

    shared = construct_given_disposition()
    for construct in CONSTRUCTS:
        nodes.append(
            Node(
                name=construct.id,
                states=LEVELS,
                parents=(DISPOSITION,),
                cpt=shared,
            )
        )

    for indicator in ACTIVE_INDICATORS:
        nodes.append(
            Node(
                name=indicator.id,
                states=INDICATOR_STATES,
                parents=(indicator.construct,),
                cpt=indicator_cpt(indicator),
            )
        )

    for task in TASKS:
        nodes.append(
            Node(
                name=task_node_name(task.id),
                states=TASK_STATES,
                parents=task.constructs,
                cpt=task_cpt(task),
            )
        )

    return Network(nodes)


@lru_cache(maxsize=1)
def default_network() -> Network:
    """Cached because the network is pure data: same parameters, same object."""
    return build_network()
