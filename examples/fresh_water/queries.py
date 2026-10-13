"""Reporting queries of the fresh-water experiment.

Two tuples of ``Query`` objects declare what the reporters plot, in the format
of :mod:`core.reporting.query`. ``ES_QUERIES`` is rendered by the outer (ES)
level against ``ESSchema``: the fitness of the candidates per generation, the
fitness against the searched fixed quota, and the train and eval episode return
of the inner level per generation. ``INNER_QUERIES`` is rendered by the inner
(society) level against ``RaySchema``: the training curves of the farms' mean
reward, of the crop satisfaction, of the lowest reservoir level and of the
quota penalty.

The searched parameters of the eight-dimensional ``water_policy`` mechanism are
named ``water_policy[0]`` to ``water_policy[7]`` by the ES optimizer; index 0 is
the fixed quota (see :data:`examples.fresh_water.mechanism.RULE_NAMES`). Indices
6 and 7 are inert: they are searched but read by no dynamics.

Examples
--------
>>> len(ES_QUERIES), len(INNER_QUERIES)
(3, 4)
>>> [query.title for query in ES_QUERIES][:2]
['Fitness over outer optimization iterations', 'Candidate fitness vs fixed quota']
"""

from core.metrics.enums import ReduceProtocol
from core.reporting.query import Query


def _episode_path(*prefix: str, field: str) -> tuple:
    """Path of an episode-level field, averaged over seeds and episodes."""
    return (
        *prefix,
        "rollout",
        "by_mechanism",
        ReduceProtocol.SERIES,
        "by_seed",
        ReduceProtocol.MEAN,
        "by_episode",
        ReduceProtocol.MEAN,
        field,
    )


def _inner_query(title: str, field: str, y_label: str) -> Query:
    """Training curve of one episode-level field of the inner level."""
    return Query(
        title=title,
        legend_labels=("train",),
        x=("iter",),
        y=(_episode_path("train", field=field),),
        x_label="training iteration",
        y_label=y_label,
    )


ES_QUERIES = (
    Query(
        title="Fitness over outer optimization iterations",
        x=("iter",),
        y=(
            ("by_mechanism", ReduceProtocol.SERIES, "fitness"),
            ("by_mechanism", ReduceProtocol.MEAN, "fitness"),
            ("fitness_best",),
        ),
        legend_labels=("Candidates", "Generation mean", "Generation best"),
        plot_modes=("markers", "lines+markers", "lines+markers"),
        show_group_labels=False,
        x_label="outer optimization iteration",
        y_label="fitness",
        error="std",
        error_path=("by_mechanism",),
    ),
    Query(
        title="Candidate fitness vs fixed quota",
        x=(
            "by_mechanism",
            ReduceProtocol.SERIES,
            "by_parameter",
            "water_policy[0]",
            "value",
        ),
        y=("by_mechanism", ReduceProtocol.SERIES, "fitness"),
        legend_labels=("Evaluated mechanisms",),
        plot_modes=("markers",),
        show_group_labels=False,
        color=("iter",),
        color_label="Outer iteration",
        colorscale="Viridis",
        x_label="fixed quota (normalized parameter)",
        y_label="fitness",
    ),
    Query(
        title="Final train vs eval episode return",
        legend_labels=("train", "eval"),
        x=("iter",),
        y=(
            _episode_path("inner", "train", field="reward_mean"),
            _episode_path("inner", "eval", field="reward_mean"),
        ),
        x_label="ES generation",
        y_label="mean step reward of the farms",
        error="std",
        error_path=(
            "inner",
            "eval",
            "rollout",
            "by_mechanism",
            ReduceProtocol.SERIES,
            "by_seed",
        ),
    ),
)

INNER_QUERIES = (
    _inner_query(
        "Mean step reward over training episodes",
        "reward_mean",
        "mean step reward of the farms",
    ),
    _inner_query(
        "Crop satisfaction over training episodes",
        "crop_satisfaction",
        "mean crop satisfaction",
    ),
    _inner_query(
        "Lowest reservoir level over training episodes",
        "reservoir_level_norm_min",
        "lowest filled fraction",
    ),
    _inner_query(
        "Quota penalty over training episodes", "quota_penalty", "mean quota penalty"
    ),
)
