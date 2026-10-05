"""Reporting queries of the cart-pole experiment.

Two tuples of ``Query`` objects declare what the reporters plot, in the format
of :mod:`core.reporting.query`. ``ES_QUERIES`` is rendered by the outer (ES)
level against ``ESSchema``: the fitness of the candidates per generation, the
fitness against the searched dial value, and the train and eval episode
return of the inner level per generation. ``INNER_QUERIES`` is rendered by the
inner (society) level against ``RaySchema``: the training curves of the episode
return (the ``reward_total`` of an episode is the number of steps the pole
stayed up) and of the largest pole angle.

Examples
--------
>>> len(ES_QUERIES), len(INNER_QUERIES)
(3, 2)
>>> [query.title for query in ES_QUERIES][:2]
['Fitness over outer optimization iterations', 'Candidate fitness vs dial value']
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
        y_label="mean step reward",
        error="std",
        error_path=("by_mechanism",),
    ),
    Query(
        title="Candidate fitness vs dial value",
        x=("by_mechanism", ReduceProtocol.SERIES, "by_parameter", "dial", "value"),
        y=("by_mechanism", ReduceProtocol.SERIES, "fitness"),
        legend_labels=("Evaluated mechanisms",),
        plot_modes=("markers",),
        show_group_labels=False,
        color=("iter",),
        color_label="Outer iteration",
        colorscale="Viridis",
        x_label="dial value",
        y_label="mean step reward",
    ),
    Query(
        title="Final train vs eval episode return",
        legend_labels=("train", "eval"),
        x=("iter",),
        y=(
            _episode_path("inner", "train", field="reward_total"),
            _episode_path("inner", "eval", field="reward_total"),
        ),
        x_label="ES generation",
        y_label="episode return (steps balanced)",
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
    Query(
        title="Episode return mean over training episodes",
        legend_labels=("train",),
        x=("iter",),
        y=(_episode_path("train", field="reward_total"),),
        x_label="training iteration",
        y_label="episode return (steps balanced)",
        error="std",
        error_path=(
            "train",
            "rollout",
            "by_mechanism",
            ReduceProtocol.SERIES,
            "by_seed",
        ),
    ),
    Query(
        title="Largest pole angle over training episodes",
        legend_labels=("train",),
        x=("iter",),
        y=(_episode_path("train", field="pole_angle_abs_max"),),
        x_label="training iteration",
        y_label="largest absolute pole angle (rad)",
    ),
)
