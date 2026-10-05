"""Reporting queries of the bilevel fishery experiment.

Three tuples of ``Query`` objects declare what the reporters plot. A query
names an x path and one or several y paths inside a metric schema, as
described in :mod:`core.reporting.query`.

``ES_QUERIES`` is rendered by the outer (ES) level against ``ESSchema``. Its
first two queries read the ES's own series (fitness per candidate, per
generation and against the searched quota); the other five read the inner
optimizer's metrics under the ``inner`` branch and plot the ``train`` and
``eval`` values of one statistic per generation. ``INNER_QUERIES`` is rendered
by the inner (society) level against ``RaySchema`` and plots the training
curves of the return, the fish biomass statistics and the learner losses.
``FISHERY_ENV_QUERIES`` holds environment-level queries against
``FisheryMetricSchema``; neither ``config.yaml`` nor ``debug.py`` wires them.

Every query names a series that the fishery fills. The environment-level
``Normalized Fish biomass`` query reads ``fish_norm_next_mean``, the
normalized biomass after each transition, which ``FisheryRegulatedEnv``
pushes at every step.

Examples
--------
>>> len(ES_QUERIES), len(INNER_QUERIES), len(FISHERY_ENV_QUERIES)
(7, 9, 1)
>>> [q.title for q in ES_QUERIES if q.y_paths[0][0] != "inner"]
['Fitness over outer optimization iterations', 'Candidate fitness vs quota']
"""

from core.metrics.enums import ReduceProtocol
from core.reporting.query import Query

# Outer-level queries, rendered against ESSchema (see the module docstring).
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
        y_label="objective fitness",
        error="std",
        error_path=("by_mechanism",),
    ),
    Query(
        title="Candidate fitness vs quota",
        x=("by_mechanism", ReduceProtocol.SERIES, "by_parameter", "quota", "value"),
        y=("by_mechanism", ReduceProtocol.SERIES, "fitness"),
        legend_labels=("Evaluated mechanisms",),
        plot_modes=("markers",),
        show_group_labels=False,
        color=("iter",),
        color_label="Outer iteration",
        colorscale="Viridis",
        x_label="quota",
        y_label="objective fitness",
    ),
    Query(
        title="Final train vs eval episode return",
        legend_labels=("train", "eval"),
        x=("iter",),
        y=(
            (
                "inner",
                "train",
                "rollout",
                "by_mechanism",
                ReduceProtocol.SERIES,
                "by_seed",
                ReduceProtocol.MEAN,
                "by_episode",
                ReduceProtocol.MEAN,
                "reward_mean",
            ),
            (
                "inner",
                "eval",
                "rollout",
                "by_mechanism",
                ReduceProtocol.SERIES,
                "by_seed",
                ReduceProtocol.MEAN,
                "by_episode",
                ReduceProtocol.MEAN,
                "reward_mean",
            ),
        ),
        x_label="ES generation",
        y_label="Episode return",
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
    Query(
        title="Final train vs eval mean normalized fish biomass",
        legend_labels=("train", "eval"),
        x=("iter",),
        y=(
            (
                "inner",
                "train",
                "rollout",
                "by_mechanism",
                ReduceProtocol.SERIES,
                "by_seed",
                ReduceProtocol.MEAN,
                "by_episode",
                ReduceProtocol.MEAN,
                "fish_norm_next_mean",
            ),
            (
                "inner",
                "eval",
                "rollout",
                "by_mechanism",
                ReduceProtocol.SERIES,
                "by_seed",
                ReduceProtocol.MEAN,
                "by_episode",
                ReduceProtocol.MEAN,
                "fish_norm_next_mean",
            ),
        ),
    ),
    Query(
        title="Final train vs eval minimum normalized fish biomass",
        legend_labels=("train", "eval"),
        x=("iter",),
        y=(
            (
                "inner",
                "train",
                "rollout",
                "by_mechanism",
                ReduceProtocol.SERIES,
                "by_seed",
                ReduceProtocol.MEAN,
                "by_episode",
                ReduceProtocol.MEAN,
                "fish_norm_next_min",
            ),
            (
                "inner",
                "eval",
                "rollout",
                "by_mechanism",
                ReduceProtocol.SERIES,
                "by_seed",
                ReduceProtocol.MEAN,
                "by_episode",
                ReduceProtocol.MEAN,
                "fish_norm_next_min",
            ),
        ),
    ),
    Query(
        title="Final train vs eval maximum normalized fish biomass",
        legend_labels=("train", "eval"),
        x=("iter",),
        y=(
            (
                "inner",
                "train",
                "rollout",
                "by_mechanism",
                ReduceProtocol.SERIES,
                "by_seed",
                ReduceProtocol.MEAN,
                "by_episode",
                ReduceProtocol.MEAN,
                "fish_norm_next_max",
            ),
            (
                "inner",
                "eval",
                "rollout",
                "by_mechanism",
                ReduceProtocol.SERIES,
                "by_seed",
                ReduceProtocol.MEAN,
                "by_episode",
                ReduceProtocol.MEAN,
                "fish_norm_next_max",
            ),
        ),
    ),
    Query(
        title="Final train vs eval terminal normalized fish biomass",
        legend_labels=("train", "eval"),
        x=("iter",),
        y=(
            (
                "inner",
                "train",
                "rollout",
                "by_mechanism",
                ReduceProtocol.SERIES,
                "by_seed",
                ReduceProtocol.MEAN,
                "by_episode",
                ReduceProtocol.MEAN,
                "fish_norm_next_last",
            ),
            (
                "inner",
                "eval",
                "rollout",
                "by_mechanism",
                ReduceProtocol.SERIES,
                "by_seed",
                ReduceProtocol.MEAN,
                "by_episode",
                ReduceProtocol.MEAN,
                "fish_norm_next_last",
            ),
        ),
    ),
)
# Inner-level queries, rendered against RaySchema.
INNER_QUERIES = (
    Query(
        title="Episode return mean over training episodes",
        legend_labels=("train",),
        x=("iter",),
        y=(
            (
                "train",
                "rollout",
                "by_mechanism",
                ReduceProtocol.SERIES,
                "by_seed",
                ReduceProtocol.MEAN,
                "by_episode",
                ReduceProtocol.MEAN,
                "reward_mean",
            ),
        ),
        x_label="training episode",
        y_label="return mean",
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
        title="Mean normalized fish biomass over training episdoes",
        legend_labels=("train",),
        x=("iter",),
        y=(
            (
                "train",
                "rollout",
                "by_mechanism",
                ReduceProtocol.SERIES,
                "by_seed",
                ReduceProtocol.MEAN,
                "by_episode",
                ReduceProtocol.MEAN,
                "fish_norm_next_mean",
            ),
        ),
        x_label="training episode",
        y_label="normalized mean fish biomass",
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
        title="Min normalized fish biomass over training episdoes",
        legend_labels=("train",),
        x=("iter",),
        y=(
            (
                "train",
                "rollout",
                "by_mechanism",
                ReduceProtocol.SERIES,
                "by_seed",
                ReduceProtocol.MEAN,
                "by_episode",
                ReduceProtocol.MEAN,
                "fish_norm_next_min",
            ),
        ),
        x_label="training episode",
        y_label="normalized min fish biomass",
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
        title="Max normalized fish biomass over training episdoes",
        legend_labels=("train",),
        x=("iter",),
        y=(
            (
                "train",
                "rollout",
                "by_mechanism",
                ReduceProtocol.SERIES,
                "by_seed",
                ReduceProtocol.MEAN,
                "by_episode",
                ReduceProtocol.MEAN,
                "fish_norm_next_max",
            ),
        ),
        x_label="training episode",
        y_label="normalized max fish biomass",
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
        title="Terminal rollout normalized fish biomass over training episdoes",
        legend_labels=("train",),
        x=("iter",),
        y=(
            (
                "train",
                "rollout",
                "by_mechanism",
                ReduceProtocol.SERIES,
                "by_seed",
                ReduceProtocol.MEAN,
                "by_episode",
                ReduceProtocol.MEAN,
                "fish_norm_next_last",
            ),
        ),
        x_label="training episode",
        y_label="normalized terminal fish biomass",
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
        title="Value loss over training episdoes",
        legend_labels=("train",),
        x=("iter",),
        y=(
            (
                "train",
                "learner",
                "by_mechanism",
                ReduceProtocol.SERIES,
                "by_seed",
                ReduceProtocol.MEAN,
                "by_policy",
                ReduceProtocol.SERIES,
                "value_loss",
            ),
        ),
        x_label="training episode",
        y_label="value loss",
        error="std",
        error_path=(
            "train",
            "learner",
            "by_mechanism",
            ReduceProtocol.SERIES,
            "by_seed",
        ),
    ),
    Query(
        title="Total loss over training episodes",
        legend_labels=("train",),
        x=("iter",),
        y=(
            (
                "train",
                "learner",
                "by_mechanism",
                ReduceProtocol.SERIES,
                "by_seed",
                ReduceProtocol.MEAN,
                "by_policy",
                ReduceProtocol.SERIES,
                "total_loss",
            ),
        ),
        x_label="training episode",
        y_label="total loss",
        error="std",
        error_path=(
            "train",
            "learner",
            "by_mechanism",
            ReduceProtocol.SERIES,
            "by_seed",
        ),
    ),
    Query(
        title="policy_loss over training episodes",
        legend_labels=("train",),
        x=("iter",),
        y=(
            (
                "train",
                "learner",
                "by_mechanism",
                ReduceProtocol.SERIES,
                "by_seed",
                ReduceProtocol.MEAN,
                "by_policy",
                ReduceProtocol.SERIES,
                "policy_loss",
            ),
        ),
        x_label="training episode",
        y_label="policy loss",
        error="std",
        error_path=(
            "train",
            "learner",
            "by_mechanism",
            ReduceProtocol.SERIES,
            "by_seed",
        ),
    ),
    Query(
        title="Policy entropy over training episodes",
        legend_labels=("train",),
        x=("iter",),
        y=(
            (
                "train",
                "learner",
                "by_mechanism",
                ReduceProtocol.SERIES,
                "by_seed",
                ReduceProtocol.MEAN,
                "by_policy",
                ReduceProtocol.SERIES,
                "policy_entropy",
            ),
        ),
        x_label="training episode",
        y_label="policy entropy",
        error="std",
        error_path=(
            "train",
            "learner",
            "by_mechanism",
            ReduceProtocol.SERIES,
            "by_seed",
        ),
    ),
)

# Environment-level queries against FisheryMetricSchema; not wired by default.
FISHERY_ENV_QUERIES = (
    Query(
        title="Normalized Fish biomass",
        legend_labels=("normalized fish biomass",),
        x=("iter",),
        y=("fish_norm_next_mean",),
    ),
)
