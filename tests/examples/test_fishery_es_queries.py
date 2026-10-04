"""The fishery's ES queries resolve against what the ES logs for its regulator.

The ES keys every candidate's parameters by the regulator's mechanism ids, so a
query that names another parameter fails at the end of the first generation.
The optimizer is built from the fishery YAML itself, so renaming a mechanism
there without updating ``examples/bilevel_fishery/queries.py`` fails here. The
queries on the ``inner`` branch read the society's metrics and are not resolved
here.
"""

from pathlib import Path

import numpy as np
import pytest

from core.config.yaml import load_experiment
from core.optimizers.es.optimizer import ESOptimizer
from core.reporting.base import Reporter
from examples.bilevel_fishery.queries import ES_QUERIES

CONFIG = Path(__file__).parents[2] / "examples" / "bilevel_fishery" / "config.yaml"
POPULATION_SIZE = 4
GENERATIONS = 2


class ResolvingReporter(Reporter):
    """Reporter with no backend: only the query resolution is exercised."""

    def _report(self, *args) -> None:
        pass

    def close(self) -> None:
        pass


def es_level(queries):
    return [q for q in queries if q.y_paths[0][0] != "inner"]


@pytest.fixture(scope="module")
def fishery_es() -> ESOptimizer:
    outer_cfg = load_experiment(CONFIG).outer_cfg
    opt = ESOptimizer(outer_cfg)
    opt.batch_capacity = POPULATION_SIZE

    rng = np.random.default_rng(0)
    for generation in range(GENERATIONS):
        population = rng.uniform(size=(POPULATION_SIZE, opt.dimension))
        fitness = rng.normal(size=POPULATION_SIZE)
        opt.best_candidate = population[int(np.argmax(fitness))]
        opt.logger.push_data(
            opt._to_logger_payload(
                generation=generation,
                inner=None,
                population=population,
                fitness=fitness,
                mean=population.mean(axis=0),
                sigma=0.1,
            )
        )

    return opt


@pytest.mark.unit
def test_the_yaml_wires_the_example_queries():
    assert load_experiment(CONFIG).outer_cfg._reporting_queries == ES_QUERIES


@pytest.mark.unit
def test_every_parameter_named_by_a_query_is_searched_by_the_es(fishery_es):
    named = {
        path[path.index("by_parameter") + 1]
        for query in ES_QUERIES
        for path in (query.x, *query.y_paths)
        if "by_parameter" in path
    }

    assert named
    assert named <= set(fishery_es.parameter_names)


@pytest.mark.unit
@pytest.mark.parametrize("query", es_level(ES_QUERIES), ids=lambda q: q.title)
def test_every_es_level_query_resolves(fishery_es, query):
    xs, yss, _, _ = ResolvingReporter()._resolve_query(fishery_es.logger.peek(), query)

    assert xs and all(ys for ys in yss)
