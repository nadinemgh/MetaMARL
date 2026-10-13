"""``ESOptimizer._to_logger_payload`` and the ES metric schema.

The payload turns one finished generation into an ``ESSchema``. The tests call
it directly with chosen arrays, push several payloads into a ``MetricLogger``
and check that every series grows by one value per generation. The payload
stores the generation index in ``iter``, the field shared by all schemas, and
in the ``generation`` field of ``ESSchema``.
"""

import numpy as np
import pytest
from gymnasium import spaces

from core.metrics.enums import ReduceProtocol
from core.metrics.logger import MetricLogger
from core.metrics.schemas import MetricSchema
from core.optimizers.es.schema import ESSchema

POPULATION = np.array(
    [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]], dtype=np.float32
)


def unit_box(size: int = 1) -> spaces.Box:
    return spaces.Box(low=0.0, high=1.0, shape=(size,), dtype=np.float32)


class InnerSchema(MetricSchema):
    value: float | None = None


@pytest.fixture
def two_parameter_es(es_factory):
    return es_factory({"quota": unit_box(1), "subsidy": unit_box(1)}, population=4)


@pytest.mark.unit
def test_schema_series_fields_reduce_as_series():
    for name in ("sigma", "population_size", "fitness_mean", "fitness_best"):
        extra = ESSchema.model_fields[name].json_schema_extra
        assert extra["reduce"] is ReduceProtocol.SERIES


@pytest.mark.unit
class TestPopulationPayload:
    def make_payload(self, opt, fitness, *, mean=(0.4, 0.5), sigma=0.1, generation=0):
        opt.best_fitness = float(max(opt.best_fitness, fitness.max()))
        opt.best_candidate = POPULATION[int(np.argmax(fitness))]

        return opt._to_logger_payload(
            generation=generation,
            inner=InnerSchema(value=float(generation)),
            population=POPULATION,
            fitness=fitness,
            mean=np.array(mean, dtype=np.float32),
            sigma=sigma,
        )

    def test_scalars_describe_the_generation(self, two_parameter_es):
        payload = self.make_payload(
            two_parameter_es, np.array([1.0, 4.0, 2.0, 3.0]), generation=5, sigma=0.12
        )

        assert payload.iter == 5
        assert payload.generation == 5
        assert payload.sigma == 0.12
        assert payload.population_size == 4
        assert payload.fitness_mean == pytest.approx(2.5)
        assert payload.fitness_best == 4.0
        assert payload.best_mechanism_idx == 1
        assert payload.best_fitness_global == 4.0

    def test_every_candidate_is_keyed_by_its_index_and_parameter_name(
        self, two_parameter_es
    ):
        payload = self.make_payload(two_parameter_es, np.array([1.0, 4.0, 2.0, 3.0]))

        assert set(payload.by_mechanism) == {"0", "1", "2", "3"}
        candidate = payload.by_mechanism["2"]
        assert candidate.fitness == 2.0
        assert candidate.by_parameter["quota"].value == pytest.approx(0.5)
        assert candidate.by_parameter["subsidy"].value == pytest.approx(0.6)

    def test_mean_global_best_and_generation_best_come_from_different_sources(
        self, two_parameter_es
    ):
        payload = self.make_payload(
            two_parameter_es, np.array([1.0, 4.0, 2.0, 3.0]), mean=(0.4, 0.5)
        )

        assert payload.search_mean["quota"].value == pytest.approx(0.4)
        assert payload.search_mean["subsidy"].value == pytest.approx(0.5)
        # Global best is the optimizer's best candidate, here candidate 1.
        assert payload.global_best["quota"].value == pytest.approx(0.3)
        assert payload.global_best["subsidy"].value == pytest.approx(0.4)
        # Generation best is the row of the best fitness of this generation.
        assert payload.generation_best["quota"].value == pytest.approx(0.3)
        assert payload.generation_best["subsidy"].value == pytest.approx(0.4)

    def test_global_best_can_differ_from_the_generation_best(self, two_parameter_es):
        two_parameter_es.best_fitness = 99.0
        two_parameter_es.best_candidate = np.array([0.9, 0.9], dtype=np.float32)

        payload = two_parameter_es._to_logger_payload(
            generation=3,
            inner=InnerSchema(),
            population=POPULATION,
            fitness=np.array([1.0, 2.0, 5.0, 3.0]),
            mean=np.array([0.5, 0.5], dtype=np.float32),
            sigma=0.1,
        )

        assert payload.best_fitness_global == 99.0
        assert payload.global_best["quota"].value == pytest.approx(0.9)
        assert payload.generation_best["quota"].value == pytest.approx(0.5)
        assert payload.best_mechanism_idx == 2

    def test_a_multi_dimensional_mechanism_is_split_into_indexed_names(
        self, es_factory
    ):
        opt = es_factory({"quota": unit_box(2)}, population=2)
        opt.best_candidate = np.array([0.2, 0.3], dtype=np.float32)

        payload = opt._to_logger_payload(
            generation=0,
            inner=InnerSchema(),
            population=np.array([[0.2, 0.3], [0.6, 0.7]], dtype=np.float32),
            fitness=np.array([1.0, 0.0]),
            mean=np.array([0.4, 0.5], dtype=np.float32),
            sigma=0.1,
        )

        assert set(payload.search_mean) == {"quota[0]", "quota[1]"}
        assert payload.by_mechanism["1"].by_parameter[
            "quota[1]"
        ].value == pytest.approx(0.7)

    def test_series_grow_by_one_value_per_generation(self, two_parameter_es):
        logger = MetricLogger.from_schema(ESSchema)

        for generation, fitness in enumerate(
            [np.array([1.0, 4.0, 2.0, 3.0]), np.array([2.0, 1.0, 5.0, 0.0])]
        ):
            logger.push_data(
                self.make_payload(two_parameter_es, fitness, generation=generation)
            )

        peeked = logger.peek()
        assert peeked.iter == [0, 1]
        assert peeked.generation == [0, 1]
        assert peeked.fitness_best == [4.0, 5.0]
        assert peeked.best_fitness_global == [4.0, 5.0]
        assert peeked.best_mechanism_idx == [1, 2]
        assert peeked.by_mechanism["2"].fitness == [2.0, 5.0]
        assert peeked.by_mechanism["2"].by_parameter["quota"].value == pytest.approx(
            [0.5, 0.5]
        )
        assert peeked.generation_best["quota"].value == pytest.approx([0.3, 0.5])
        assert isinstance(peeked.inner, InnerSchema)
        assert peeked.inner.value == [0.0, 1.0]

    def test_peeking_does_not_consume_the_series(self, two_parameter_es):
        logger = MetricLogger.from_schema(ESSchema)
        logger.push_data(self.make_payload(two_parameter_es, np.array([1.0, 4, 2, 3])))

        first = logger.peek()
        second = logger.peek()

        assert first.fitness_best == second.fitness_best == [4.0]
