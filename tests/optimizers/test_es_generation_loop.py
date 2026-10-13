"""``ESOptimizer.train``: one generation is sample, step the environment, update.

The environment is a scripted fake (``ScriptedRegulatorEnv`` in ``conftest.py``)
whose fitness is a plain function of the candidates, so the whole loop runs
without Ray. The summary keys and the stop rules are already covered by
``test_es_train.py``; this file checks what the loop hands to the environment,
what it keeps from the answer and how it fails.

A regulator that holds only mechanisms with an empty action space puts the
optimizer in fixed mode; that path is exercised at the end of the file.
"""

import logging

import numpy as np
import pytest
from gymnasium import spaces

from core.metrics.schemas import MetricSchema

SEED = 21


def unit_box(size: int = 1) -> spaces.Box:
    return spaces.Box(low=0.0, high=1.0, shape=(size,), dtype=np.float32)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def quota_fitness(target: float):
    """Fitness peaking at ``quota == target`` for a one-parameter regulator."""

    def fitness(actions):
        return np.array(
            [-((a["quota"][0] - target) ** 2) for a in actions], dtype=np.float32
        )

    return fitness


class InnerMetrics(MetricSchema):
    value: float | None = None


@pytest.mark.unit
class TestCandidatesHandedToTheEnvironment:
    def test_each_candidate_is_a_dict_of_mechanism_actions(
        self, es_factory, scripted_env
    ):
        opt = es_factory({"subsidy": unit_box(1), "quota": unit_box(2)}, population=4)
        opt.env = scripted_env(lambda actions: np.zeros(len(actions)))

        opt.train()

        population, _ = opt.population_history[0]
        (actions,) = opt.env.received
        assert len(actions) == 4
        for row, action in zip(population, actions):
            assert set(action) == {"quota", "subsidy"}
            np.testing.assert_array_equal(action["quota"], row[:2])
            np.testing.assert_array_equal(action["subsidy"], row[2:])

    def test_the_environment_is_reset_once_per_generation(
        self, es_factory, scripted_env
    ):
        opt = es_factory(episodes=3)
        opt.env = scripted_env(quota_fitness(0.8))

        opt.train()

        assert opt.env.resets == 3
        assert len(opt.env.received) == 3

    def test_the_last_step_of_a_multi_step_episode_gives_the_fitness(
        self, es_factory, scripted_env
    ):
        steps = []

        def fitness(actions):
            steps.append(1)

            return np.full(len(actions), float(len(steps)), dtype=np.float32)

        opt = es_factory(population=2)
        opt.env = scripted_env(fitness, steps=3)

        opt.train()

        assert len(opt.env.received) == 3
        _, kept = opt.population_history[0]
        np.testing.assert_array_equal(kept, [3.0, 3.0])


@pytest.mark.unit
class TestGenerationNumerics:
    def test_one_generation_moves_the_mean_as_computed_by_hand(
        self, es_factory, scripted_env
    ):
        # Two candidates with eps = (z, -z), z the first seeded normal draw, and
        # fitness (2, 1): standardised (+1, -1), g = 2 * z / (2 * sigma) = z / sigma,
        # new logit = mean_lr * g.
        sigma, lr = 0.3, 0.1
        opt = es_factory(population=2, seed=SEED, sigma=sigma, mean_lr=lr, sigma_lr=0.0)
        opt.env = scripted_env(lambda actions: np.array([2.0, 1.0]))

        opt.train()

        z = float(
            np.random.default_rng(SEED).standard_normal((1, 1), dtype=np.float32)[0, 0]
        )
        expected_population = sigmoid([sigma * z, -sigma * z])
        np.testing.assert_allclose(
            opt.population_history[0][0][:, 0], expected_population, atol=1e-6
        )
        np.testing.assert_allclose(opt.mean, sigmoid(lr * z / sigma), rtol=2e-4)
        assert opt.best_fitness == 2.0
        np.testing.assert_allclose(opt.best_candidate, [sigmoid(sigma * z)], atol=1e-6)

    def test_history_keeps_every_generation_population_and_fitness(
        self, es_factory, scripted_env
    ):
        opt = es_factory(episodes=3, population=4)
        opt.env = scripted_env(quota_fitness(0.8))

        result = opt.train()

        assert len(result["population_history"]) == 3
        for population, fitness in result["population_history"]:
            assert population.shape == (4, 1)
            assert fitness.shape == (4,)
            np.testing.assert_allclose(
                fitness, -((population[:, 0] - 0.8) ** 2), atol=1e-6
            )

    def test_the_search_converges_on_a_known_landscape(self, es_factory, scripted_env):
        opt = es_factory(
            episodes=80, population=16, seed=0, sigma=0.3, mean_lr=0.1, sigma_lr=0.05
        )
        opt.env = scripted_env(quota_fitness(0.8))

        result = opt.train()

        assert abs(opt.mean[0] - 0.8) < 0.05
        assert result["best_fitness"] > -1e-3
        assert abs(result["best_mechanism"][0] - 0.8) < 0.05

    def test_single_candidate_mode_evaluates_the_mean_then_keeps_improvements(
        self, es_factory, scripted_env
    ):
        opt = es_factory(episodes=20, population=1, seed=3, sigma=0.3, sigma_lr=0.1)
        opt.env = scripted_env(quota_fitness(0.9))

        result = opt.train()

        first_population, _ = result["population_history"][0]
        np.testing.assert_array_equal(first_population, [[0.5]])
        fitness = np.array([f[0] for _, f in result["population_history"]])
        # The remembered parent is the best candidate seen so far.
        assert opt.fitness_baseline == pytest.approx(fitness.max())
        assert result["best_fitness"] == pytest.approx(fitness.max())


@pytest.mark.unit
class TestFailures:
    def test_no_environment(self, es_factory):
        with pytest.raises(RuntimeError, match="requires a RegulatorEnv"):
            es_factory().train()

    def test_a_regulator_without_horizon_is_rejected(self, es_factory, scripted_env):
        # RegulatorEnv(horizon=None) never terminates, so the generation loop
        # would step forever; the ES refuses it before the first generation.
        opt = es_factory()
        opt.env = scripted_env(quota_fitness(0.8))
        opt.env.horizon = None

        with pytest.raises(ValueError, match="finite horizon"):
            opt.train()

        assert opt.env.resets == 0

    def test_truncation_before_a_terminal_fitness(self, es_factory, scripted_env):
        opt = es_factory()
        opt.env = scripted_env(quota_fitness(0.8), end="truncated")

        with pytest.raises(RuntimeError, match="truncated before terminal ES fitness"):
            opt.train()

    def test_non_finite_fitness_names_the_candidates(self, es_factory, scripted_env):
        opt = es_factory()
        opt.env = scripted_env(lambda actions: np.array([1.0, np.nan, 1.0, np.inf]))

        with pytest.raises(RuntimeError, match=r"Non-finite fitness .* \[1, 3\]"):
            opt.train()

    def test_wrong_number_of_fitness_values(self, es_factory, scripted_env):
        opt = es_factory()
        opt.env = scripted_env(lambda actions: np.ones(3))

        with pytest.raises(RuntimeError, match="returned 3 fitness values for 4"):
            opt.train()

    def test_an_empty_fitness_is_a_count_mismatch(self, es_factory, scripted_env):
        opt = es_factory()
        opt.env = scripted_env(lambda actions: np.empty(0))

        with pytest.raises(RuntimeError, match="returned 0 fitness values for 4"):
            opt.train()

    def test_a_failed_generation_leaves_the_state_untouched(
        self, es_factory, scripted_env
    ):
        opt = es_factory()
        opt.env = scripted_env(lambda actions: np.array([1.0, np.nan, 1.0, 1.0]))

        with pytest.raises(RuntimeError):
            opt.train()

        np.testing.assert_array_equal(opt.mean, [0.5])
        assert opt.population_history == []
        assert opt.best_fitness == -float("inf")


@pytest.mark.unit
class TestMetricsReporting:
    def test_one_report_per_generation_with_the_accumulated_series(
        self, es_factory, scripted_env, recording_reporting
    ):
        opt = es_factory(episodes=3, population=4)
        opt.reporting = recording_reporting
        opt.env = scripted_env(quota_fitness(0.8))

        opt.train()

        assert len(recording_reporting.reports) == 3
        last = recording_reporting.reports[-1]
        assert last.iter == [0, 1, 2]
        assert last.population_size == [4, 4, 4]
        assert len(last.by_mechanism) == 4
        assert set(last.search_mean) == {"quota"}

    def test_the_logged_series_match_the_history(self, es_factory, scripted_env):
        opt = es_factory(episodes=2, population=4)
        opt.env = scripted_env(quota_fitness(0.8))

        opt.train()

        logged = opt.logger.peek()
        fitness = [f for _, f in opt.population_history]
        np.testing.assert_allclose(
            logged.fitness_mean, [f.mean() for f in fitness], rtol=1e-6
        )
        np.testing.assert_allclose(
            logged.fitness_best, [f.max() for f in fitness], rtol=1e-6
        )
        assert logged.best_mechanism_idx == [int(f.argmax()) for f in fitness]

    def test_the_logged_sigma_and_mean_are_those_the_population_was_drawn_from(
        self, es_factory, scripted_env
    ):
        opt = es_factory(
            episodes=2, population=4, sigma=0.3, sigma_lr=1.0, sigma_decay=0.5
        )
        opt.env = scripted_env(quota_fitness(0.8))

        opt.train()

        logged = opt.logger.peek()
        assert logged.sigma[0] == pytest.approx(0.3)
        assert logged.search_mean["quota"].value[0] == pytest.approx(0.5)
        # The second generation is drawn around the mean the first one produced.
        second_mean = logged.search_mean["quota"].value[1]
        assert second_mean != pytest.approx(0.5)
        assert logged.sigma == [pytest.approx(0.3), pytest.approx(0.3)]

    def test_the_inner_metrics_of_the_environment_are_forwarded(
        self, es_factory, scripted_env
    ):
        opt = es_factory(episodes=2)
        opt.env = scripted_env(
            quota_fitness(0.8), info={"metrics": InnerMetrics(value=7.0)}
        )

        opt.train()

        logged = opt.logger.peek()
        assert isinstance(logged.inner, InnerMetrics)
        assert logged.inner.value == [7.0, 7.0]

    @pytest.mark.parametrize("info", [{}, "not a dict"], ids=["no-key", "not-a-dict"])
    def test_an_environment_without_inner_metrics_logs_an_empty_schema(
        self, es_factory, scripted_env, info
    ):
        opt = es_factory()
        opt.env = scripted_env(quota_fitness(0.8), info=info)

        opt.train()

        inner = opt.logger.peek().inner
        assert type(inner) is MetricSchema
        assert inner.iter == []

    def test_the_loop_narrates_each_generation(self, es_factory, scripted_env, caplog):
        opt = es_factory(episodes=2)
        opt.env = scripted_env(quota_fitness(0.8))

        with caplog.at_level(logging.INFO, logger="core.optimizers.es.optimizer"):
            opt.train()

        for marker in (
            "Generation started | gen=0",
            "gen=1",
            "BEFORE UPDATE",
            "AFTER UPDATE",
        ):
            assert marker in caplog.text


def flat_fitness(actions):
    """Same fitness for every candidate: no directional information."""
    return np.ones(len(actions), dtype=np.float32)


@pytest.mark.unit
class TestConvergenceStop:
    """The run stops once the search mean has stopped moving."""

    def test_defaults_are_the_documented_ones(self, es_factory):
        opt = es_factory()

        assert (opt.convergence_eps, opt.convergence_patience) == (0.0, 10)
        assert opt.stalled_generations == 0

    def test_the_default_config_runs_every_generation(self, es_factory, scripted_env):
        # The stop is opt-in: a frozen mean does not end a default run early.
        opt = es_factory(episodes=12)
        opt.env = scripted_env(flat_fitness)

        result = opt.train()

        assert result["episodes"] == 12
        assert result["converged"] is False

    def test_a_frozen_mean_stops_the_run_after_patience_generations(
        self, es_factory, scripted_env, caplog
    ):
        opt = es_factory(episodes=20, convergence_eps=1e-4, convergence_patience=3)
        opt.env = scripted_env(flat_fitness)

        with caplog.at_level(logging.INFO, logger="core.optimizers.es.optimizer"):
            result = opt.train()

        assert result["episodes"] == 3
        assert result["converged"] is True
        assert opt.env.resets == 3
        assert len(result["population_history"]) == 3
        assert "Converged | gen=2" in caplog.text

    def test_a_moving_mean_never_converges(self, es_factory, scripted_env):
        opt = es_factory(
            episodes=15, convergence_eps=1e-4, convergence_patience=2, mean_lr=0.5
        )
        opt.env = scripted_env(quota_fitness(0.9))

        result = opt.train()

        assert result["episodes"] == 15
        assert result["converged"] is False

    def test_zero_eps_disables_the_stop(self, es_factory, scripted_env):
        opt = es_factory(episodes=12, convergence_eps=0.0, convergence_patience=1)
        opt.env = scripted_env(flat_fitness)

        result = opt.train()

        assert result["episodes"] == 12
        assert result["converged"] is False

    def test_the_displacement_is_the_euclidean_norm_in_the_unit_cube(self, es_factory):
        opt = es_factory(
            {"a": unit_box(1), "b": unit_box(1)},
            convergence_eps=0.5,
            convergence_patience=1,
        )
        opt.mean = np.array([0.5, 0.5], dtype=np.float32)

        # Each coordinate moves by 0.3: the norm is 0.424 and stays below eps.
        assert opt._update_convergence(np.array([0.2, 0.8], dtype=np.float32))
        # Each coordinate moves by 0.4: the norm is 0.566 and exceeds eps although
        # no single coordinate does.
        assert not opt._update_convergence(np.array([0.1, 0.9], dtype=np.float32))

    def test_a_large_displacement_resets_the_streak(self, es_factory):
        opt = es_factory(convergence_eps=0.1, convergence_patience=3)
        opt.mean = np.array([0.5], dtype=np.float32)
        still, moved = np.array([0.5], np.float32), np.array([0.9], np.float32)

        assert [opt._update_convergence(v) for v in (still, still)] == [False, False]
        assert opt.stalled_generations == 2
        assert opt._update_convergence(moved) is False
        assert opt.stalled_generations == 0
        assert [opt._update_convergence(v) for v in (still, still, still)] == [
            False,
            False,
            True,
        ]

    def test_single_candidate_mode_counts_rejected_candidates_as_no_movement(
        self, es_factory, scripted_env
    ):
        # A constant fitness is never an improvement: the parent is kept.
        opt = es_factory(
            population=1, episodes=20, convergence_eps=1e-4, convergence_patience=4
        )
        opt.env = scripted_env(flat_fitness)

        result = opt.train()

        assert result["episodes"] == 4
        assert result["converged"] is True

    def test_single_candidate_mode_with_accepted_candidates_keeps_running(
        self, es_factory, scripted_env
    ):
        # Every candidate improves on the previous one, so the parent moves.
        counter = iter(range(1, 1000))
        opt = es_factory(
            population=1, episodes=12, convergence_eps=1e-4, convergence_patience=3
        )
        opt.env = scripted_env(lambda actions: np.array([float(next(counter))]))

        result = opt.train()

        assert result["episodes"] == 12
        assert result["converged"] is False

    def test_fixed_mode_runs_every_generation(self, es_factory, scripted_env):
        opt = es_factory(
            {"penalty": unit_box(0)},
            population=2,
            episodes=12,
            convergence_eps=1e-4,
            convergence_patience=2,
        )
        opt.env = scripted_env(lambda actions: np.array([1.0, 2.0]))

        result = opt.train()

        assert result["episodes"] == 12
        assert result["converged"] is False

    def test_the_streak_restarts_with_each_train_call(self, es_factory, scripted_env):
        opt = es_factory(episodes=20, convergence_eps=1e-4, convergence_patience=3)
        opt.env = scripted_env(flat_fitness)

        assert opt.train()["episodes"] == 3
        assert opt.train()["episodes"] == 3


@pytest.mark.unit
def test_fixed_mode_generation_completes(es_factory, scripted_env):
    """A regulator with only empty-action mechanisms is a documented regime.

    The module docstring describes a fixed mode in which the mechanism is
    evaluated and reported; the payload of the generation must carry the
    fitness values and no parameter entries.
    """
    opt = es_factory({"penalty": unit_box(0)}, population=2)
    opt.env = scripted_env(lambda actions: np.array([1.0, 2.0]))

    result = opt.train()

    assert result["episodes"] == 1
    assert result["best_fitness"] == 2.0

    payload = opt.logger.peek()
    assert payload.fitness_mean == [1.5]
    assert payload.fitness_best == [2.0]
    assert payload.search_mean == {}
    assert payload.by_mechanism["1"].fitness == [2.0]
    assert payload.by_mechanism["1"].by_parameter == {}
