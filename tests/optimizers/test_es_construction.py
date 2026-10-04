"""``ESConfig`` and ``ESOptimizer`` construction: defaults, validation, regimes.

The search dimension is not configured directly: it is the size of the flattened
action space of the regulator's mechanisms, so every test builds a regulator
with the mechanisms it needs (see ``es_factory`` in ``conftest.py``). Gymnasium
sorts the keys of a ``Dict`` space, so the flat vector and ``parameter_names``
follow the alphabetical order of the mechanism ids, not the declaration order.
"""

import logging

import numpy as np
import pytest
from gymnasium import spaces

from core.optimizers.es.config import ESConfig
from core.optimizers.es.optimizer import ESOptimizer


def unit_box(size: int = 1) -> spaces.Box:
    return spaces.Box(low=0.0, high=1.0, shape=(size,), dtype=np.float32)


@pytest.mark.unit
class TestESConfig:
    def test_defaults(self):
        cfg = ESConfig()

        assert cfg.opt_class is ESOptimizer
        assert (cfg.sigma, cfg.mean_lr, cfg.sigma_lr) == (0.15, 0.1, 0.05)
        assert (cfg.sigma_decay, cfg.min_sigma, cfg.max_sigma) == (0.99, 1e-3, 0.5)
        assert cfg.break_symmetry is False
        assert (cfg.convergence_eps, cfg.convergence_patience) == (1e-4, 10)
        assert cfg.initial_mean is None
        assert cfg.episodes is None

    def test_training_stores_every_hyperparameter_and_chains(self):
        cfg = ESConfig()

        returned = cfg.training(
            episodes=9,
            sigma=0.2,
            mean_lr=0.3,
            sigma_lr=0.4,
            sigma_decay=0.9,
            min_sigma=0.01,
            max_sigma=0.4,
            break_symmetry=True,
            convergence_eps=1e-3,
            convergence_patience=3,
            initial_mean=[0.1, 0.9],
        )

        assert returned is cfg
        assert cfg.episodes == 9
        assert (cfg.sigma, cfg.mean_lr, cfg.sigma_lr, cfg.sigma_decay) == (
            0.2,
            0.3,
            0.4,
            0.9,
        )
        assert (cfg.min_sigma, cfg.max_sigma) == (0.01, 0.4)
        assert cfg.break_symmetry is True
        assert (cfg.convergence_eps, cfg.convergence_patience) == (1e-3, 3)
        assert cfg.initial_mean == [0.1, 0.9]

    def test_the_generation_is_stored_though_nothing_reads_it(self):
        cfg = ESConfig().training(generation=4)

        assert cfg.generation == 4

    def test_unset_arguments_keep_their_current_value(self):
        cfg = ESConfig().training(sigma=0.3, episodes=5)

        cfg.training(mean_lr=0.2)

        assert (cfg.sigma, cfg.mean_lr, cfg.episodes) == (0.3, 0.2, 5)

    def test_custom_optimizer_class_is_kept(self):
        class Other(ESOptimizer):
            pass

        assert ESConfig(opt_class=Other).opt_class is Other


@pytest.mark.unit
class TestHyperparameterValidation:
    @pytest.mark.parametrize(
        "training, message",
        [
            ({"mean_lr": 0.0}, "mean_lr must be positive"),
            ({"mean_lr": -0.1}, "mean_lr must be positive"),
            ({"sigma_lr": -0.1}, "sigma_lr must be non-negative"),
            ({"sigma_decay": 0.0}, r"sigma_decay must be in \(0, 1\]"),
            ({"sigma_decay": 1.5}, r"sigma_decay must be in \(0, 1\]"),
            ({"min_sigma": 0.0}, "min_sigma must be positive"),
            ({"min_sigma": 0.5, "max_sigma": 0.1}, "max_sigma must be >= min_sigma"),
        ],
    )
    def test_invalid_hyperparameter_is_rejected(
        self, es_config_factory, training, message
    ):
        with pytest.raises(ValueError, match=message):
            ESOptimizer(es_config_factory(**training))

    def test_boundary_values_are_accepted(self, es_factory):
        opt = es_factory(sigma_lr=0.0, sigma_decay=1.0, min_sigma=0.2, max_sigma=0.2)

        assert (opt.sigma_lr, opt.sigma_decay) == (0.0, 1.0)
        assert opt.sigma == pytest.approx(0.2)

    @pytest.mark.parametrize(
        "sigma, expected",
        [(5.0, 0.4), (1e-9, 0.01), (0.2, 0.2)],
        ids=["above", "below", "inside"],
    )
    def test_initial_sigma_is_clipped_to_its_bounds(self, es_factory, sigma, expected):
        opt = es_factory(sigma=sigma, min_sigma=0.01, max_sigma=0.4)

        assert opt.sigma == pytest.approx(expected)


@pytest.mark.unit
class TestInitialMean:
    def test_default_is_the_centre_of_the_unit_cube(self, es_factory):
        opt = es_factory({"quota": unit_box(2)})

        assert opt.mean.dtype == np.float32
        np.testing.assert_array_equal(opt.mean, [0.5, 0.5])
        np.testing.assert_array_equal(opt.best_candidate, [0.5, 0.5])

    def test_given_mean_is_used_and_copied(self, es_factory):
        given = [0.2, 0.8]
        opt = es_factory({"quota": unit_box(2)}, initial_mean=given)

        given[0] = 0.99

        np.testing.assert_allclose(opt.mean, [0.2, 0.8])

    @pytest.mark.parametrize(
        "initial_mean, message",
        [
            ([0.5], r"shape \(2,\), got \(1,\)"),
            ([0.5, float("nan")], "finite"),
            ([0.5, float("inf")], "finite"),
            ([0.5, 1.5], r"must be in \[0, 1\]"),
            ([-0.1, 0.5], r"must be in \[0, 1\]"),
        ],
    )
    def test_invalid_mean_is_rejected(self, es_factory, initial_mean, message):
        with pytest.raises(ValueError, match=message):
            es_factory({"quota": unit_box(2)}, initial_mean=initial_mean)


@pytest.mark.unit
class TestSearchSpace:
    def test_parameter_names_follow_the_sorted_mechanism_ids(self, es_factory):
        opt = es_factory(
            {"subsidy": unit_box(1), "quota": unit_box(2), "penalty": unit_box(0)}
        )

        assert opt.parameter_names == ["quota[0]", "quota[1]", "subsidy"]
        assert opt.dimension == 3
        assert opt.fixed_mode is False
        assert opt.search_space.shape == (3,)

    def test_a_single_parameter_mechanism_is_named_by_its_id(self, es_factory):
        assert es_factory().parameter_names == ["quota"]

    def test_only_empty_action_spaces_make_the_fixed_mode(self, es_factory):
        opt = es_factory({"penalty": unit_box(0)}, population=3)

        assert opt.dimension == 0
        assert opt.fixed_mode is True
        assert opt.parameter_names == []
        assert opt.mean.shape == (0,)

    def test_non_box_action_space_is_rejected(self, es_factory):
        with pytest.raises(TypeError, match=r"'rule' has Discrete"):
            es_factory({"rule": spaces.Discrete(3)})

    def test_the_regulator_is_the_first_declared_agent(self, es_config_factory):
        cfg = es_config_factory({"quota": unit_box()})

        opt = ESOptimizer(cfg)

        assert opt.agents_cfgs is cfg.agents_cfgs["regulator"]
        assert list(opt.action_space.spaces) == ["quota"]


@pytest.mark.unit
class TestBatchCapacity:
    @pytest.mark.parametrize("value", [0, -2])
    def test_non_positive_population_is_rejected(self, es_factory, value):
        opt = es_factory(population=None)

        with pytest.raises(ValueError, match="population_size must be positive"):
            opt.batch_capacity = value

    def test_odd_population_needs_break_symmetry(self, es_factory):
        with pytest.raises(ValueError, match="even batch size, got 3"):
            es_factory(population=3)

        assert es_factory(population=3, break_symmetry=True).batch_capacity == 3

    def test_even_population_is_stored(self, es_factory):
        assert es_factory(population=6).batch_capacity == 6

    def test_single_candidate_mode_is_announced(self, es_factory, caplog):
        with caplog.at_level(logging.INFO, logger="core.optimizers.es.optimizer"):
            opt = es_factory(population=1)

        assert opt.batch_capacity == 1
        assert "Single-candidate mode enabled" in caplog.text

    def test_fixed_mode_accepts_any_size_and_announces_it(self, es_factory, caplog):
        with caplog.at_level(logging.INFO, logger="core.optimizers.es.optimizer"):
            opt = es_factory({"penalty": unit_box(0)}, population=3)

        assert opt.batch_capacity == 3
        assert "batch_capacity=3" in caplog.text


@pytest.mark.unit
class TestRandomGenerator:
    def test_same_seed_gives_the_same_population(self, es_factory):
        first = es_factory(seed=7)._sample_population()
        second = es_factory(seed=7)._sample_population()

        np.testing.assert_array_equal(first, second)

    def test_different_seeds_give_different_populations(self, es_factory):
        first = es_factory(seed=7)._sample_population()
        second = es_factory(seed=8)._sample_population()

        assert not np.allclose(first, second)

    def test_without_a_seed_the_optimizer_is_still_usable(self, es_factory):
        population = es_factory(seed=None)._sample_population()

        assert population.shape == (4, 1)
