"""Numerics of ``ESOptimizer``: sampling, fitness shaping, mean and sigma updates.

Each expected value is computed in the test from the formulas of the module
docstring, never read back from the optimizer. Two exact structural facts make
the hand computations short:

- the search runs in logit space around ``logit(mean)``, so a candidate is
  ``sigmoid(logit(mean) + sigma * eps)``; with ``mean = 0.5`` the logit is 0;
- the population is antithetic, ``eps`` for the first half and ``-eps`` for the
  second half, so the estimator of the first half is
  ``mean((F+ - F-) * eps) / (2 sigma)`` with ``F`` the fitness standardised to
  mean 0 and standard deviation 1.

Candidates handed to ``_update_parameters`` are built from chosen ``eps``
vectors, so the gradient can be written down without running any random draw.
The sampling tests replay the generator with the same seed instead.
"""

import math

import numpy as np
import pytest
from gymnasium import spaces

LOGIT_CLIP = 1e-6


def unit_box(size: int = 1) -> spaces.Box:
    return spaces.Box(low=0.0, high=1.0, shape=(size,), dtype=np.float32)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), LOGIT_CLIP, 1.0 - LOGIT_CLIP)
    return np.log(p / (1.0 - p))


def candidates(mean_logit, sigma, eps):
    """Float32 population ``sigmoid(mean_logit + sigma * eps)``."""
    return sigmoid(np.asarray(mean_logit) + sigma * np.asarray(eps)).astype(np.float32)


@pytest.mark.unit
class TestLogitSpace:
    def test_sigmoid_matches_the_closed_form(self, es_factory):
        x = np.array([-3.0, -0.5, 0.0, 2.0])

        np.testing.assert_allclose(es_factory()._sigmoid(x), 1 / (1 + np.exp(-x)))

    def test_sigmoid_saturates_without_overflow(self, es_factory):
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            out = es_factory()._sigmoid(np.array([-1e4, 1e4]))

        np.testing.assert_array_equal(out, [0.0, 1.0])

    def test_logit_inverts_sigmoid(self, es_factory):
        p = np.array([0.1, 0.5, 0.9])

        np.testing.assert_allclose(es_factory()._logit(p), np.log(p / (1 - p)))

    def test_logit_is_finite_on_the_closed_interval(self, es_factory):
        out = es_factory()._logit(np.array([0.0, 1.0]))

        expected = math.log(LOGIT_CLIP / (1 - LOGIT_CLIP))
        np.testing.assert_allclose(out, [expected, -expected])


@pytest.mark.unit
class TestSampling:
    def test_population_replays_the_seeded_antithetic_noise(self, es_factory):
        opt = es_factory({"quota": unit_box(2)}, population=4, seed=123, sigma=0.3)

        population = opt._sample_population()

        z = np.random.default_rng(123).standard_normal((2, 2), dtype=np.float32)
        eps = np.vstack([z, -z])
        np.testing.assert_allclose(population, candidates(0.0, 0.3, eps), atol=1e-6)
        assert population.dtype == np.float32

    def test_mirrored_candidates_are_symmetric_about_the_mean(self, es_factory):
        # With mean 0.5, sigmoid(-x) = 1 - sigmoid(x): the pairs sum to 1.
        population = es_factory(
            {"quota": unit_box(3)}, population=6
        )._sample_population()

        np.testing.assert_allclose(population[:3] + population[3:], 1.0, atol=1e-6)

    def test_mirrored_candidates_are_symmetric_in_logit_space_around_any_mean(
        self, es_factory
    ):
        opt = es_factory(
            {"quota": unit_box(2)}, population=4, initial_mean=[0.2, 0.9], sigma=0.4
        )

        population = opt._sample_population()

        centre = logit(np.array([0.2, 0.9]))
        np.testing.assert_allclose(
            logit(population[:2]) - centre, -(logit(population[2:]) - centre), atol=1e-4
        )

    def test_population_is_centred_on_the_logit_of_the_mean(self, es_factory):
        opt = es_factory(
            {"quota": unit_box(2)},
            population=4,
            seed=5,
            initial_mean=[0.2, 0.9],
            sigma=0.4,
        )

        population = opt._sample_population()

        z = np.random.default_rng(5).standard_normal((2, 2), dtype=np.float32)
        expected = candidates(logit([0.2, 0.9]), 0.4, np.vstack([z, -z]))
        np.testing.assert_allclose(population, expected, atol=1e-6)

    def test_break_symmetry_draws_an_independent_last_candidate(self, es_factory):
        opt = es_factory(
            {"quota": unit_box(2)},
            population=4,
            seed=11,
            sigma=0.3,
            break_symmetry=True,
        )

        population = opt._sample_population()

        rng = np.random.default_rng(11)
        z = rng.standard_normal((2, 2), dtype=np.float32)
        eps = np.vstack([z, -z])
        eps[-1] = rng.standard_normal(2, dtype=np.float32)
        np.testing.assert_allclose(population, candidates(0.0, 0.3, eps), atol=1e-6)
        assert not np.allclose(population[-1] + population[1], 1.0)

    def test_odd_population_appends_one_independent_candidate(self, es_factory):
        opt = es_factory(population=3, seed=2, sigma=0.3, break_symmetry=True)

        population = opt._sample_population()

        rng = np.random.default_rng(2)
        z = rng.standard_normal((1, 1), dtype=np.float32)
        extra = rng.standard_normal((1, 1), dtype=np.float32)
        eps = np.vstack([z, -z, extra])
        np.testing.assert_allclose(population, candidates(0.0, 0.3, eps), atol=1e-6)

    def test_single_candidate_mode_evaluates_the_mean_first(self, es_factory):
        opt = es_factory({"quota": unit_box(2)}, population=1, initial_mean=[0.3, 0.6])

        first = opt._sample_population()

        np.testing.assert_array_equal(first, [[np.float32(0.3), np.float32(0.6)]])
        assert first is not opt.mean

    def test_single_candidate_mode_then_perturbs_the_parent(self, es_factory):
        opt = es_factory(population=1, seed=4, sigma=0.3)
        opt._update_parameters(opt._sample_population(), [1.0])

        offspring = opt._sample_population()

        z = np.random.default_rng(4).standard_normal((1, 1), dtype=np.float32)
        np.testing.assert_allclose(offspring, candidates(0.0, 0.3, z), atol=1e-6)

    def test_fixed_mode_samples_empty_candidates(self, es_factory):
        opt = es_factory({"penalty": unit_box(0)}, population=3)

        population = opt._sample_population()

        assert population.shape == (3, 0)
        assert population.dtype == np.float32

    def test_candidates_stay_inside_the_unit_cube_for_a_huge_sigma(self, es_factory):
        opt = es_factory(
            {"quota": unit_box(3)}, population=8, sigma=50.0, max_sigma=50.0
        )

        with np.errstate(over="raise", invalid="raise", divide="raise"):
            population = opt._sample_population()

        assert np.all(np.isfinite(population))
        assert np.all((population >= 0.0) & (population <= 1.0))
        # A sigma that large pushes nearly every coordinate to a face of the cube.
        assert np.mean((population < 1e-3) | (population > 1 - 1e-3)) > 0.5


@pytest.mark.unit
class TestMeanUpdate:
    def test_two_candidates_one_dimension(self, es_factory):
        # eps = (+1, -1), fitness (2, 1): standardised fitness is (+1, -1), so
        # g = ((1 - (-1)) * 1) / (2 * 0.5) = 2 and the logit moves by 0.1 * 2.
        opt = es_factory(population=2, sigma=0.5, mean_lr=0.1, sigma_lr=0.0)
        population = candidates(0.0, 0.5, [[1.0], [-1.0]])

        opt._update_parameters(population, [2.0, 1.0])

        np.testing.assert_allclose(opt.mean, sigmoid(0.2), rtol=1e-5)
        assert opt.mean.dtype == np.float32

    def test_the_worse_direction_moves_the_mean_the_other_way(self, es_factory):
        opt = es_factory(population=2, sigma=0.5, mean_lr=0.1, sigma_lr=0.0)
        population = candidates(0.0, 0.5, [[1.0], [-1.0]])

        opt._update_parameters(population, [1.0, 2.0])

        np.testing.assert_allclose(opt.mean, sigmoid(-0.2), rtol=1e-5)

    def test_four_candidates_two_dimensions(self, es_factory):
        # Standardisation of f = (4, 1, 2, 3): mean 2.5, variance
        # (1.5**2 + 1.5**2 + 0.5**2 + 0.5**2) / 4 = 1.25.
        # Pairs (candidate i, candidate i + 2): F+ = (4, 1), F- = (2, 3).
        # eps = [[1, 0], [0, 1]] for the first half.
        # g_j = mean_i((F+_i - F-_i) * eps_ij) / (2 * sigma) with sigma = 0.5:
        #   g_0 = (z4 - z2) / 2 = (1.5 - (-0.5)) / sqrt(1.25) / 2 / 1.0 = 1 / sqrt(1.25)
        #   g_1 = (z1 - z3) / 2 = (-1.5 - 0.5) / sqrt(1.25) / 2 / 1.0 = -1 / sqrt(1.25)
        opt = es_factory(
            {"quota": unit_box(2)}, population=4, sigma=0.5, mean_lr=0.5, sigma_lr=0.0
        )
        eps = np.array([[1.0, 0.0], [0.0, 1.0]])
        population = candidates(0.0, 0.5, np.vstack([eps, -eps]))

        opt._update_parameters(population, [4.0, 1.0, 2.0, 3.0])

        g = math.sqrt(1 / 1.25)
        np.testing.assert_allclose(opt.mean, sigmoid([0.5 * g, -0.5 * g]), rtol=1e-4)

    def test_break_symmetry_uses_the_plain_weighted_average(self, es_factory):
        # Three candidates, eps = (1, -1, 0.5), fitness (3, 1, 2): standardised
        # (+sqrt(1.5), -sqrt(1.5), 0). Not antithetic, so
        # g = mean(F * eps) / sigma = (2 * sqrt(1.5) / 3) / 0.5.
        opt = es_factory(
            population=3, sigma=0.5, mean_lr=0.1, sigma_lr=0.0, break_symmetry=True
        )
        population = candidates(0.0, 0.5, [[1.0], [-1.0], [0.5]])

        opt._update_parameters(population, [3.0, 1.0, 2.0])

        g = (2 * math.sqrt(1.5) / 3) / 0.5
        np.testing.assert_allclose(opt.mean, sigmoid(0.1 * g), rtol=1e-4)

    def test_the_step_is_taken_in_logit_space_around_the_current_mean(self, es_factory):
        opt = es_factory(
            population=2, sigma=0.5, mean_lr=0.1, sigma_lr=0.0, initial_mean=[0.2]
        )
        population = candidates(logit(0.2), 0.5, [[1.0], [-1.0]])

        opt._update_parameters(population, [2.0, 1.0])

        np.testing.assert_allclose(opt.mean, sigmoid(logit(0.2) + 0.2), rtol=1e-4)

    def test_a_gradient_longer_than_five_is_rescaled_to_norm_five(self, es_factory):
        # sigma = 0.1 gives g = 2 / (2 * 0.1) = 10, which is clipped to 5.
        opt = es_factory(population=2, sigma=0.1, mean_lr=0.2, sigma_lr=0.0)
        population = candidates(0.0, 0.1, [[1.0], [-1.0]])

        opt._update_parameters(population, [2.0, 1.0])

        np.testing.assert_allclose(opt.mean, sigmoid(0.2 * 5.0), rtol=1e-5)

    def test_a_gradient_of_norm_below_five_is_not_rescaled(self, es_factory):
        opt = es_factory(population=2, sigma=0.5, mean_lr=0.2, sigma_lr=0.0)
        population = candidates(0.0, 0.5, [[1.0], [-1.0]])

        opt._update_parameters(population, [2.0, 1.0])

        np.testing.assert_allclose(opt.mean, sigmoid(0.2 * 2.0), rtol=1e-5)

    @pytest.mark.parametrize(
        "scale, shift", [(100.0, 7.0), (0.01, -3.0)], ids=["stretch", "shrink"]
    )
    def test_positive_affine_changes_of_the_fitness_do_not_matter(
        self, es_factory, scale, shift
    ):
        eps = np.array([[1.0, 0.5], [-0.3, 1.2]])
        population = candidates(0.0, 0.5, np.vstack([eps, -eps]))
        fitness = np.array([4.0, 1.0, 2.0, 3.0])
        reference = es_factory({"quota": unit_box(2)}, population=4, sigma=0.5)
        rescaled = es_factory({"quota": unit_box(2)}, population=4, sigma=0.5)

        reference._update_parameters(population, fitness)
        rescaled._update_parameters(population, fitness * scale + shift)

        np.testing.assert_allclose(rescaled.mean, reference.mean, rtol=1e-4)

    def test_a_negative_scale_reverses_the_direction(self, es_factory):
        eps = np.array([[1.0, 0.5], [-0.3, 1.2]])
        population = candidates(0.0, 0.5, np.vstack([eps, -eps]))
        fitness = np.array([4.0, 1.0, 2.0, 3.0])
        up = es_factory({"quota": unit_box(2)}, population=4, sigma=0.5)
        down = es_factory({"quota": unit_box(2)}, population=4, sigma=0.5)

        up._update_parameters(population, fitness)
        down._update_parameters(population, -fitness)

        np.testing.assert_allclose(logit(down.mean), -logit(up.mean), atol=1e-5)

    def test_a_flat_population_carries_no_direction(self, es_factory):
        opt = es_factory({"quota": unit_box(2)}, population=4, initial_mean=[0.3, 0.7])
        population = opt._sample_population()

        opt._update_parameters(population, [5.0, 5.0, 5.0, 5.0])

        np.testing.assert_allclose(opt.mean, [0.3, 0.7], atol=1e-6)

    def test_a_spread_below_the_absolute_epsilon_counts_as_flat(self, es_factory):
        # Standard deviation 5e-10 is below EPS = 1e-8, so no direction is taken.
        opt = es_factory(population=2, sigma=0.5)
        population = candidates(0.0, 0.5, [[1.0], [-1.0]])

        opt._update_parameters(population, [0.0, 1e-9])

        np.testing.assert_allclose(opt.mean, 0.5, atol=1e-6)

    def test_the_mean_stays_inside_the_unit_interval_whatever_the_step(
        self, es_factory
    ):
        opt = es_factory(population=2, sigma=0.5, mean_lr=1e4, sigma_lr=0.0)
        population = candidates(0.0, 0.5, [[1.0], [-1.0]])

        with np.errstate(over="raise", invalid="raise", divide="raise"):
            opt._update_parameters(population, [2.0, 1.0])

        assert 0.0 <= opt.mean[0] <= 1.0
        assert np.all(np.isfinite(opt._logit(opt.mean)))
        assert np.all(np.isfinite(opt._sample_population()))


@pytest.mark.unit
class TestBestTracking:
    def test_best_candidate_is_the_argmax_of_the_raw_fitness(self, es_factory):
        opt = es_factory({"quota": unit_box(2)}, population=4)
        population = np.array(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]], dtype=np.float32
        )

        opt._update_parameters(population, [1.0, 4.0, 2.0, 3.0])

        assert opt.best_fitness == 4.0
        assert opt.best_mechanism_idx == 1
        np.testing.assert_array_equal(opt.best_candidate, population[1])

    def test_a_worse_generation_does_not_replace_the_best(self, es_factory):
        opt = es_factory({"quota": unit_box(2)}, population=2)
        first = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        second = np.array([[0.6, 0.7], [0.8, 0.9]], dtype=np.float32)

        opt._update_parameters(first, [1.0, 5.0])
        opt._update_parameters(second, [4.0, 2.0])

        assert opt.best_fitness == 5.0
        np.testing.assert_array_equal(opt.best_candidate, first[1])
        assert opt.best_mechanism_idx == 1

    def test_a_better_generation_replaces_the_best(self, es_factory):
        opt = es_factory({"quota": unit_box(2)}, population=2)
        first = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        second = np.array([[0.6, 0.7], [0.8, 0.9]], dtype=np.float32)

        opt._update_parameters(first, [1.0, 5.0])
        opt._update_parameters(second, [6.0, 2.0])

        assert opt.best_fitness == 6.0
        np.testing.assert_array_equal(opt.best_candidate, second[0])
        assert opt.best_mechanism_idx == 0

    def test_best_candidate_is_a_copy_of_the_population_row(self, es_factory):
        opt = es_factory(population=2)
        population = np.array([[0.2], [0.8]], dtype=np.float32)

        opt._update_parameters(population, [1.0, 2.0])
        population[1, 0] = 0.0

        assert opt.best_candidate[0] == pytest.approx(0.8)


@pytest.mark.unit
class TestUpdateValidation:
    def test_population_must_be_two_dimensional(self, es_factory):
        with pytest.raises(ValueError, match="population must be a 2D array"):
            es_factory()._update_parameters(np.zeros(4), [1, 2, 3, 4])

    def test_population_and_fitness_counts_must_match(self, es_factory):
        with pytest.raises(ValueError, match="do not match"):
            es_factory()._update_parameters(np.zeros((4, 1)), [1, 2, 3])

    def test_population_width_must_match_the_dimension(self, es_factory):
        with pytest.raises(ValueError, match="do not match"):
            es_factory({"quota": unit_box(2)})._update_parameters(
                np.zeros((4, 1)), [1, 2, 3, 4]
            )

    def test_fitness_must_not_be_empty(self, es_factory):
        with pytest.raises(ValueError, match="must not be empty"):
            es_factory()._update_parameters(np.zeros((0, 1)), [])

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_fitness_must_be_finite(self, es_factory, bad):
        with pytest.raises(ValueError, match="fitness_scores must all be finite"):
            es_factory()._update_parameters(np.full((4, 1), 0.5), [1, 2, bad, 4])

    def test_a_rejected_update_leaves_the_state_untouched(self, es_factory):
        opt = es_factory()

        with pytest.raises(ValueError):
            opt._update_parameters(np.full((4, 1), 0.5), [1, 2, np.nan, 4])

        np.testing.assert_array_equal(opt.mean, [0.5])
        assert opt.best_fitness == -float("inf")
        assert opt.previous_population_mean_fitness is None


@pytest.mark.unit
class TestSigmaAdaptation:
    def make(self, es_factory, **training):
        defaults = {
            "sigma": 0.2,
            "sigma_lr": 1.0,
            "sigma_decay": 0.5,
            "min_sigma": 0.01,
            "max_sigma": 0.5,
        }

        return es_factory(population=4, **{**defaults, **training})

    def test_the_first_generation_only_records_the_reference(self, es_factory):
        opt = self.make(es_factory)

        assert opt._update_sigma(1.0) == "initialized"
        assert opt.sigma == pytest.approx(0.2)
        assert opt.previous_population_mean_fitness == 1.0

    def test_contract_expand_hold_with_a_full_learning_rate(self, es_factory):
        opt = self.make(es_factory)
        opt.previous_population_mean_fitness = 1.0

        assert opt._update_sigma(2.0) == "contracted"
        assert opt.sigma == pytest.approx(0.1)  # 0.2 * 0.5
        assert opt._update_sigma(0.5) == "expanded"
        assert opt.sigma == pytest.approx(0.2)  # 0.1 / 0.5
        assert opt._update_sigma(0.5) == "held"
        assert opt.sigma == pytest.approx(0.2)
        assert opt.previous_population_mean_fitness == 0.5

    def test_the_learning_rate_is_the_exponent_of_the_decay_factor(self, es_factory):
        # decay 0.81 ** lr 0.5 = 0.9
        opt = self.make(es_factory, sigma_decay=0.81, sigma_lr=0.5)
        opt.previous_population_mean_fitness = 0.0

        opt._update_sigma(10.0)
        assert opt.sigma == pytest.approx(0.2 * 0.9)

        opt._update_sigma(0.0)
        assert opt.sigma == pytest.approx(0.2)

    def test_a_zero_learning_rate_freezes_sigma(self, es_factory):
        opt = self.make(es_factory, sigma_lr=0.0)
        opt.previous_population_mean_fitness = 0.0

        assert opt._update_sigma(10.0) == "contracted"
        assert opt._update_sigma(-10.0) == "expanded"
        assert opt.sigma == pytest.approx(0.2)

    def test_a_unit_decay_freezes_sigma(self, es_factory):
        opt = self.make(es_factory, sigma_decay=1.0)
        opt.previous_population_mean_fitness = 0.0

        opt._update_sigma(10.0)
        opt._update_sigma(-10.0)

        assert opt.sigma == pytest.approx(0.2)

    @pytest.mark.parametrize(
        "previous, current, action",
        [
            (1.0, 1.0005, "held"),  # below 1e-3 * max(1, |f|)
            (1.0, 1.002, "contracted"),
            (1.0, 0.998, "expanded"),
            (1000.0, 1000.5, "held"),  # tolerance scales to 1.0
            (1000.0, 1002.0, "contracted"),
            (0.0, 0.0005, "held"),  # tolerance floor is 1e-3
        ],
    )
    def test_changes_within_the_relative_tolerance_are_ignored(
        self, es_factory, previous, current, action
    ):
        opt = self.make(es_factory)
        opt.previous_population_mean_fitness = previous

        assert opt._update_sigma(current) == action

    def test_sigma_never_leaves_its_bounds(self, es_factory):
        opt = self.make(es_factory, sigma=0.2, min_sigma=0.15, max_sigma=0.25)
        opt.previous_population_mean_fitness = 0.0

        opt._update_sigma(10.0)
        assert opt.sigma == pytest.approx(0.15)  # 0.1 clipped up

        opt._update_sigma(0.0)
        opt._update_sigma(-10.0)
        assert opt.sigma == pytest.approx(0.25)  # 0.3 clipped down

    def test_a_decay_factor_that_underflows_to_zero_still_gives_a_valid_sigma(
        self, es_factory
    ):
        opt = self.make(es_factory, sigma_decay=1e-200, sigma_lr=2.0)
        opt.previous_population_mean_fitness = 0.0

        opt._update_sigma(10.0)
        assert opt.sigma == pytest.approx(0.01)  # factor 0: contracts to the floor

        opt._update_sigma(0.0)
        assert opt.sigma == pytest.approx(0.5)  # expansion by 1 / 0 goes to the cap

    def test_update_parameters_adapts_sigma_from_the_population_mean(self, es_factory):
        opt = self.make(es_factory, sigma_lr=1.0, sigma_decay=0.5)
        eps = [[1.0], [-1.0], [0.4], [-0.4]]
        better = candidates(0.0, 0.2, eps)

        opt._update_parameters(better, [1.0, 2.0, 1.0, 2.0])  # mean 1.5: initialises
        assert opt.sigma == pytest.approx(0.2)
        opt._update_parameters(better, [3.0, 4.0, 3.0, 4.0])  # mean 3.5: contracts

        assert opt.sigma == pytest.approx(0.1)
        assert opt.previous_population_mean_fitness == pytest.approx(3.5)

    def test_the_update_logs_the_action_it_took(self, es_factory, caplog):
        opt = self.make(es_factory)
        opt.previous_population_mean_fitness = 1.0

        with caplog.at_level("INFO", logger="core.optimizers.es.optimizer"):
            opt._update_sigma(2.0)

        assert "SIGMA UPDATE | action=contracted" in caplog.text
        assert "sigma=0.200000->0.100000" in caplog.text


@pytest.mark.unit
class TestSingleCandidateUpdate:
    """The (1+1)-ES: accept improvements only, expand on success, shrink on failure."""

    def make(self, es_factory, **training):
        defaults = {"sigma": 0.2, "sigma_lr": 0.5, "sigma_decay": 0.25}

        return es_factory(
            {"quota": unit_box(2)}, population=1, **{**defaults, **training}
        )

    def candidate(self, value):
        return np.array([value, value], dtype=np.float32)

    def test_the_first_candidate_becomes_the_parent(self, es_factory):
        opt = self.make(es_factory)

        opt._update_parameters(self.candidate(0.3)[None, :], [1.5])

        np.testing.assert_allclose(opt.mean, [0.3, 0.3])
        assert opt.fitness_baseline == 1.5
        assert opt.best_fitness == 1.5
        assert opt.best_mechanism_idx == 0
        assert opt.previous_population_mean_fitness == 1.5
        assert opt.sigma == pytest.approx(0.2)

    def test_a_worse_offspring_is_rejected_and_sigma_contracts(self, es_factory):
        # factor = 0.25 ** 0.5 = 0.5
        opt = self.make(es_factory)
        opt._update_parameters(self.candidate(0.3)[None, :], [1.0])

        opt._update_parameters(self.candidate(0.9)[None, :], [0.5])

        np.testing.assert_allclose(opt.mean, [0.3, 0.3])
        assert opt.fitness_baseline == 1.0
        assert opt.best_fitness == 1.0
        assert opt.sigma == pytest.approx(0.1)

    def test_an_equal_offspring_is_rejected(self, es_factory):
        opt = self.make(es_factory)
        opt._update_parameters(self.candidate(0.3)[None, :], [1.0])

        opt._update_parameters(self.candidate(0.9)[None, :], [1.0])

        np.testing.assert_allclose(opt.mean, [0.3, 0.3])
        assert opt.sigma == pytest.approx(0.1)

    def test_a_better_offspring_is_accepted_and_sigma_expands(self, es_factory):
        opt = self.make(es_factory)
        opt._update_parameters(self.candidate(0.3)[None, :], [1.0])

        opt._update_parameters(self.candidate(0.9)[None, :], [2.0])

        np.testing.assert_allclose(opt.mean, [0.9, 0.9])
        assert opt.fitness_baseline == 2.0
        assert opt.best_fitness == 2.0
        assert opt.previous_population_mean_fitness == 2.0
        assert opt.sigma == pytest.approx(0.4)  # 0.2 / 0.5

    def test_sigma_expansion_is_capped(self, es_factory):
        opt = self.make(es_factory, sigma=0.4, max_sigma=0.5)
        opt._update_parameters(self.candidate(0.3)[None, :], [1.0])

        opt._update_parameters(self.candidate(0.9)[None, :], [2.0])

        assert opt.sigma == pytest.approx(0.5)

    def test_sigma_contraction_is_floored(self, es_factory):
        opt = self.make(es_factory, sigma=0.012, min_sigma=0.01)
        opt._update_parameters(self.candidate(0.3)[None, :], [1.0])

        opt._update_parameters(self.candidate(0.9)[None, :], [0.0])

        assert opt.sigma == pytest.approx(0.01)

    def test_a_zero_learning_rate_keeps_sigma_fixed(self, es_factory):
        opt = self.make(es_factory, sigma_lr=0.0)
        opt._update_parameters(self.candidate(0.3)[None, :], [1.0])

        opt._update_parameters(self.candidate(0.9)[None, :], [0.0])
        opt._update_parameters(self.candidate(0.6)[None, :], [3.0])

        assert opt.sigma == pytest.approx(0.2)
        assert opt.previous_population_mean_fitness == 3.0

    def test_the_best_ever_candidate_is_kept_when_the_parent_is_replaced(
        self, es_factory
    ):
        opt = self.make(es_factory)
        opt._update_parameters(self.candidate(0.3)[None, :], [1.0])
        opt._update_parameters(self.candidate(0.9)[None, :], [2.0])

        opt._update_parameters(self.candidate(0.5)[None, :], [1.5])

        np.testing.assert_allclose(opt.mean, [0.9, 0.9])
        np.testing.assert_allclose(opt.best_candidate, [0.9, 0.9])

    def test_a_non_finite_fitness_is_refused_by_the_direct_update(self, es_factory):
        opt = self.make(es_factory)

        with pytest.raises(ValueError, match="Single-candidate fitness must be finite"):
            opt._update_single_candidate(self.candidate(0.3), float("nan"))

    def test_updates_are_logged(self, es_factory, caplog):
        opt = self.make(es_factory)

        with caplog.at_level("INFO", logger="core.optimizers.es.optimizer"):
            opt._update_parameters(self.candidate(0.3)[None, :], [1.0])
            opt._update_parameters(self.candidate(0.9)[None, :], [2.0])

        assert "SINGLE INITIALIZED" in caplog.text
        assert "SINGLE UPDATE | accepted=True" in caplog.text


@pytest.mark.unit
class TestFixedModeUpdate:
    def make(self, es_factory, population=3):
        return es_factory({"penalty": unit_box(0)}, population=population)

    def test_a_batch_records_its_best_and_its_mean(self, es_factory):
        opt = self.make(es_factory)

        opt._update_parameters(np.empty((3, 0)), [1.0, 3.0, 2.0])

        assert opt.best_fitness == 3.0
        assert opt.best_mechanism_idx == 1
        assert opt.best_candidate.shape == (0,)
        assert opt.fitness_baseline == pytest.approx(2.0)
        assert opt.previous_population_mean_fitness == pytest.approx(2.0)

    def test_a_worse_batch_keeps_the_best_but_updates_the_mean(self, es_factory):
        opt = self.make(es_factory, population=2)
        opt._update_parameters(np.empty((2, 0)), [1.0, 5.0])

        opt._update_parameters(np.empty((2, 0)), [3.0, 2.0])

        assert opt.best_fitness == 5.0
        assert opt.best_mechanism_idx == 1
        assert opt.fitness_baseline == pytest.approx(2.5)

    def test_the_batch_is_logged(self, es_factory, caplog):
        opt = self.make(es_factory, population=2)

        with caplog.at_level("INFO", logger="core.optimizers.es.optimizer"):
            opt._update_parameters(np.empty((2, 0)), [1.0, 3.0])

        assert "FIXED MECHANISM BATCH" in caplog.text

    def test_sigma_and_mean_are_not_touched(self, es_factory):
        opt = self.make(es_factory)
        sigma = opt.sigma

        opt._update_parameters(np.empty((3, 0)), [1.0, 3.0, 2.0])
        opt._update_parameters(np.empty((3, 0)), [9.0, 3.0, 2.0])

        assert opt.sigma == sigma
        assert opt.mean.shape == (0,)


@pytest.mark.unit
class TestSearchDynamics:
    """Whole searches on known landscapes, with a fixed seed."""

    @staticmethod
    def search(opt, target, generations):
        for _ in range(generations):
            population = opt._sample_population()
            fitness = -np.sum((population - target) ** 2, axis=1)
            opt._update_parameters(population, fitness)

    @pytest.mark.parametrize("break_symmetry", [False, True])
    def test_the_mean_converges_on_a_quadratic_optimum(
        self, es_factory, break_symmetry
    ):
        target = np.array([0.8, 0.2, 0.6], dtype=np.float32)
        opt = es_factory(
            {"quota": unit_box(3)},
            population=32,
            seed=0,
            sigma=0.25,
            mean_lr=0.05,
            sigma_lr=0.1,
            break_symmetry=break_symmetry,
        )
        start = np.linalg.norm(opt.mean - target)

        self.search(opt, target, 200)

        assert np.linalg.norm(opt.mean - target) < 0.2 * start

    def test_the_mean_converges_next_to_the_boundary_without_leaving_the_cube(
        self, es_factory
    ):
        target = np.array([0.001, 0.999, 0.5], dtype=np.float32)
        opt = es_factory(
            {"quota": unit_box(3)},
            population=40,
            seed=0,
            sigma=0.3,
            mean_lr=0.05,
            sigma_lr=0.15,
        )

        self.search(opt, target, 300)

        assert np.all((opt.mean > 0.0) & (opt.mean < 1.0))
        assert opt.mean[0] < 0.2 and opt.mean[1] > 0.8

    def test_sigma_stays_between_its_bounds_during_a_search(self, es_factory):
        opt = es_factory(
            {"quota": unit_box(3)},
            population=30,
            seed=1,
            sigma=0.25,
            mean_lr=0.03,
            sigma_lr=0.2,
            min_sigma=0.05,
            max_sigma=0.3,
        )
        sigmas = []

        for _ in range(100):
            self.search(opt, np.array([0.7, 0.3, 0.6]), 1)
            sigmas.append(opt.sigma)

        assert all(0.05 <= s <= 0.3 for s in sigmas)
        assert len(set(np.round(sigmas, 6))) > 1  # it did adapt

    def test_the_one_plus_one_search_climbs_to_the_optimum(self, es_factory):
        target = np.array([0.8, 0.2], dtype=np.float32)
        opt = es_factory(
            {"quota": unit_box(2)}, population=1, seed=3, sigma=0.2, sigma_lr=0.2
        )
        start = np.linalg.norm(opt.mean - target)

        self.search(opt, target, 300)

        assert np.linalg.norm(opt.mean - target) < 0.1 * start
        assert opt.best_fitness > -0.01
