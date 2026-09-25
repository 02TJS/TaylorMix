"""Synthetic Method/Algorithm 1 checks, not evidence of training replication."""

import json

import numpy as np
import pytest

import taylormix as reference


def make_selector(domains=None, budget=3, **overrides):
    options = {
        "allocation_temperature": 1.0, "sample_temperature": 0.7,
        "score_weights": [0.4, 0.3, 0.3], "epsilon": 1e-8, "clip": 3.0, "seed": 7,
    }
    options.update(overrides)
    return reference.TaylorMixSelector(domains or {"math": list(range(8))}, budget, **options)


def inputs(total=24, dimension=3):
    generator = np.random.default_rng(17)
    return {
        "relevance": generator.normal(size=total),
        "learnability": generator.uniform(size=total),
        "sketches": generator.normal(size=(total, dimension)),
        "gram": np.eye(dimension),
    }


def test_normalization_domain_cost_and_degenerate_signals():
    signals = np.array([1.0, 2.0, 7.0])
    expected = np.clip((signals - signals.mean()) / (signals.std() + 1e-8), -1, 1)
    np.testing.assert_allclose(reference.normalize(signals, epsilon=1e-8, clip=1), expected)
    np.testing.assert_array_equal(reference.normalize([1, 1, 1], epsilon=1e-8, clip=3), 0)
    np.testing.assert_array_equal(reference.normalize([1, 1 + 1e-10], epsilon=1e-8, clip=3), 0)
    np.testing.assert_allclose(
        reference.domain_costs(signals, [2, 2, 2], epsilon=1e-8, clip=1), -expected
    )


@pytest.mark.parametrize("temperature", [0.1, 0.7, 10.0])
def test_kl_closed_form_translation_and_objective_gap(temperature):
    costs = np.array([-0.4, 0.2, 0.7])
    prior = np.array([0.2, 0.3, 0.5])
    shares = reference.allocate_domains(costs, prior, temperature)
    expected = prior * np.exp(-costs / temperature)
    expected /= expected.sum()
    np.testing.assert_allclose(shares, expected)
    np.testing.assert_allclose(reference.allocate_domains(costs + 123, prior, temperature), shares)
    assert np.all(shares > 0)
    challenger = np.array([0.15, 0.25, 0.6])
    objective = lambda mixture: mixture @ costs + temperature * np.sum(mixture * np.log(mixture / prior))
    gap = temperature * np.sum(challenger * np.log(challenger / shares))
    assert objective(challenger) - objective(shares) == pytest.approx(gap)


def test_no_added_floor_or_integer_coverage_guarantee():
    shares = reference.allocate_domains([0, 8, 9], [1, 1, 1], 1)
    assert shares[1] < 0.001
    np.testing.assert_array_equal(reference.allocate_quotas(shares, 4, [10, 10, 10]), [4, 0, 0])
    with pytest.raises(FloatingPointError):
        reference.allocate_domains([0, 10000], [1, 1], 1)


def test_rounding_ties_and_repeated_undersupply_redistribution():
    np.testing.assert_array_equal(reference.largest_remainder([1, 1, 1], 2), [1, 1, 0])
    np.testing.assert_array_equal(reference.allocate_quotas([0.8, 0.15, 0.05], 10, [1, 2, 20]), [1, 2, 7])
    np.testing.assert_array_equal(reference.allocate_quotas([0.8, 0.15, 0.05], 4, [0, 0, 9]), [0, 0, 4])
    with pytest.raises(ValueError):
        reference.allocate_quotas([1, 1], 5, [2, 2])


def test_randomized_quota_conservation_and_capacity_bounds():
    generator = np.random.default_rng(42)
    for _ in range(300):
        capacities = generator.integers(0, 20, size=5)
        budget = int(generator.integers(0, int(capacities.sum()) + 1))
        shares = generator.dirichlet(np.ones(5))
        quotas = reference.allocate_quotas(shares, budget, capacities)
        assert quotas.sum() == budget
        assert np.all(quotas >= 0) and np.all(quotas <= capacities)


def test_projection_domain_residual_and_anchor_ema():
    projection = reference.make_projection(6, 3, seed=19)
    expected = np.random.default_rng(19).normal(scale=1 / np.sqrt(6), size=(6, 3))
    np.testing.assert_array_equal(projection, expected)
    representations = np.arange(24).reshape(4, 6)
    prototype = np.arange(6)
    sketches = reference.residual_sketches(representations, prototype, projection)
    np.testing.assert_allclose(sketches, (representations - prototype) @ projection)
    previous = np.eye(3)
    for decay in (0.0, 0.8, 1.0):
        gram = reference.update_gram(previous, sketches[:2], decay)
        np.testing.assert_allclose(gram, decay * previous + (1 - decay) * sketches[:2].T @ sketches[:2] / 2)
        assert np.linalg.eigvalsh(gram).min() >= -1e-10
    np.testing.assert_allclose(reference.cosine_similarity([[1, 0], [0, 1]], [1, 0]), [1, 0])
    np.testing.assert_allclose(reference.ema([1, 2], [3, 4], 0.25), [2.5, 3.5])


def test_score_is_exactly_three_normalized_coordinates():
    relevance = [0.2, 0.5, 0.9]
    learning = [0.9, 0.1, 0.4]
    risk = [1, 4, 9]
    normalized = [reference.normalize(values, epsilon=1e-8, clip=3) for values in (relevance, learning, risk)]
    expected = 0.4 * normalized[0] + 0.3 * normalized[1] - 0.3 * normalized[2]
    actual = reference.candidate_scores(relevance, learning, risk, [0.4, 0.3, 0.3], epsilon=1e-8, clip=3)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("prefix_size", [0, 1, 2, 7, 20])
def test_finite_step_matches_original_utility_not_a_fixed_redundancy_penalty(prefix_size):
    generator = np.random.default_rng(101 + prefix_size)
    factors = generator.normal(size=(4, 4))
    gram = factors.T @ factors
    sketches = generator.normal(size=(6, 4))
    scores = generator.normal(size=6)
    prefix_sketches = generator.normal(size=(prefix_size, 4))
    prefix_scores = generator.normal(size=prefix_size)
    mean = prefix_sketches.mean(axis=0) if prefix_size else np.zeros(4)
    gamma = reference.finite_step_scores(scores, sketches, gram, mean, prefix_size)
    next_mean = (prefix_size * mean + sketches) / (prefix_size + 1)
    next_value = (prefix_scores.sum() + scores) / (prefix_size + 1) - 0.5 * np.einsum(
        "ij,ij->i", next_mean @ gram, next_mean
    )
    if prefix_size == 0:
        np.testing.assert_allclose(gamma, next_value)
    else:
        old_value = prefix_scores.mean() - 0.5 * mean @ gram @ mean
        actual_gain = next_value - old_value
        np.testing.assert_allclose(
            actual_gain - actual_gain[0], (gamma - gamma[0]) / (prefix_size + 1), atol=1e-12
        )


@pytest.mark.parametrize("gram", [np.eye(3), np.zeros((3, 3)), np.ones((3, 3))])
def test_actual_draw_probabilities_freeze_scores_and_restart_domain_prefix(gram):
    domains = {"math": list(range(6)), "code": list(range(6, 12))}
    sampler = make_selector(domains, budget=6)
    sampler.remaining = {name: values.copy() for name, values in domains.items()}
    coordinates = inputs(total=12)
    coordinates["gram"] = gram
    expected_calls = []
    for ids in domains.values():
        sketches = coordinates["sketches"][ids]
        risk = np.einsum("ij,ij->i", sketches @ gram, sketches)
        scores = reference.candidate_scores(
            coordinates["relevance"][ids], coordinates["learnability"][ids], risk,
            [0.4, 0.3, 0.3], epsilon=1e-8, clip=3,
        )
        available = list(range(6))
        mean = np.zeros(3)
        for prefix_size in range(3):
            gamma = reference.finite_step_scores(
                scores[available], sketches[available], gram, mean, prefix_size
            )
            expected_calls.append(reference.softmax(gamma, 0.7))
            position = available.pop(0)
            mean = (prefix_size * mean + sketches[position]) / (prefix_size + 1)
    calls = []

    def deterministic_choice(self, size, p):
        calls.append(p.copy())
        assert size == len(p)
        return 0

    class RecordingGenerator:
        bit_generator = sampler.rng.bit_generator
        choice = deterministic_choice

        def shuffle(self, values):
            values.reverse()

    sampler.rng = RecordingGenerator()
    result = sampler.select([0, 0], [1, 1], **coordinates)
    assert result["selected_by_domain"] == {"math": [0, 1, 2], "code": [6, 7, 8]}
    assert result["indices"] == [8, 7, 6, 2, 1, 0]
    assert len(calls) == len(expected_calls)
    for actual, expected in zip(calls, expected_calls):
        np.testing.assert_allclose(actual, expected)
    assert sampler.remaining == {"math": [3, 4, 5], "code": [9, 10, 11]}


def test_softmax_exact_ties_are_uniform_and_temperature_is_respected():
    np.testing.assert_allclose(reference.softmax([5, 5, 5], 0.1), [1 / 3] * 3)
    for temperature in (0.3, 1.0, 10.0):
        probabilities = reference.softmax([1.0, 1.5], temperature)
        assert probabilities[1] / probabilities[0] == pytest.approx(np.exp(0.5 / temperature))


def test_reset_nonempty_pool_exclusion_and_quota_sized_window():
    sampler = make_selector(budget=3, excluded_ids=[0, 1], window_size=1)
    sampler.remaining["math"] = [6, 7]
    result = sampler.select([0], [1], **inputs())
    assert result["reset_domains"] == ["math"]
    assert len(result["candidate_ids"]["math"]) == 3
    assert result["selected_by_domain"]["math"] == result["candidate_ids"]["math"]
    assert set(result["indices"]).isdisjoint({0, 1})
    assert len(sampler.remaining["math"]) == 3


def test_unselected_ids_keep_order_across_steps_and_equal_supply_does_not_reset():
    sampler = make_selector(budget=2, window_size=4)
    sampler.remaining["math"] = list(range(8))
    result = sampler.select([0], [1], **inputs())
    selected = set(result["indices"])
    assert selected.issubset({0, 1, 2, 3})
    assert sampler.remaining["math"] == [value for value in range(8) if value not in selected]
    sampler.remaining["math"] = [5, 6]
    result = sampler.select([0], [1], **inputs())
    assert result["reset_domains"] == []
    assert result["selected_by_domain"]["math"] == [5, 6]
    assert sampler.remaining["math"] == []


def test_end_to_end_undersupply_is_redistributed_without_duplicate_ids():
    sampler = make_selector({"math": [0], "code": list(range(1, 10))}, budget=6)
    result = sampler.select([-3, 3], [1, 1], **inputs())
    assert result["quotas"] == {"math": 1, "code": 5}
    assert len(set(result["indices"])) == len(result["indices"]) == 6


def test_empty_domain_remains_in_allocation_but_has_zero_integer_quota():
    sampler = make_selector({"math": [], "code": list(range(8))}, budget=3)
    result = sampler.select([0, 0], [1, 1], **inputs())
    assert result["shares"]["math"] > 0
    assert result["quotas"] == {"math": 0, "code": 3}
    assert result["candidate_ids"]["math"] == []


def test_checkpoint_json_round_trip_replays_multiple_resets():
    sampler = make_selector(budget=3, window_size=4)
    sampler.select([0], [1], **inputs())
    state = json.loads(json.dumps(sampler.state_dict()))
    restored = make_selector(budget=3, window_size=4, seed=123)
    restored.load_state_dict(state)
    for _ in range(6):
        assert sampler.select([0], [1], **inputs()) == restored.select([0], [1], **inputs())
        assert sampler.state_dict() == restored.state_dict()


def test_failed_selection_and_invalid_restore_do_not_mutate_state():
    sampler = make_selector({"math": list(range(4)), "code": list(range(4, 8))}, budget=4)
    sampler.remaining["math"] = [0]
    before = sampler.state_dict()
    coordinates = inputs()
    coordinates["learnability"] = {value: 1 for value in range(4)}
    with pytest.raises(KeyError):
        sampler.select([0, 0], [1, 1], **coordinates)
    assert sampler.state_dict() == before
    bad = sampler.state_dict()
    bad["remaining"]["math"] = [0, 0]
    with pytest.raises(ValueError):
        sampler.load_state_dict(bad)
    assert sampler.state_dict() == before


@pytest.mark.parametrize("operation", [
    lambda: reference.normalize([np.nan], epsilon=1e-8, clip=3),
    lambda: reference.normalize([1], epsilon=0, clip=3),
    lambda: reference.allocate_domains([0, 1], [1, 0], 1),
    lambda: reference.allocate_domains([0, 1], [1, 1], 0),
    lambda: reference.allocate_quotas([1, 1], 2, [1.5, 2]),
    lambda: reference.allocate_quotas([1], 2, [-1]),
    lambda: reference.softmax([np.inf], 1),
    lambda: reference.cosine_similarity([[0, 0]], [1, 0]),
    lambda: reference.sketch_risk([[1, 0]], [[1, 1], [0, 1]]),
    lambda: reference.sketch_risk([[1, 0]], [[1, 0], [0, -1]]),
    lambda: reference.update_gram(np.eye(2), [], 0.9),
    lambda: reference.ema([1], [1], -0.1),
    lambda: reference.candidate_scores([1], [1], [1], [1, 1, 1], epsilon=1e-8, clip=3),
    lambda: reference.finite_step_scores([1], [[1]], [[1]], [1], 0),
    lambda: make_selector({"math": [0, 1], "code": [1, 2]}, budget=2),
    lambda: make_selector(budget=8, excluded_ids=[0]),
    lambda: make_selector(excluded_ids=[99]),
    lambda: make_selector(sample_temperature=-1),
])
def test_invalid_inputs_are_not_silently_reinterpreted(operation):
    with pytest.raises(ValueError):
        operation()
