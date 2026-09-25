"""NumPy reference for TaylorMix Method Eqs. (4)-(15) and Algorithm 1."""

from copy import deepcopy
from numbers import Integral

import numpy as np


def _array(values, name, ndim=None):
    result = np.asarray(values, dtype=np.float64)
    if not result.size or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be nonempty and finite.")
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} has an invalid number of dimensions.")
    return result


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive.")
    return value


def _gram(matrix):
    matrix = _array(matrix, "gram", 2)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("gram must be square.")
    tolerance = 1e-10 * max(1.0, float(np.max(np.abs(matrix))))
    if not np.allclose(matrix, matrix.T, atol=tolerance, rtol=0):
        raise ValueError("gram must be symmetric.")
    matrix = (matrix + matrix.T) / 2
    if np.linalg.eigvalsh(matrix).min() < -tolerance:
        raise ValueError("gram must be positive semidefinite.")
    return matrix


def normalize(values, *, epsilon, clip):
    """Eq. (6), using population standard deviation on one comparison set."""
    values = _array(values, "values", 1)
    epsilon = _positive(epsilon, "epsilon")
    clip = _positive(clip, "clip")
    deviation = float(np.std(values))
    if deviation < epsilon:
        return np.zeros_like(values)
    return np.clip((values - np.mean(values)) / (deviation + epsilon), -clip, clip)


def cosine_similarity(representations, target):
    """Row-wise cosine for Eqs. (4) and (12); zero vectors are rejected."""
    representations = _array(representations, "representations", 2)
    target = _array(target, "target", 1)
    if representations.shape[1] != target.size:
        raise ValueError("Representation and target dimensions must agree.")
    norms = np.linalg.norm(representations, axis=1) * np.linalg.norm(target)
    if np.any(norms == 0):
        raise ValueError("Cosine similarity requires nonzero vectors.")
    return representations @ target / norms


def domain_costs(raw_alignment, feedback_cost, *, epsilon, clip):
    """Eq. (7); feedback_cost is the caller's psi_cost(reward EMA)."""
    alignment = _array(raw_alignment, "raw_alignment", 1)
    feedback = _array(feedback_cost, "feedback_cost", 1)
    if alignment.shape != feedback.shape:
        raise ValueError("Domain signal shapes must agree.")
    return -normalize(alignment, epsilon=epsilon, clip=clip) + normalize(
        feedback, epsilon=epsilon, clip=clip
    )


def softmax(scores, temperature):
    """Stable Gibbs probabilities, without an artificial score or mass floor."""
    scores = _array(scores, "scores", 1)
    temperature = _positive(temperature, "temperature")
    with np.errstate(over="ignore", under="ignore"):
        weights = np.exp((scores - np.max(scores)) / temperature)
    return weights / weights.sum()


def allocate_domains(costs, reference, temperature):
    """Eq. (8); reject unrepresentable full support instead of adding a floor."""
    costs = _array(costs, "costs", 1)
    reference = _array(reference, "reference", 1)
    temperature = _positive(temperature, "temperature")
    if costs.shape != reference.shape or np.any(reference <= 0):
        raise ValueError("reference must have one positive weight per domain.")
    with np.errstate(over="ignore"):
        logits = np.log(reference) - (costs - np.min(costs)) / temperature
    if not np.all(np.isfinite(logits)):
        raise FloatingPointError("Domain logits exceed floating-point range.")
    shares = softmax(logits, 1.0)
    if np.any(shares == 0):
        raise FloatingPointError("Domain share underflow; increase the allocation temperature.")
    return shares


def largest_remainder(shares, budget):
    """Eq. (9); exact fractional ties follow input domain order."""
    shares = _array(shares, "shares", 1)
    budget = _integer(budget, "budget")
    if np.any(shares < 0) or shares.max() <= 0:
        raise ValueError("shares must be nonnegative with positive total mass.")
    shares = shares / shares.max()
    targets = budget * (shares / shares.sum())
    counts = np.floor(targets).astype(np.int64)
    remainder = budget - int(counts.sum())
    order = np.argsort(-(targets - counts), kind="stable")
    counts[order[:remainder]] += 1
    return counts


def allocate_quotas(shares, budget, capacities):
    """Round, cap at eligible supply, and redistribute with the original shares."""
    shares = _array(shares, "shares", 1)
    capacities = np.asarray([_integer(value, "capacity") for value in capacities])
    budget = _integer(budget, "budget")
    if shares.shape != capacities.shape or np.any(shares <= 0):
        raise ValueError("One positive share and one capacity are required per domain.")
    if int(capacities.sum()) < budget:
        raise ValueError("Total eligible supply is smaller than the batch budget.")
    counts = np.minimum(largest_remainder(shares, budget), capacities)
    while int(counts.sum()) < budget:
        available = np.flatnonzero(counts < capacities)
        additions = largest_remainder(shares[available], budget - int(counts.sum()))
        counts[available] += np.minimum(additions, capacities[available] - counts[available])
    return counts


def make_projection(hidden_dimension, sketch_dimension, *, seed):
    """Eq. (10): sample P once, with entry variance 1 / hidden_dimension."""
    hidden_dimension = _integer(hidden_dimension, "hidden_dimension", 1)
    sketch_dimension = _integer(sketch_dimension, "sketch_dimension", 1)
    return np.random.default_rng(seed).normal(
        scale=1 / np.sqrt(hidden_dimension), size=(hidden_dimension, sketch_dimension)
    )


def residual_sketches(representations, domain_prototype, projection):
    """Eq. (10): subtract the domain prototype, not the target prototype."""
    representations = _array(representations, "representations", 2)
    prototype = _array(domain_prototype, "domain_prototype", 1)
    projection = _array(projection, "projection", 2)
    if representations.shape[1] != prototype.size or projection.shape[0] != prototype.size:
        raise ValueError("Representation, prototype, and projection dimensions must agree.")
    return (representations - prototype) @ projection


def ema(previous, observed, decay):
    """Explicit EMA convention: decay * previous + (1 - decay) * observed."""
    previous = _array(previous, "previous")
    observed = _array(observed, "observed")
    decay = float(decay)
    if previous.shape != observed.shape or not np.isfinite(decay) or not 0 <= decay <= 1:
        raise ValueError("EMA shapes must agree and decay must lie in [0, 1].")
    return decay * previous + (1 - decay) * observed


def update_gram(previous, anchor_sketches, decay):
    """Eq. (11); the caller supplies only excluded shadow-anchor sketches."""
    previous = _gram(previous)
    anchors = _array(anchor_sketches, "anchor_sketches", 2)
    if anchors.shape[1] != previous.shape[0]:
        raise ValueError("Anchor and Gram dimensions must agree.")
    return ema(previous, anchors.T @ anchors / len(anchors), decay)


def sketch_risk(sketches, gram):
    """Unnormalized diagonal risk from Eq. (11)."""
    sketches = _array(sketches, "sketches", 2)
    gram = _gram(gram)
    if sketches.shape[1] != gram.shape[0]:
        raise ValueError("Sketch and Gram dimensions must agree.")
    return np.einsum("ij,ij->i", sketches @ gram, sketches)


def candidate_scores(relevance, learnability, risk, weights, *, epsilon, clip):
    """Eq. (13); freeze each coordinate's normalization for the entire window."""
    relevance = _array(relevance, "relevance", 1)
    learnability = _array(learnability, "learnability", 1)
    risk = _array(risk, "risk", 1)
    weights = _array(weights, "weights", 1)
    if relevance.shape != learnability.shape or relevance.shape != risk.shape:
        raise ValueError("Candidate coordinate shapes must agree.")
    if weights.shape != (3,) or np.any(weights < 0) or not np.isclose(weights.sum(), 1):
        raise ValueError("Three nonnegative score weights summing to one are required.")
    coordinates = [
        normalize(values, epsilon=epsilon, clip=clip)
        for values in (relevance, learnability, risk)
    ]
    return weights[0] * coordinates[0] + weights[1] * coordinates[1] - weights[2] * coordinates[2]


def finite_step_scores(scores, sketches, gram, prefix_mean, prefix_size):
    """Method conditional Gamma, including the first-draw diagonal correction."""
    scores = _array(scores, "scores", 1)
    sketches = _array(sketches, "sketches", 2)
    gram = _gram(gram)
    mean = _array(prefix_mean, "prefix_mean", 1)
    prefix_size = _integer(prefix_size, "prefix_size")
    if sketches.shape != (scores.size, mean.size) or gram.shape != (mean.size, mean.size):
        raise ValueError("Score, sketch, Gram, and prefix dimensions must agree.")
    if prefix_size == 0 and np.any(mean != 0):
        raise ValueError("An empty prefix must have a zero mean.")
    projected = sketches @ gram
    risk = np.einsum("ij,ij->i", projected, sketches)
    return scores - prefix_size / (prefix_size + 1) * (projected @ mean) - risk / (
        2 * (prefix_size + 1)
    )


class TaylorMixSelector:
    """Persistent ID pools; no model loading, rollout, optimizer, or filesystem IO."""

    def __init__(
        self, domain_ids, batch_size, *, allocation_temperature, sample_temperature,
        score_weights, epsilon, clip, seed=0, excluded_ids=(), window_size=None,
    ):
        if not domain_ids or any(not isinstance(name, str) or not name for name in domain_ids):
            raise ValueError("At least one named domain is required.")
        self.batch_size = _integer(batch_size, "batch_size", 1)
        self.allocation_temperature = _positive(allocation_temperature, "allocation_temperature")
        self.sample_temperature = _positive(sample_temperature, "sample_temperature")
        self.epsilon = _positive(epsilon, "epsilon")
        self.clip = _positive(clip, "clip")
        self.score_weights = _array(score_weights, "score_weights", 1).copy()
        candidate_scores([0], [0], [0], self.score_weights, epsilon=epsilon, clip=clip)
        self.window_size = None if window_size is None else _integer(window_size, "window_size", 1)
        self.domains = list(domain_ids)
        excluded = {_integer(value, "excluded ID") for value in excluded_ids}
        all_ids = [_integer(value, "case ID") for values in domain_ids.values() for value in values]
        if len(set(all_ids)) != len(all_ids) or not excluded.issubset(all_ids):
            raise ValueError("Case IDs must be globally unique and exclusions must be known IDs.")
        self.eligible = {
            name: [int(value) for value in values if value not in excluded]
            for name, values in domain_ids.items()
        }
        if sum(map(len, self.eligible.values())) < self.batch_size:
            raise ValueError("Insufficient eligible supply after anchor exclusion.")
        self.rng = np.random.default_rng(seed)
        self.remaining = {
            name: self.rng.permutation(values).astype(int).tolist()
            for name, values in self.eligible.items()
        }

    def state_dict(self):
        """JSON-compatible pool/RNG snapshot; caller separately checkpoints model/cache state."""
        return deepcopy({
            "schema": 1,
            "domains": self.domains,
            "eligible": self.eligible,
            "config": {
                "batch_size": self.batch_size,
                "allocation_temperature": self.allocation_temperature,
                "sample_temperature": self.sample_temperature,
                "score_weights": self.score_weights.tolist(),
                "epsilon": self.epsilon,
                "clip": self.clip,
                "window_size": self.window_size,
            },
            "remaining": self.remaining,
            "rng": self.rng.bit_generator.state,
        })

    def load_state_dict(self, state):
        state = deepcopy(state)
        current = self.state_dict()
        for field in ("schema", "domains", "eligible", "config"):
            if state.get(field) != current[field]:
                raise ValueError(f"Incompatible selector state: {field}.")
        pools = state.get("remaining", {})
        if set(pools) != set(self.domains):
            raise ValueError("State must contain every domain pool.")
        for name, values in pools.items():
            values = [_integer(value, "case ID") for value in values]
            if len(set(values)) != len(values) or not set(values).issubset(self.eligible[name]):
                raise ValueError("State contains duplicate or ineligible IDs.")
            pools[name] = values
        generator = np.random.default_rng()
        generator.bit_generator.state = state["rng"]
        self.remaining = pools
        self.rng = generator

    def select(self, costs, reference, *, relevance, learnability, sketches, gram):
        """Algorithm 1: consume cached coordinates and return a shuffled ordinary ID list."""
        gram = _gram(gram).copy()
        shares = allocate_domains(costs, reference, self.allocation_temperature)
        quotas = allocate_quotas(
            shares, self.batch_size, [len(self.eligible[name]) for name in self.domains]
        )
        rng_state = deepcopy(self.rng.bit_generator.state)
        pools = deepcopy(self.remaining)
        selected_by_domain = {}
        windows = {}
        reset_domains = []
        try:
            for name, quota in zip(self.domains, quotas):
                quota = int(quota)
                selected_by_domain[name] = []
                windows[name] = []
                if quota == 0:
                    continue
                if len(pools[name]) < min(quota, len(self.eligible[name])):
                    pools[name] = self.rng.permutation(self.eligible[name]).astype(int).tolist()
                    reset_domains.append(name)
                width = len(pools[name]) if self.window_size is None else max(self.window_size, quota)
                candidates = pools[name][:width]
                windows[name] = candidates.copy()
                candidate_sketches = _array([sketches[value] for value in candidates], "sketches", 2).copy()
                if candidate_sketches.shape[1] != gram.shape[0]:
                    raise ValueError("Sketch and Gram dimensions must agree.")
                projected = candidate_sketches @ gram
                risk = np.einsum("ij,ij->i", projected, candidate_sketches)
                scores = candidate_scores(
                    [relevance[value] for value in candidates],
                    [learnability[value] for value in candidates],
                    risk, self.score_weights, epsilon=self.epsilon, clip=self.clip,
                )
                if quota == len(candidates):
                    chosen = candidates.copy()
                else:
                    available = list(range(len(candidates)))
                    chosen = []
                    mean = np.zeros(gram.shape[0])
                    for prefix_size in range(quota):
                        conditional = (
                            scores[available]
                            - prefix_size / (prefix_size + 1) * (projected[available] @ mean)
                            - risk[available] / (2 * (prefix_size + 1))
                        )
                        probabilities = softmax(conditional, self.sample_temperature)
                        position = available.pop(int(self.rng.choice(len(available), p=probabilities)))
                        chosen.append(candidates[position])
                        mean = (prefix_size * mean + candidate_sketches[position]) / (prefix_size + 1)
                selected_by_domain[name] = chosen
                selected_set = set(chosen)
                pools[name] = [value for value in pools[name] if value not in selected_set]
            indices = [value for name in self.domains for value in selected_by_domain[name]]
            self.rng.shuffle(indices)
        except Exception:
            self.rng.bit_generator.state = rng_state
            raise
        self.remaining = pools
        return {
            "indices": indices,
            "shares": dict(zip(self.domains, shares.tolist())),
            "quotas": dict(zip(self.domains, quotas.tolist())),
            "selected_by_domain": selected_by_domain,
            "candidate_ids": windows,
            "reset_domains": reset_domains,
        }
