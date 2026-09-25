"""Synthetic CPU example, not a training run or an experiment reproduction."""

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from taylormix import (
    TaylorMixSelector, cosine_similarity, domain_costs, make_projection,
    residual_sketches, update_gram,
)


def main():
    generator = np.random.default_rng(7)
    domains = {"math": list(range(8)), "code": list(range(8, 16)), "general": list(range(16, 24))}
    anchors = [0, 8, 16]
    representations = generator.normal(size=(24, 12))
    prototypes = np.stack([representations[values].mean(axis=0) for values in domains.values()])
    target = generator.normal(size=12)
    projection = make_projection(12, 4, seed=11)
    sketches = np.empty((24, 4))
    for domain_index, values in enumerate(domains.values()):
        sketches[values] = residual_sketches(representations[values], prototypes[domain_index], projection)
    gram = update_gram(np.zeros((4, 4)), sketches[anchors], decay=0.8)
    costs = domain_costs(
        cosine_similarity(prototypes, target), [0.3, 0.6, 0.4], epsilon=1e-8, clip=3.0
    )
    relevance = cosine_similarity(representations, target)
    learnability = generator.uniform(size=24)
    selector = TaylorMixSelector(
        domains, 6, allocation_temperature=1.0, sample_temperature=0.7,
        score_weights=[0.4, 0.3, 0.3], epsilon=1e-8, clip=3.0,
        seed=19, excluded_ids=anchors, window_size=5,
    )
    result = selector.select(
        costs, [1, 1, 1], relevance=relevance, learnability=learnability,
        sketches=sketches, gram=gram,
    )
    assert len(result["indices"]) == len(set(result["indices"])) == 6
    assert set(result["indices"]).isdisjoint(anchors)
    restored = TaylorMixSelector(
        domains, 6, allocation_temperature=1.0, sample_temperature=0.7,
        score_weights=[0.4, 0.3, 0.3], epsilon=1e-8, clip=3.0,
        seed=0, excluded_ids=anchors, window_size=5,
    )
    restored.load_state_dict(json.loads(json.dumps(selector.state_dict())))
    assert restored.state_dict() == selector.state_dict()
    print(json.dumps({"kind": "synthetic-selector-demo", **result}, indent=2))


if __name__ == "__main__":
    main()
