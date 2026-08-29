"""Allocation methods and synthetic DORA weight generation for Phase D1."""

from dataclasses import dataclass

import numpy as np


def softmax(values, temperature, axis=-1):
    values = np.asarray(values, dtype=np.float64) / temperature
    values -= values.max(axis=axis, keepdims=True)
    exponentials = np.exp(values)
    return exponentials / exponentials.sum(axis=axis, keepdims=True)


def largest_remainder(probabilities, budget):
    """Hamilton/largest-remainder allocation used by the current local DORA."""
    raw = np.asarray(probabilities, dtype=np.float64) * budget
    allocation = np.floor(raw).astype(np.int64)
    remaining = budget - int(allocation.sum())
    order = np.argsort(-(raw - allocation), kind="stable")
    allocation[order[:remaining]] += 1
    return allocation


def systematic_stochastic(probabilities, budget, offset):
    """Fixed-budget unbiased systematic sampling with offset in [0, 1)."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    cumulative = np.cumsum(probabilities)
    positions = (offset + np.arange(budget, dtype=np.float64)) / budget
    indices = np.searchsorted(cumulative, positions, side="right")
    return np.bincount(indices, minlength=len(probabilities)).astype(np.int64)


@dataclass
class CumulativeDeficitAllocator:
    """Carry fractional allocation debt across rounds and repay largest deficits."""

    candidate_count: int

    def __post_init__(self):
        self.target = np.zeros(self.candidate_count, dtype=np.float64)
        self.actual = np.zeros(self.candidate_count, dtype=np.int64)

    def allocate(self, probabilities, budget):
        self.target += np.asarray(probabilities, dtype=np.float64) * budget
        allocation = np.zeros(self.candidate_count, dtype=np.int64)
        for _ in range(budget):
            deficit = self.target - (self.actual + allocation)
            allocation[int(np.argmax(deficit))] += 1
        self.actual += allocation
        return allocation


def synthetic_dora_weights(rng, candidate_count, quality_temperature, similarity_temperature):
    """Create quality-only and DORA weights using the project's actual formula.

    PRM-like scores are drawn from a mixture of high-scoring beta distributions.
    A latent cluster assignment creates redundant and distinct reasoning directions.
    """
    mixture = rng.random(candidate_count) < 0.7
    scores = np.where(
        mixture,
        rng.beta(10.0, 1.8, candidate_count),
        rng.beta(4.0, 3.0, candidate_count),
    )
    quality = softmax(scores, quality_temperature)

    cluster_count = max(2, min(candidate_count, int(np.ceil(np.sqrt(candidate_count)))))
    clusters = rng.integers(0, cluster_count, candidate_count)
    if candidate_count >= cluster_count:
        clusters[:cluster_count] = np.arange(cluster_count)
        rng.shuffle(clusters)

    similarity = np.eye(candidate_count, dtype=np.float64)
    for left in range(candidate_count):
        for right in range(left + 1, candidate_count):
            if clusters[left] == clusters[right]:
                value = rng.uniform(0.72, 0.97)
            else:
                value = rng.uniform(0.05, 0.55)
            similarity[left, right] = similarity[right, left] = value

    row_probability = softmax(similarity, similarity_temperature, axis=1)
    diversity = np.diag(row_probability)
    combined = quality * diversity
    combined /= combined.sum()
    return quality, combined


def total_variation(left, right):
    return 0.5 * float(np.abs(np.asarray(left) - np.asarray(right)).sum())


def validate_allocation(allocation, budget, candidate_count):
    allocation = np.asarray(allocation)
    assert allocation.shape == (candidate_count,)
    assert np.issubdtype(allocation.dtype, np.integer)
    assert np.all(allocation >= 0)
    assert int(allocation.sum()) == budget
