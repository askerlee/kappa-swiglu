import math

import pytest
import torch

from scripts.eval_kappa_effect import compute_routing_metrics


def test_compute_routing_metrics_uniform_assignments_and_probabilities():
    top_k_indices = torch.tensor([[0, 1], [2, 3]])
    all_router_probs = torch.full((2, 4), 0.25)

    utilization, load_entropy, load_cv, router_entropy = compute_routing_metrics(
        top_k_indices,
        all_router_probs,
        n_exp=4,
    )

    torch.testing.assert_close(utilization, torch.full((4,), 0.25, dtype=torch.float64))
    assert load_entropy.item() == pytest.approx(1.0)
    assert load_cv.item() == pytest.approx(0.0)
    assert router_entropy.item() == pytest.approx(math.log(4))


def test_compute_routing_metrics_detects_concentrated_assignments_and_confidence():
    top_k_indices = torch.tensor([[0, 1], [0, 1]])
    all_router_probs = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])

    utilization, load_entropy, load_cv, router_entropy = compute_routing_metrics(
        top_k_indices,
        all_router_probs,
        n_exp=4,
    )

    torch.testing.assert_close(utilization, torch.tensor([0.5, 0.5, 0.0, 0.0], dtype=torch.float64))
    assert load_entropy.item() == pytest.approx(0.5)
    assert load_cv.item() == pytest.approx(1.0)
    assert router_entropy.item() == pytest.approx(0.0)