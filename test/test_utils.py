from typing import Dict, Tuple

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

import numpy as np

from task_similarity.constants import _IoUTaskSimilarityParameters
from task_similarity.utils import (
    _calculate_order,
    _get_hypervolume,
    _get_promising_pdf,
    _get_promising_pdfs,
)


def get_random_config(n_configs: int = 10) -> Tuple[CS.ConfigurationSpace, Dict[str, np.ndarray]]:
    config_space = CS.ConfigurationSpace()
    meta_info = {"lower": -2, "upper": 1}
    config_space.add_hyperparameters(
        [
            CSH.UniformFloatHyperparameter("x0", lower=-2, upper=3),
            CSH.UniformIntegerHyperparameter("x1", lower=-2, upper=2),
            CSH.OrdinalHyperparameter("x2", sequence=list(range(-2, 2)), meta=meta_info),
            CSH.CategoricalHyperparameter("x3", choices=["a", "b"]),
        ]
    )
    return config_space, {
        f"x{d}": np.array([config_space.sample_configuration()[f"x{d}"] for _ in range(n_configs)]) for d in range(4)
    }


def test_calculate_order() -> None:
    observations = {"f1": np.array([0, 1, 2, 1]), "f2": np.array([0, 1, 2, 0])}
    order = _calculate_order(
        observations=observations,
        objective_names=["f1", "f2"],
        larger_is_better_objectives=None,
    )
    assert np.allclose(order, [0, 3, 1, 2])

    observations = {"f1": np.array([0, 1, 2, 1]), "f2": -np.array([0, 1, 2, 0])}
    order = _calculate_order(
        observations=observations,
        objective_names=["f1", "f2"],
        larger_is_better_objectives=[1],
    )
    assert np.allclose(order, [0, 3, 1, 2])


def test_get_hypervolume() -> None:
    config_space, _ = get_random_config()
    hv = _get_hypervolume(config_space)
    assert hv == 5 * 4 * 3 * 2


def test_get_promising_pdf_with_resampling() -> None:
    n_configs = 10
    loss_metric = "loss"
    config_space, configs = get_random_config(n_configs)
    loss_vals = np.arange(n_configs)
    configs[loss_metric] = loss_vals

    n_resamples = 100
    params = _IoUTaskSimilarityParameters(
        n_samples=10,
        config_space=config_space,
        promising_quantile=0.1,
        objective_names=[loss_metric],
        default_min_bandwidth_factor=0.1,
        larger_is_better_objectives=None,
        rng=np.random.RandomState(),
        n_resamples=n_resamples,
    )
    pdf = _get_promising_pdf(observations=configs, params=params)
    assert pdf.size == n_resamples
    assert pdf.dim == len(configs) - 1


def test_get_promising_pdf() -> None:
    n_configs = 10
    loss_metric = "loss"
    config_space, configs = get_random_config(n_configs)
    loss_vals = np.arange(n_configs)
    configs[loss_metric] = loss_vals

    for quantile in [0.1, 0.3, 0.5, 0.7, 0.9]:
        for lower_is_better in [True, False]:
            params = _IoUTaskSimilarityParameters(
                n_samples=10,
                config_space=config_space,
                promising_quantile=quantile,
                objective_names=[loss_metric],
                default_min_bandwidth_factor=0.1,
                larger_is_better_objectives=None if lower_is_better else [0],
                rng=np.random.RandomState(),
                n_resamples=None,
            )
            pdf = _get_promising_pdf(observations=configs, params=params)
            n_promisings = int(n_configs * quantile)
            assert pdf.dim == len(configs) - 1
            assert pdf.size == n_promisings
            for hp_name, pe in pdf._parzen_estimators.items():
                if pe.__class__.__name__.startswith("Numerical"):
                    mask = loss_vals[:n_promisings] if lower_is_better else loss_vals[-n_promisings:][::-1]
                    assert np.allclose(pe._means, configs[hp_name][mask])


def test_get_promising_pdfs() -> None:
    n_configs = 10
    loss_metric = "loss"
    config_space, configs = get_random_config(n_configs)
    loss_vals = np.arange(n_configs)
    configs[loss_metric] = loss_vals
    params = _IoUTaskSimilarityParameters(
        n_samples=10,
        config_space=config_space,
        promising_quantile=0.1,
        objective_names=[loss_metric],
        default_min_bandwidth_factor=0.1,
        larger_is_better_objectives=None,
        rng=np.random.RandomState(),
        n_resamples=None,
    )
    n_pdfs = 3
    pdfs = _get_promising_pdfs(observations_set=[configs] * n_pdfs, params=params)
    assert len(pdfs) == n_pdfs
