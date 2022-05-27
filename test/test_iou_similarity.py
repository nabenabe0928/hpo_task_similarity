from typing import Dict, Tuple
import pytest
import unittest

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

import numpy as np

from task_similarity import IoUTaskSimilarity
from task_similarity.iou_similarity import _get_hypervolume, _get_promising_pdf, get_promising_pdfs


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
    pdf = _get_promising_pdf(
        config_space=config_space,
        observations=configs,
        objective_name=loss_metric,
        promising_quantile=0.1,
        default_min_bandwidth_factor=0.1,
        lower_is_better=True,
        rng=np.random.RandomState(),
        n_resamples=n_resamples,
    )
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
            pdf = _get_promising_pdf(
                config_space=config_space,
                observations=configs,
                objective_name=loss_metric,
                promising_quantile=quantile,
                default_min_bandwidth_factor=0.1,
                lower_is_better=lower_is_better,
                rng=np.random.RandomState(),
                n_resamples=None,
            )
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
    n_pdfs = 3
    pdfs = get_promising_pdfs(config_space, observations_set=[configs] * n_pdfs)

    assert len(pdfs) == n_pdfs


class TestIoUTaskSimilarity(unittest.TestCase):
    def test_init(self) -> None:
        n_configs = 10
        config_space, configs = get_random_config(n_configs=n_configs)
        n_samples = 10
        configs["loss"] = np.arange(n_configs)
        for promising_quantile in [-0.1, 1.1]:
            with pytest.raises(ValueError):
                IoUTaskSimilarity(
                    n_samples, config_space, promising_quantile=promising_quantile, observations_set=[configs]
                )

        with pytest.raises(ValueError):
            IoUTaskSimilarity(n_samples, config_space)

        IoUTaskSimilarity(n_samples, config_space, observations_set=[configs])

    def test_compute_promising_indices(self) -> None:
        n_configs = 10
        config_space, configs = get_random_config(n_configs=n_configs)
        n_samples = 10
        configs["loss"] = np.arange(n_configs)
        ts = IoUTaskSimilarity(n_samples, config_space, observations_set=[configs])
        assert ts._promising_indices.size == n_samples * ts._promising_quantile

    def test_compute_task_similarity(self) -> None:
        pass

    def test_compute(self) -> None:
        pass


if __name__ == "__main__":
    unittest.main()
