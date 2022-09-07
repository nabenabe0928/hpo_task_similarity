from typing import Dict, Tuple
import unittest

import ConfigSpace as CS

import numpy as np

import pytest

from parzen_estimator import get_multivar_pdf

from task_similarity.parameter_selection import (
    compute_importance,
    reduce_dimension,
)


def get_random_config(n_configs: int = 10) -> Tuple[CS.ConfigurationSpace, Dict[str, np.ndarray]]:
    config_space = CS.ConfigurationSpace()
    meta_info = {"lower": -2, "upper": 1}
    config_space.add_hyperparameters(
        [
            CS.UniformFloatHyperparameter("x0", lower=-2, upper=3),
            CS.UniformIntegerHyperparameter("x1", lower=-2, upper=2),
            CS.OrdinalHyperparameter("x2", sequence=list(range(-2, 2)), meta=meta_info),
            CS.CategoricalHyperparameter("x3", choices=["a", "b"]),
        ]
    )
    return config_space, {
        f"x{d}": np.array([config_space.sample_configuration()[f"x{d}"] for _ in range(n_configs)]) for d in range(4)
    }


def test_compute_importance() -> None:
    config_space, configs = get_random_config()
    rng = np.random.RandomState()
    mvpe = get_multivar_pdf(observations=configs, config_space=config_space)
    imp = compute_importance(promising_pdfs=[mvpe, mvpe], rng=rng)
    for k in config_space:
        assert imp[k].size == 2

    config_space = CS.ConfigurationSpace()
    config_space.add_hyperparameter(CS.UniformFloatHyperparameter("x0", lower=0, upper=1))
    config_space.add_hyperparameter(CS.UniformFloatHyperparameter("x1", lower=0, upper=4))
    same_vals = np.array([1 / 2] * 10)
    configs = {"x0": same_vals, "x1": same_vals * 4}
    mvpe = get_multivar_pdf(configs, config_space)

    imp0, imp1 = 0.0, 0.0
    for _ in range(10):
        imp = compute_importance(promising_pdfs=[mvpe], rng=rng)
        imp0 += imp["x0"][0]
        imp1 += imp["x1"][0]
    else:
        imp0 /= 10
        imp1 /= 10

    # 2560 samples --> sqrt(2560) simeq 16 * 3.3 = 51.2
    # The error should be around the order of imp0 / 51.2.
    eps = imp0 / 30
    assert imp0 - eps <= imp1 <= imp0 + eps

    n_points = 1000
    configs = {"x0": np.linspace(0, 1, n_points), "x1": np.linspace(0, 4, n_points)}
    mvpe = get_multivar_pdf(configs, config_space)
    imp = compute_importance([mvpe], rng)
    assert 0 < imp["x0"] < 0.005
    assert 0 < imp["x1"] < 0.005


def test_reduce_dimension() -> None:
    config_space, configs = get_random_config()
    imp = {"x0": 0.1, "x1": 0.3, "x2": 0.2, "x3": 0.15}
    mvpe = get_multivar_pdf(configs, config_space)

    hp_names_set = [
        [],
        ["x1"],
        ["x1", "x2"],
        ["x1", "x2", "x3"],
        ["x0", "x1", "x2", "x3"],
    ]

    for dim in range(-1, 7):
        if dim < 0 or dim > 4:
            with pytest.raises(ValueError):
                pdfs, cs = reduce_dimension(
                    hp_importance=imp,
                    dim_after=dim,
                    config_space=config_space,
                    promising_pdfs=[mvpe, mvpe],
                )
            continue
        else:
            pdfs, cs = reduce_dimension(
                hp_importance=imp,
                dim_after=dim,
                config_space=config_space,
                promising_pdfs=[mvpe, mvpe],
            )

        assert len(cs) == dim
        if dim == 0:
            assert pdfs == []
        else:
            assert len(pdfs) == 2
            assert all(name in cs.get_hyperparameter_names() for name in hp_names_set[dim])
            for i in range(2):
                assert pdfs[i].dim == dim
                assert pdfs[i].param_names == hp_names_set[dim]


if __name__ == "__main__":
    unittest.main()
