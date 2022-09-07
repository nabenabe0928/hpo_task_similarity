from typing import Dict, Tuple
import pytest
import unittest

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

import numpy as np

from task_similarity import IoUTaskSimilarity


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

        IoUTaskSimilarity(n_samples, config_space, observations_set=[configs])

    def test_compute_promising_indices(self) -> None:
        n_configs = 10
        config_space, configs = get_random_config(n_configs=n_configs)
        n_samples = 10
        configs["loss"] = np.arange(n_configs)
        ts = IoUTaskSimilarity(n_samples, config_space, observations_set=[configs])
        assert ts._promising_indices.size == n_samples * ts._promising_quantile

    def test_reduced_size(self) -> None:
        for n_configs in [1, 10, 100, 1000]:
            config_space, configs = get_random_config(n_configs=n_configs)
            configs["loss"] = np.arange(n_configs)
            for eta in [0.5, 1, 2, 3, 5, 10]:
                if eta < 1:
                    with pytest.raises(ValueError):
                        ts = IoUTaskSimilarity(
                            n_samples=10,
                            config_space=config_space,
                            observations_set=[configs],
                            dim_reduction_factor=eta,
                        )
                else:
                    ts = IoUTaskSimilarity(
                        n_samples=10,
                        config_space=config_space,
                        observations_set=[configs],
                        dim_reduction_factor=eta,
                    )
                    if eta != 1:
                        ans = min(len(config_space), int(np.log(n_configs) / np.log(eta)))
                        assert len(ts._params.config_space) == ans
                    else:
                        assert len(ts._params.config_space) == 0

    def test_compute_importance(self) -> None:
        n_configs = 10
        config_space, configs = get_random_config(n_configs=n_configs)
        n_samples = 10
        configs["loss"] = np.arange(n_configs)
        print(configs)
        src_hp_imp = {
            "x0": np.array([0.1]),
            "x1": np.array([0.2]),
            "x2": np.array([0.3]),
            "x3": np.array([0.4]),
        }
        ts = IoUTaskSimilarity(
            n_samples, config_space, observations_set=[configs, configs], source_task_hp_importance=src_hp_imp
        )
        for k, v in src_hp_imp.items():
            np.all(ts.source_task_hp_importance[k] == v)

        ts = IoUTaskSimilarity(n_samples, config_space, observations_set=[configs, configs])
        assert all(k in ts.source_task_hp_importance for k in src_hp_imp)

    def test_compute_task_similarity(self) -> None:
        n_configs = 10
        config_space, configs = get_random_config(n_configs=n_configs)
        n_samples = 10
        configs["loss"] = np.arange(n_configs)
        ts = IoUTaskSimilarity(n_samples=n_samples, config_space=config_space, observations_set=[configs, configs])
        with pytest.raises(ValueError):
            ts._compute_task_similarity(task1_id=0, task2_id=1, method="dummy")

        for choice in ts.method_choices:
            assert ts._compute_task_similarity(task1_id=0, task2_id=1, method=choice) == 1.0

        ts._compute_task_similarity_by_total_variation(task1_id=0, task2_id=1)

        for n_configs in [3, 20]:
            observations_set = []
            config_space, configs = get_random_config(n_configs=n_configs)
            configs["loss"] = np.arange(n_configs)
            observations_set.append(configs)
            _, configs = get_random_config(n_configs=n_configs)
            configs["loss"] = np.arange(n_configs)
            observations_set.append(configs)
            rng = np.random.RandomState(0)
            ts = IoUTaskSimilarity(
                n_samples=n_samples, config_space=config_space, observations_set=observations_set, rng=rng
            )

            for choice in ts.method_choices:
                if n_configs < 5:  # not sufficient number of observations
                    assert ts._compute_task_similarity(task1_id=0, task2_id=1, method=choice) == 1.0
                else:
                    assert ts._compute_task_similarity(task1_id=0, task2_id=1, method=choice) < 1.0

    def test_compute(self) -> None:
        n_configs = 10
        config_space, configs = get_random_config(n_configs=n_configs)
        n_samples = 10
        configs["loss"] = np.arange(n_configs)
        ts = IoUTaskSimilarity(
            n_samples=n_samples, config_space=config_space, observations_set=[configs, configs, configs]
        )
        assert np.allclose(ts.compute(), np.ones(9).reshape(3, 3))
        assert np.allclose(ts.compute(task_pairs=[]), np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        assert np.allclose(ts.compute(task_pairs=[(0, 1), (0, 2)]), np.array([[1, 1, 1], [1, 1, 0], [1, 0, 1]]))

        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameters(
            [
                CSH.UniformFloatHyperparameter("x0", lower=-1, upper=1),
                CSH.UniformFloatHyperparameter("x1", lower=-1, upper=1),
                CSH.UniformFloatHyperparameter("x2", lower=-1, upper=1),
            ]
        )
        loss_metric = "dummy_loss"
        size = 100
        configs1 = {f"x{idx}": vals for idx, vals in enumerate(np.random.random((3, 100)))}
        configs2 = {f"x{idx}": vals for idx, vals in enumerate(np.random.random((3, 100)) - 1)}
        configs1[loss_metric] = np.arange(size)
        configs2[loss_metric] = np.arange(size)
        ts = IoUTaskSimilarity(
            n_samples=n_samples,
            config_space=config_space,
            observations_set=[configs1, configs2],
            objective_names=[loss_metric],
        )
        assert np.allclose(ts.compute(), np.identity(2))


if __name__ == "__main__":
    unittest.main()
