from typing import Dict, List, Optional, Tuple

import ConfigSpace as CS

import numpy as np

from parzen_estimator import MultiVariateParzenEstimator, get_multivar_pdf


def get_promising_pdf(
    config_space: CS.ConfigurationSpace,
    observations: Dict[str, np.ndarray],
    objective_name: str,
    promising_quantile: float,
    default_min_bandwidth_factor: float,
    lower_is_better: bool,
) -> MultiVariateParzenEstimator:
    hp_names = config_space.get_hyperparameter_names()
    n_promisings = max(1, int(promising_quantile * observations[objective_name].size))
    _sign = 1 if lower_is_better else -1
    promising_indices = np.argsort(_sign * observations[objective_name])[:n_promisings]
    promising_configs = {}
    for hp_name in hp_names:
        promising_configs[hp_name] = observations[hp_name][promising_indices]

    return get_multivar_pdf(
        observations=promising_configs,
        config_space=config_space,
        default_min_bandwidth_factor=default_min_bandwidth_factor,
        prior=False,
    )


def get_promising_pdfs(
    config_space: CS.ConfigurationSpace,
    observations_set: List[Dict[str, np.ndarray]],
    *,
    objective_name: str = "loss",
    promising_quantile: float = 0.1,
    default_min_bandwidth_factor: float = 1e-1,
    lower_is_better: bool = True,
) -> List[MultiVariateParzenEstimator]:
    promising_pdfs: List[MultiVariateParzenEstimator] = []
    for observations in observations_set:
        promising_pdfs.append(
            get_promising_pdf(
                config_space=config_space,
                observations=observations,
                objective_name=objective_name,
                promising_quantile=promising_quantile,
                default_min_bandwidth_factor=default_min_bandwidth_factor,
                lower_is_better=lower_is_better,
            )
        )

    return promising_pdfs


class IoUTaskSimilarity:
    """
    The task similarity measure class for blackbox optimization.
    IoU stands for Intersection over union.

    Args:
        config_space (CS.ConfigurationSpace):
            The configuration space for the parzen estimator.
        n_samples (int):
            The number of samples we use for the Monte-Carlo.
        promising_quantile (float):
            How much quantile we should consider as promising.
        objective_name (str):
            The name of the objective metric.
        rng (Optional[np.random.RandomState]):
            The random number generator to be used.
        Either of the following must be not None:
            promising_pdfs (Optional[List[MultiVariateParzenEstimator]]):
                The promising probability density functions (PDFs) for each task.
                Each PDF must be built by MultiVariateParzenEstimator.
            observations_set (Optional[List[MultiVariateParzenEstimator]]):
                The observations for each task.
    """

    def __init__(
        self,
        n_samples: int,
        config_space: CS.ConfigurationSpace,
        *,
        promising_quantile: float = 0.1,
        observations_set: Optional[List[Dict[str, np.ndarray]]] = None,
        promising_pdfs: Optional[List[MultiVariateParzenEstimator]] = None,
        rng: Optional[np.random.RandomState] = None,
        objective_name: str = "loss",
        default_min_bandwidth_factor: float = 1e-1,
        lower_is_better: bool = True,
    ):
        """
        Attributes:
            samples (List[np.ndarray]):
                Samples drawn from sobol sampler.
            n_tasks (int):
                The number of tasks.
            promising_indices (np.ndarray):
                The indices of promising samples drawn from sobol sequence.
                The promise is determined via the promising pdf values.
        """
        if promising_quantile < 0 or promising_quantile > 1:
            raise ValueError(f"The quantile for the promising domain must be in [0, 1], but got {promising_quantile}")
        if observations_set is None and promising_pdfs is None:
            raise ValueError("Either observations_set or promising_pdfs must be provided.")
        elif promising_pdfs is None:
            assert observations_set is not None
            promising_pdfs = get_promising_pdfs(
                config_space=config_space,
                observations_set=observations_set,
                objective_name=objective_name,
                promising_quantile=promising_quantile,
                default_min_bandwidth_factor=default_min_bandwidth_factor,
                lower_is_better=lower_is_better,
            )

        assert promising_pdfs is not None  # mypy re-definition
        self._parzen_estimators = promising_pdfs
        self._samples = promising_pdfs[0].uniform_sample(n_samples, rng=rng if rng else np.random.RandomState())
        self._n_tasks = len(promising_pdfs)
        self._promising_quantile = promising_quantile
        self._promising_indices = self._compute_promising_indices()

    def _compute_promising_indices(self) -> np.ndarray:
        n_promisings = max(1, int(self._samples[0].size * self._promising_quantile))
        # Negative log pdf is better when it is larger
        negative_log_promising_pdf_vals = np.array([-pe.log_pdf(self._samples) for pe in self._parzen_estimators])
        indices = np.arange(negative_log_promising_pdf_vals[0].size)
        promising_indices = np.array(
            [
                indices[sorted_indices[:n_promisings]]
                for sorted_indices in np.argsort(negative_log_promising_pdf_vals, axis=-1)
            ]
        )
        return promising_indices

    def _compute_task_similarity(self, task1_id: int, task2_id: int) -> float:
        idx1, idx2 = self._promising_indices[task1_id], self._promising_indices[task2_id]
        n_intersect = np.sum(np.in1d(idx1, idx2, assume_unique=True))
        return n_intersect / (idx1.size + idx2.size - n_intersect)

    def compute(self, task_pairs: List[Tuple[int, int]]) -> np.ndarray:
        """
        Compute the task similarity and return the task similarity array.

        Args:
            task_pairs (List[Tuple[int, int]]):
                The pairs of task indices of which we would like to compute the task similarity.

        Returns:
            task_similarities (np.ndarray):
                The task similarities of each task.
                task_similarities[i][j] := the task similarity of the task i and task j.
                Note that the following always holds:
                    1. task_similarities[i][j] == task_similarities[j][i]
                    2. task_similarities[i][i] == 1
                    3. 0 <= task_similarities[i][j] <= 1
        """
        task_similarities = np.full((self._n_tasks, self._n_tasks), 0.0)
        computed = np.full((self._n_tasks, self._n_tasks), False)
        diag_slice = (range(self._n_tasks), range(self._n_tasks))

        task_similarities[diag_slice] = 1
        computed[diag_slice] = True

        for task1_id, task2_id in task_pairs:
            if not computed[task1_id, task2_id]:
                sim = self._compute_task_similarity(task1_id, task2_id)
                task_similarities[task1_id, task2_id] = task_similarities[task2_id, task1_id] = sim
                computed[task1_id, task2_id] = computed[task2_id, task1_id] = True

        return task_similarities
