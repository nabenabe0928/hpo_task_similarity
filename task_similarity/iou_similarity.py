from typing import Dict, List, NamedTuple, Optional, Tuple

import ConfigSpace as CS

import numpy as np

from parzen_estimator import MultiVariateParzenEstimator, get_multivar_pdf, over_resample


class _IoUTaskSimilarityParameters(NamedTuple):
    """
    Args:
        config_space (CS.ConfigurationSpace):
            The configuration space for the parzen estimator.
        objective_name (str):
            The name of the objective metric.
        promising_quantile (float):
            The quantile of the promising configs.
        default_min_bandwidth_factor (float):
            The factor of min bandwidth.
            For example, when we take 0.1, the bandwidth will be larger
            than 0.1 * (ub - lb).
        lower_is_better (bool):
            Whether the objective metric is better when lower.
        rng (np.random.RandomState):
            The random number generator.
        n_resamples (Optional[int]):
            How many resamplings we use for the parzen estimator.
            If None, we do not use resampling.
    """

    n_samples: int
    config_space: CS.ConfigurationSpace
    promising_quantile: float
    rng: Optional[np.random.RandomState]
    objective_name: str
    default_min_bandwidth_factor: float
    lower_is_better: bool
    n_resamples: Optional[int]


def _get_promising_pdf(
    observations: Dict[str, np.ndarray],
    params: _IoUTaskSimilarityParameters,
) -> MultiVariateParzenEstimator:
    hp_names = params.config_space.get_hyperparameter_names()
    n_promisings = max(1, int(params.promising_quantile * observations[params.objective_name].size))
    _sign = 1 if params.lower_is_better else -1
    promising_indices = np.argsort(_sign * observations[params.objective_name])[:n_promisings]
    promising_configs = {}
    for hp_name in hp_names:
        promising_configs[hp_name] = observations[hp_name][promising_indices]

    if params.n_resamples is None:
        return get_multivar_pdf(
            observations=promising_configs,
            config_space=params.config_space,
            default_min_bandwidth_factor=params.default_min_bandwidth_factor,
            prior=False,
        )
    else:
        return over_resample(
            config_space=params.config_space,
            observations=promising_configs,
            n_resamples=params.n_resamples,
            rng=params.rng,
            default_min_bandwidth_factor=params.default_min_bandwidth_factor,
        )


def _get_promising_pdfs(
    observations_set: List[Dict[str, np.ndarray]],
    params: _IoUTaskSimilarityParameters,
) -> List[MultiVariateParzenEstimator]:
    """
    Get the promising distributions for each task.

    Args:
        observations_set (List[Dict[str, np.ndarray]]):
            The observations for each task.
        params (IoUTaskSimilarityParameters):
            The parameters for the task similarity measure class.

    Returns:
        promising_pdfs (List[MultiVariateParzenEstimator]):
            The list of the promising distributions of each task.
            The shape is (n_tasks, ).
    """
    promising_pdfs: List[MultiVariateParzenEstimator] = []
    for observations in observations_set:
        promising_pdfs.append(_get_promising_pdf(observations=observations, params=params))

    return promising_pdfs


def _get_hypervolume(config_space: CS.ConfigurationSpace) -> float:
    """
    Compute the hypervolumen given the config space.

    Args:
        config_space (CS.ConfigurationSpace):
            The configuration space for the parzen estimator.

    Returns:
        hypervolume (float):
            The hypervolume of the config space.
    """
    hp_names = config_space.get_hyperparameter_names()
    hv = 1.0
    for hp_name in hp_names:
        config = config_space.get_hyperparameter(hp_name)
        config_type = config.__class__.__name__
        if config_type.startswith("Categorical"):
            hv *= len(config.choices)
        elif config_type.startswith("Ordinal"):
            hv *= config.meta["upper"] - config.meta["lower"]
        else:
            hv *= config.upper - config.lower

    return hv


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
        n_resamples (Optional[int]):
            The number of over-resampling for promising distributions.
            If None, we do not over-resample.
        Either of the following must be not None:
            promising_pdfs (Optional[List[MultiVariateParzenEstimator]]):
                The promising probability density functions (PDFs) for each task.
                Each PDF must be built by MultiVariateParzenEstimator.
            observations_set (Optional[List[Dict[str, np.ndarray]]]):
                The observations for each task.
    """

    _method_choices = ["top_set", "total_variation"]

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
        n_resamples: Optional[int] = None,
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
        self._params = _IoUTaskSimilarityParameters(
            n_samples=n_samples,
            config_space=config_space,
            promising_quantile=promising_quantile,
            rng=rng,
            objective_name=objective_name,
            default_min_bandwidth_factor=default_min_bandwidth_factor,
            lower_is_better=lower_is_better,
            n_resamples=n_resamples,
        )
        promising_pdfs = self._validate_input_and_promising_pdfs(observations_set, promising_pdfs)
        assert promising_pdfs is not None  # mypy re-definition
        self._hypervolume = _get_hypervolume(config_space)
        self._parzen_estimators = promising_pdfs
        self._samples = promising_pdfs[0].uniform_sample(n_samples, rng=rng if rng else np.random.RandomState())
        self._n_tasks = len(promising_pdfs)
        self._promising_quantile = promising_quantile
        self._negative_log_promising_pdf_vals: np.ndarray
        self._promising_pdf_vals: Optional[np.ndarray] = None
        self._promising_indices = self._compute_promising_indices()

    def _validate_input_and_promising_pdfs(
        self,
        observations_set: Optional[List[Dict[str, np.ndarray]]],
        promising_pdfs: Optional[List[MultiVariateParzenEstimator]],
    ) -> List[MultiVariateParzenEstimator]:
        promising_quantile = self._params.promising_quantile
        if promising_quantile < 0 or promising_quantile > 1:
            raise ValueError(f"The quantile for the promising domain must be in [0, 1], but got {promising_quantile}")
        if observations_set is None and promising_pdfs is None:
            raise ValueError("Either observations_set or promising_pdfs must be provided.")

        if promising_pdfs is not None:  # it is redundant, but needed for mypy redefinition
            promising_pdfs = promising_pdfs
        else:
            assert observations_set is not None
            promising_pdfs = _get_promising_pdfs(observations_set, self._params)

        return promising_pdfs

    @property
    def method_choices(self) -> List[str]:
        return self._method_choices[:]

    def _compute_promising_indices(self) -> np.ndarray:
        """
        Compute the indices of the top-(promising_quantile) quantile observations.
        The level of the promise are determined via the promising pdf values.

        Returns:
            promising_indices (np.ndarray):
                The indices for the promising samples.
                The shape is (n_tasks, n_promisings).
        """
        n_promisings = max(1, int(self._samples[0].size * self._promising_quantile))
        # Negative log pdf is better when it is larger
        self._negative_log_promising_pdf_vals = np.array([-pe.log_pdf(self._samples) for pe in self._parzen_estimators])
        indices = np.arange(self._negative_log_promising_pdf_vals[0].size)
        promising_indices = np.array(
            [
                indices[sorted_indices[:n_promisings]]
                for sorted_indices in np.argsort(self._negative_log_promising_pdf_vals, axis=-1)
            ]
        )
        return promising_indices

    def _compute_task_similarity_by_top_set(self, task1_id: int, task2_id: int) -> float:
        """
        Compute the task similarity via the IoU between the promising sets.

        Args:
            task1_id (int):
                The index of the task 1.
            task2_id (int):
                The index of the task 2.

        Returns:
            task_similarity (float):
                Task similarity estimated via the IoU of the promising sets.
        """
        idx1, idx2 = self._promising_indices[task1_id], self._promising_indices[task2_id]
        n_intersect = np.sum(np.in1d(idx1, idx2, assume_unique=True))
        return n_intersect / (idx1.size + idx2.size - n_intersect)

    def _compute_task_similarity_by_total_variation(self, task1_id: int, task2_id: int) -> float:
        """
        Compute the task similarity via the total variation distance between two promising distributions.

        Args:
            task1_id (int):
                The index of the task 1.
            task2_id (int):
                The index of the task 2.

        Returns:
            task_similarity (float):
                Task similarity estimated via the total variation distance.
        """
        if self._promising_pdf_vals is None:
            self._promising_pdf_vals = np.exp(-self._negative_log_promising_pdf_vals)
        else:  # it is redundant, but needed for mypy redefinition
            self._promising_pdf_vals = self._promising_pdf_vals

        pdf_diff = self._promising_pdf_vals[task1_id] - self._promising_pdf_vals[task2_id]
        total_variation = 0.5 * np.abs(pdf_diff * self._hypervolume).mean()
        return np.clip((1.0 - total_variation) / (1.0 + total_variation), 0.0, 1.0)

    def _compute_task_similarity(self, task1_id: int, task2_id: int, method: str = "top_set") -> float:
        """
        Compute the task similarity.

        Args:
            task1_id (int):
                The index of the task 1.
            task2_id (int):
                The index of the task 2.
            mode (str):
                The name of the task similarity method.

        Returns:
            task_similarity (float):
                Task similarity estimated via the total variation distance.
        """
        if method not in self._method_choices:
            raise ValueError(f"Task similarity method must be in {self._method_choices}, but got {method}")

        return getattr(self, f"_compute_task_similarity_by_{method}")(task1_id, task2_id)

    def compute(self, task_pairs: Optional[List[Tuple[int, int]]] = None, method: str = "top_set") -> np.ndarray:
        """
        Compute the task similarity and return the task similarity array.

        Args:
            task_pairs (Optional[List[Tuple[int, int]]]):
                The pairs of task indices of which we would like to compute the task similarity.
                If None, we compute all possible pairs.
            method (str):
                The method name of the task similarity method.

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

        task_pairs = (
            task_pairs
            if task_pairs is not None
            else [(i, j) for i in range(self._n_tasks) for j in range(self._n_tasks)]
        )
        for task1_id, task2_id in task_pairs:
            if not computed[task1_id, task2_id]:
                sim = self._compute_task_similarity(task1_id, task2_id, method=method)
                task_similarities[task1_id, task2_id] = task_similarities[task2_id, task1_id] = sim
                computed[task1_id, task2_id] = computed[task2_id, task1_id] = True

        return task_similarities
