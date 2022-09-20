from typing import Dict, List, Optional, Tuple

import ConfigSpace as CS

import numpy as np

from parzen_estimator import MultiVariateParzenEstimator

from task_similarity.constants import _IoUTaskSimilarityParameters
from task_similarity.parameter_selection import compute_importance, reduce_dimension
from task_similarity.utils import _get_hypervolume, _get_promising_pdfs


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
        objective_names (List[str]):
            The names of the objective metrics.
        rng (Optional[np.random.RandomState]):
            The random number generator to be used.
        n_resamples (Optional[int]):
            The number of over-resampling for promising distributions.
            If None, we do not over-resample.
        source_task_hp_importance (Optional[Dict[str, np.ndarray]]):
            The hyperparameter importances in source tasks.
        observations_set (Optional[List[Dict[str, np.ndarray]]]):
            The observations for each task.
        dim_reduction_factor (float):
            eta in the paper.
            This parameter controls the number of dimensions to reduce.
    """

    _method_choices = ["top_set", "total_variation"]

    def __init__(
        self,
        n_samples: int,
        config_space: CS.ConfigurationSpace,
        observations_set: List[Dict[str, np.ndarray]],
        *,
        promising_quantile: float = 0.1,
        rng: Optional[np.random.RandomState] = None,
        objective_names: List[str] = ["loss"],
        default_min_bandwidth_factor: float = 1e-1,
        larger_is_better_objectives: Optional[List[int]] = None,
        n_resamples: Optional[int] = None,
        source_task_hp_importance: Optional[Dict[str, np.ndarray]] = None,
        dim_reduction_factor: float = 5.0,
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
            rng=rng if rng is not None else np.random.RandomState(),
            objective_names=objective_names,
            default_min_bandwidth_factor=default_min_bandwidth_factor,
            larger_is_better_objectives=larger_is_better_objectives,
            n_resamples=n_resamples,
            dim_reduction_factor=dim_reduction_factor,
        )

        self._source_task_hp_importance: Dict[str, np.ndarray]
        self._n_tasks: int
        promising_pdfs = self._reduce_dimension(
            observations_set=observations_set,
            source_task_hp_importance=source_task_hp_importance,
        )
        self._is_sufficient_samples = bool(len(promising_pdfs) > 0)
        if self._is_sufficient_samples:
            self._init_variables(promising_pdfs)

    def _init_variables(
        self,
        promising_pdfs: List[MultiVariateParzenEstimator],
    ) -> None:
        # Define after the dimension reduction
        self._hypervolume = _get_hypervolume(self._params.config_space)
        self._parzen_estimators = promising_pdfs
        self._samples = promising_pdfs[0].uniform_sample(self._params.n_samples, rng=self._params.rng)
        self._promising_quantile = self._params.promising_quantile
        self._negative_log_promising_pdf_vals: np.ndarray
        self._promising_pdf_vals: Optional[np.ndarray] = None
        self._promising_indices = self._compute_promising_indices()

    def _compute_importance(
        self,
        promising_pdfs: List[MultiVariateParzenEstimator],
        source_task_hp_importance: Optional[Dict[str, np.ndarray]],
    ) -> Dict[str, np.ndarray]:

        if source_task_hp_importance is None:
            hp_imp = compute_importance(promising_pdfs=promising_pdfs, rng=self._params.rng)
            source_task_hp_importance = {hp_name: imp[1:] for hp_name, imp in hp_imp.items()}
        else:
            hp_imp = compute_importance(promising_pdfs=[promising_pdfs[0]], rng=self._params.rng)

        self._source_task_hp_importance = source_task_hp_importance

        w = 1.0 / self._n_tasks
        hp_importance = {
            hp_name: w * imp[0] + (1 - w) * np.sum(source_task_hp_importance[hp_name])
            for hp_name, imp in hp_imp.items()
        }
        return hp_importance

    def _reduce_dimension(
        self,
        observations_set: List[Dict[str, np.ndarray]],
        source_task_hp_importance: Optional[Dict[str, np.ndarray]],
    ) -> List[MultiVariateParzenEstimator]:

        promising_pdfs = self._validate_input_and_promising_pdfs(observations_set)
        assert promising_pdfs is not None  # mypy re-definition
        self._n_tasks = len(promising_pdfs)

        hp_importance = self._compute_importance(
            promising_pdfs=promising_pdfs,
            source_task_hp_importance=source_task_hp_importance,
        )
        n_observations = observations_set[0][list(hp_importance.keys())[0]].size
        if self._params.dim_reduction_factor == 1:
            dim_after = 0
        else:
            dim_after = min(int(np.log(n_observations) / np.log(self._params.dim_reduction_factor)), len(hp_importance))

        new_promising_pdfs, new_config_space = reduce_dimension(
            hp_importance=hp_importance,
            dim_after=dim_after,
            config_space=self._params.config_space,
            promising_pdfs=promising_pdfs,
        )
        self._params = self._params._replace(config_space=new_config_space)
        return new_promising_pdfs

    def _validate_input_and_promising_pdfs(
        self,
        observations_set: List[Dict[str, np.ndarray]],
    ) -> List[MultiVariateParzenEstimator]:
        promising_quantile = self._params.promising_quantile
        if promising_quantile < 0 or promising_quantile > 1:
            raise ValueError(f"The quantile for the promising domain must be in [0, 1], but got {promising_quantile}")
        if self._params.dim_reduction_factor < 1:
            val = self._params.dim_reduction_factor
            raise ValueError(f"dim_reduction_factor must be larger than or equal to 1, but got {val}")

        assert observations_set is not None
        promising_pdfs = _get_promising_pdfs(observations_set, self._params)

        return promising_pdfs

    @property
    def method_choices(self) -> List[str]:
        return self._method_choices[:]

    @property
    def source_task_hp_importance(self) -> Dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self._source_task_hp_importance.items()}

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
        if not self._is_sufficient_samples:
            return 1.0

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
        if not self._is_sufficient_samples:
            return 1.0

        if self._promising_pdf_vals is None:
            self._promising_pdf_vals = np.exp(-self._negative_log_promising_pdf_vals)
        else:  # it is redundant, but needed for mypy redefinition
            self._promising_pdf_vals = self._promising_pdf_vals

        pdf_diff = self._promising_pdf_vals[task1_id] - self._promising_pdf_vals[task2_id]
        total_variation = 0.5 * np.abs(pdf_diff * self._hypervolume).mean()
        return np.clip((1.0 - total_variation) / (1.0 + total_variation), 0.0, 1.0)

    def _compute_task_similarity(self, task1_id: int, task2_id: int, method: str = "total_variation") -> float:
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

    def compute(
        self, task_pairs: Optional[List[Tuple[int, int]]] = None, method: str = "total_variation"
    ) -> np.ndarray:
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
