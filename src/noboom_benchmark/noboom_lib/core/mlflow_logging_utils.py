import logging
from typing import Dict, List, Optional

import mlflow
import mlflow.entities
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
from mlflow.utils.time import get_current_time_millis
from ray.air._internal.mlflow import _MLflowLoggerUtil
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray._private.dict import flatten_dict
from ray.tune.experiment import Trial


logger = logging.getLogger(__name__)


class MLFlowLoggerUtilV2(_MLflowLoggerUtil):
    def log_metrics(self, step, metrics_to_log: Dict, run_id: Optional[str] = None):
        """Log metrics to MLflow with optional dataset/model context.

        Args:
            step (int): Step index for the metrics.
            metrics_to_log (Dict): Metrics dictionary to log.
            run_id (str): Run ID to log to.

        Returns:
            None: Metrics are logged to MLflow.

        Side Effects:
            Writes metrics to MLflow runs.
        """
        dataset_name = metrics_to_log.get("dataset_name")
        dataset_digest = metrics_to_log.get("dataset_digest")
        model_id = metrics_to_log.get("model_id")

        metrics_to_log = flatten_dict(metrics_to_log)
        metrics_to_log = self._parse_dict(metrics_to_log)
        timestamp = get_current_time_millis()

        if run_id and self._run_exists(run_id):
            client = self._get_client()
            mlflow_metrics = []
            for key, value in metrics_to_log.items():
                mlflow_metrics.append(mlflow.entities.Metric(key=key, value=value, timestamp=timestamp, step=step,
                                                             model_id=model_id, dataset_name=dataset_name,
                                                             dataset_digest=dataset_digest))
            client.log_batch(run_id, mlflow_metrics)
        else:
            self._mlflow.log_metrics(metrics=metrics_to_log, step=step, timestamp=timestamp, run_id=run_id)

class MLFlowLoggerCallbackSW(MLflowLoggerCallback):
    def __init__(self, *args, **kwargs):
        """Initialize the MLflow logger callback with a custom util."""
        super().__init__(*args, **kwargs)
        self.mlflow_util = MLFlowLoggerUtilV2()

    def log_trial_start(self, trial: "Trial"):
        """Log trial start with a SeaweedFS worker lock.

        Args:
            trial (Trial): Ray Tune trial instance.

        Returns:
            None: Delegates to base implementation.
        """
        # with worker_lock():
        super().log_trial_start(trial)

    def log_trial_end(self, trial: "Trial", failed: bool = False):
        """Log trial end and terminate child MLflow runs.

        Args:
            trial (Trial): Ray Tune trial instance.
            failed (bool): Whether the trial failed. Defaults to False.

        Returns:
            None: Updates MLflow run status.

        Side Effects:
            Terminates child runs in MLflow.
        """
        # with worker_lock():
        super().log_trial_end(trial, failed)
        experiment_id = mlflow.get_experiment_by_name(
            self.experiment_name
        ).experiment_id
        run_id = self._trial_runs[trial]

        runs: List[mlflow.entities.Run] = mlflow.search_runs(
            experiment_ids=[experiment_id],
            filter_string=(
                f"tags.{MLFLOW_PARENT_RUN_ID} = '{run_id}' "
                f"AND attributes.status = 'RUNNING'"
            ),
            output_format="list",
        )

        # Stop child runs
        status = "FINISHED" if not failed else "FAILED"
        for run in runs:
            self.mlflow_util.end_run(status, run.info.run_id)

    def log_trial_result(self, iteration: int, trial: "Trial", result: Dict):
        """Log trial results under a SeaweedFS worker lock.

        Args:
            iteration (int): Current iteration number.
            trial (Trial): Ray Tune trial instance.
            result (Dict): Result payload to log.

        Returns:
            None: Delegates to base implementation.
        """
        # with worker_lock():
        super().log_trial_result(iteration, trial, result)
