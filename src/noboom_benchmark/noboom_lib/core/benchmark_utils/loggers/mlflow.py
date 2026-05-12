import os
from typing import Any, Literal, Optional

from lightning.pytorch.loggers import MLFlowLogger as LMLFlowLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from typing_extensions import override


class MLFlowLogger(LMLFlowLogger):
    def __init__(self,
                 experiment_name: str = "lightning_logs",
                 run_name: Optional[str] = None,
                 tracking_uri: Optional[str] = os.getenv("MLFLOW_TRACKING_URI"),
                 tags: Optional[dict[str, Any]] = None,
                 save_dir: Optional[str] = "./mlruns",
                 log_model: Literal[True, False, "all"] = False,
                 prefix: str = "",
                 artifact_location: Optional[str] = None,
                 run_id: Optional[str] = None,
                 synchronous: Optional[bool] = None,
                 log_checkpoints: bool = False,):
        """Initialize the MLflow logger with benchmark defaults.

        Args:
            experiment_name (str): MLflow experiment name. Defaults to "lightning_logs".
            run_name (Optional[str]): Run name. Defaults to None.
            tracking_uri (Optional[str]): MLflow tracking URI. Defaults to env var.
            tags (Optional[dict[str, Any]]): Tags to attach to the run. Defaults to None.
            save_dir (Optional[str]): Local save directory. Defaults to "./mlruns".
            log_model (Literal[True, False, "all"]): Whether to log models. Defaults to False.
            prefix (str): Prefix for logged keys. Defaults to "".
            artifact_location (Optional[str]): Artifact location override. Defaults to None.
            run_id (Optional[str]): Existing run ID to use. Defaults to None.
            synchronous (Optional[bool]): Synchronous logging flag. Defaults to None.
        """
        self._log_checkpoints = log_checkpoints
        super().__init__(experiment_name=experiment_name,
                         run_name=run_name,
                         tracking_uri=tracking_uri,
                         tags=tags,
                         save_dir=save_dir,
                         log_model=log_model if log_checkpoints else False,
                         prefix=prefix,
                         artifact_location=artifact_location,
                         run_id=run_id,
                         synchronous=synchronous)

    @override
    @rank_zero_only
    def finalize(self, status: str = "success") -> None:
        """Finalize the MLflow run, mapping Lightning statuses to MLflow statuses.

        Args:
            status (str): Status string from Lightning. Defaults to "success".

        Returns:
            None: Run is terminated in MLflow.

        Side Effects:
            Logs checkpoints and updates MLflow run status.
        """
        if not self._initialized:
            return
        if status == "success":
            status = "FINISHED"
        elif status == "failed":
            status = "FAILED"
        elif status == "finished":
            status = "FINISHED"

        if self._log_checkpoints and self._checkpoint_callback:
            self._scan_and_log_checkpoints(self._checkpoint_callback)

        # We set it ourselves later
        if self.experiment.get_run(self.run_id):
          self.experiment.set_terminated(self.run_id, status)
