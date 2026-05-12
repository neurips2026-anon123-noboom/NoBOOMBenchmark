"""Lightning module wrapper for evaluating anomaly detectors on NoBoom data."""
import itertools
import logging
import os
import pickle
from contextlib import nullcontext
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Union
import copy

import lightning as L
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from lightning.fabric.utilities.throughput import measure_flops
import numpy as np
from timesead_experiments.utils.training_ingredient import instantiate_loss
import torch
from torch import nn
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from torch._dynamo.exc import TorchDynamoException, FailOnRecompileLimitHit, UserError

from timesead.models import BaseModel
from timesead.models.common import AnomalyDetector
from timesead.optim.loss import Loss
from timesead.utils.utils import pack_tuple

from .benchmark_helpers import insert_before_subsequences, pad_each_subsequence
from ..metric_utils.alarm_threshold_search import DATASET_FRACTIONS, select_threshold
from ..metric_utils.evaluation_postprocessing import (
    EvaluationPostprocessingConfig,
    evaluate_sequences_with_optional_postprocessing,
    resolve_evaluation_postprocessing_config,
)
from ..logging import setup_logging
from ..metric_utils.metrics import get_metric_by_name, is_metric_binary
from .style_transfer import StyleTransfer


logger = logging.getLogger(__name__)
DetectorCallable = Union[Callable[[BaseModel], AnomalyDetector], AnomalyDetector]
torch.fx.experimental._config.meta_nonzero_assume_all_nonzero = True


def _diagnostic_logging_enabled() -> bool:
    value = os.getenv("NOBOOM_WORKER_DEBUG", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def instantiate_losses(losses):
    """Instantiate loss objects from configuration entries.

    Args:
        losses (Any): Single loss config or list of configs passed from YAML.

    Returns:
        Any: Concrete loss instance(s) ready for training.
    """
    if isinstance(losses, List):
        return [instantiate_loss(loss) for loss in losses]
    else:
        return instantiate_loss(losses)


class BenchmarkModel(L.LightningModule):
    def __init__(
        self,
        detector: DetectorCallable,
        network: Optional[BaseModel] = None,
        losses: Optional[Union[List[torch.nn.modules.loss._Loss], torch.nn.modules.loss._Loss]] = None,
        batch_dim: Optional[int] = None,
        window_size: Optional[int] = None,
        prediction_horizon: Optional[int] = None,
        label_index_offset: int = 0,
        num_features: Optional[int] = None,
        val_metrics: Optional[Dict[str, Callable]] = None,
        optimizer: Optional[OptimizerCallable] = None,
        lr_scheduler: Optional[LRSchedulerCallable] = None,
        metrics: Optional[List[str]] = None,
        target_metric: str = "alarm_score",
        compile_torch_model: bool = False,
        compile_torch_model_mode: Optional[str] = "default",
        compile_torch_model_fullgraph: bool = False,
        verbose: bool = False,
        fix_threshold: bool = False,
        predict_on_end: bool = False,
        predict_on_end_streaming: Optional[bool] = None,
        predict_on_end_stream_chunk_size: Optional[int] = None,
        prediction_stream: Optional[Dict[str, Any]] = None,
        ckpt_path: Optional[str] = None,
        use_full_seq_prediction: bool = True,
        save_hparams: bool = False,
        model_name: str = None,
        use_test_style_transfer: bool = False,
        style_transfer_ckpt_path: Optional[str] = None,
        evaluation_postprocessing: Optional[Dict[str, Any]] = None,
    ):
        """Wrap a detector/network pair with manual optimization control.

        Args:
            detector (DetectorCallable): Callable or instance producing an anomaly detector.
            network (Optional[BaseModel]): Optional backbone network used by the detector.
            losses (Optional[Union[List[torch.nn.modules.loss._Loss],
                torch.nn.modules.loss._Loss]]): Loss module(s) for training.
            batch_dim (Optional[int]): Index treated as batch dimension for collation.
            window_size (Optional[int]): Window size for label alignment.
            label_index_offset (int): Extra raw-row offset for label alignment when
                a model transform consumes rows before windowing.
            val_metrics (Optional[Dict[str, Callable]]): Validation metrics by name.
            optimizer (Optional[OptimizerCallable]): Optimizer factory for network params.
            lr_scheduler (Optional[LRSchedulerCallable]): Scheduler factory for optimizers.
            metrics (List[str]): Names of evaluation metrics to compute.
            target_metric (str): Primary metric used for model selection. Defaults to
                "alarm_score".
            compile_torch_model (bool): Whether to invoke ``torch.compile``. Defaults to False.
            compile_torch_model_mode (Optional[str]): Compilation mode. Defaults to "default".
            compile_torch_model_fullgraph (bool): Whether to enforce fullgraph. Defaults to False.
            verbose (bool): Enable verbose logging. Defaults to False.
            fix_threshold (bool): Whether to fix threshold based on dataset fraction.
                Defaults to False.
            predict_on_end_streaming (Optional[bool]): Keep ``predict_on_end`` input windows
                out of Lightning's prediction accumulator and aggregate them in this module.
                Defaults to enabled when ``prediction_stream.class_path`` is configured.
            predict_on_end_stream_chunk_size (Optional[int]): Deprecated compatibility
                option. Use ``prediction_stream.init_args.chunk_size``.
            prediction_stream (Optional[Dict[str, Any]]): Optional nested config
                for stream settings. Streaming requires ``class_path`` and uses
                ``init_args`` for stream-specific options.
        """
        super().__init__()

        self._diagnostic_logging_enabled = _diagnostic_logging_enabled()
        setup_logging(verbosity=1 if self._diagnostic_logging_enabled else int(bool(verbose)))
        self.verbose = verbose

        prediction_stream_options: Dict[str, Any] = dict(prediction_stream or {})
        if "enabled" in prediction_stream_options:
            predict_on_end_streaming = bool(prediction_stream_options["enabled"])
        if predict_on_end_streaming is None:
            predict_on_end_streaming = bool(prediction_stream_options.get("class_path"))

        self.network = network
        self.detector = detector
        self.batch_dimension = batch_dim
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.label_index_offset = int(label_index_offset)
        self.num_features = num_features
        self.losses = losses
        self.val_metrics = val_metrics
        self.fix_threshold = fix_threshold
        self.predict_on_end = predict_on_end
        self.predict_on_end_streaming = predict_on_end_streaming
        self.predict_on_end_stream_chunk_size = predict_on_end_stream_chunk_size
        self.predict_on_end_stream_adapter = prediction_stream_options.get("adapter")
        self.predict_on_end_stream_class_path = prediction_stream_options.get("class_path")
        self.predict_on_end_stream_init_args = prediction_stream_options.get("init_args")
        self.predict_on_end_stream_policy_name = (
            prediction_stream_options.get("policy_name")
            or prediction_stream_options.get("policy")
        )
        self.prediction_stream = {
            "enabled": self.predict_on_end_streaming,
            "class_path": self.predict_on_end_stream_class_path,
            "init_args": self.predict_on_end_stream_init_args,
        }
        self.ckpt_path = ckpt_path
        self.use_full_seq_prediction = use_full_seq_prediction
        self.model_name = model_name
        self.use_test_style_transfer = use_test_style_transfer
        self.style_transfer_ckpt_path = style_transfer_ckpt_path
        self._evaluation_postprocessing = resolve_evaluation_postprocessing_config(
            evaluation_postprocessing
        )

        self.style_transfer = None
        if self.use_test_style_transfer and self.style_transfer_ckpt_path is not None:
            self.style_transfer = StyleTransfer(style_transfer_ckpt_path, batch_dim=batch_dim)
        elif self.use_test_style_transfer:
            logger.info(
                "Synthetic predict requested style transfer, but no style_transfer_ckpt_path is configured for model '%s'. "
                "Skipping styled predict cache creation.",
                self.model_name,
            )

        # if save_hparams:
        #     self.save_hyperparameters(ignore=['save_hparams', 'detector', 'network', 'losses',
        #                                       'val_metrics', 'optimizer', 'lr_scheduler'])

        if self.losses is not None:
            losses = instantiate_losses(losses)
            self.losses = losses
            if val_metrics is None:
                if not isinstance(losses, list):
                    losses = [losses]
                self.val_metrics = nn.ModuleDict({f'val_loss_{i}': loss for i, loss in enumerate(losses)})

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        # Important: This property activates manual optimization.
        self.automatic_optimization = False

        if self.network is not None:
            parameters = pack_tuple(self.network.grouped_parameters())
            if isinstance(self.losses, Loss):
                self.losses = nn.ModuleList([self.losses] * len(parameters))
            else:
                assert len(self.losses) == len(parameters)

        self._test_metrics = {metric: get_metric_by_name(metric) for metric in metrics or []}
        self._test_metrics_results = {}
        self._anomaly_scores = None
        self._anomaly_labels = None
        self._detector_fitted = False
        self._target_metric = target_metric
        self._compile_torch_model = compile_torch_model
        self._compile_torch_model_mode = compile_torch_model_mode
        self._compile_torch_model_fullgraph = compile_torch_model_fullgraph
        self._compiled_network = None
        self._compile_invocation_count = 0
        self._trace_train_batch_limit = 2
        self._trace_compile_call_limit = 3
        self._prediction_stream_state = None
        self._prediction_stream_policy = None

    @property
    def evaluation_postprocessing_config(self) -> EvaluationPostprocessingConfig:
        return self._evaluation_postprocessing

    def _clamp_gdn_topk_to_num_features(self) -> None:
        """Prevent GDN from requesting more graph neighbors than its KNN graph can supply."""
        if self.model_name != "gdn" or self.network is None:
            return

        network_topk = getattr(self.network, "topk", None)
        if network_topk is None:
            return

        available_feature_count = self.num_features
        if available_feature_count is None:
            embedding = getattr(self.network, "embedding", None)
            available_feature_count = getattr(embedding, "num_embeddings", None)

        try:
            numeric_topk = int(network_topk)
            numeric_num_features = int(available_feature_count)
        except (TypeError, ValueError):
            return

        if numeric_num_features < 2:
            raise ValueError(
                "GDN requires at least 2 features because its CUDA KNN graph excludes self-loops."
            )

        # ``torch_geometric.nn.knn_graph`` is called with ``loop=False`` in TimeSeAD's
        # CUDA path, so each node can connect to at most ``num_features - 1`` neighbors.
        clamped_topk = max(1, min(numeric_topk, numeric_num_features - 1))
        if clamped_topk == numeric_topk:
            return

        logger.warning(
            "Clamping GDN topk from %d to %d because the prepared dataset exposes %d features and "
            "the CUDA KNN graph excludes self-loops.",
            numeric_topk,
            clamped_topk,
            numeric_num_features,
        )
        self.network.topk = clamped_topk

    @property
    def anomaly_scores(self):
        """Return cached anomaly scores.

        Returns:
            Any: Cached anomaly scores array.
        """
        return self._anomaly_scores

    @property
    def anomaly_labels(self):
        """Return cached anomaly labels.

        Returns:
            Any: Cached anomaly labels array.
        """
        return self._anomaly_labels

    @property
    def detector_fitted(self):
        """Return whether the detector has been fitted.

        Returns:
            bool: True if detector fitting has completed.
        """
        return self._detector_fitted

    @property
    def metrics(self):
        """Return cached metric results.

        Returns:
            dict: Mapping of metric names to results.
        """
        return self._test_metrics_results

    def _should_trace_training_batch(self, batch_idx: int) -> bool:
        """Return whether to emit detailed trace logs for this training batch."""
        return (
            self._diagnostic_logging_enabled
            and self.current_epoch == 0
            and batch_idx < self._trace_train_batch_limit
        )

    def _should_trace_compile_call(self, call_idx: int) -> bool:
        """Return whether to emit detailed trace logs for this compile call."""
        return self._diagnostic_logging_enabled and call_idx <= self._trace_compile_call_limit

    def _summarize_runtime_value(self, value: Any, *, max_items: int = 3) -> str:
        """Summarize runtime values without logging full tensors or collections."""
        if isinstance(value, torch.Tensor):
            return (
                "Tensor("
                f"shape={tuple(value.shape)}, "
                f"dtype={value.dtype}, "
                f"device={value.device}"
                ")"
            )

        if isinstance(value, tuple):
            items = [self._summarize_runtime_value(item, max_items=max_items) for item in value[:max_items]]
            if len(value) > max_items:
                items.append("...")
            return f"tuple([{', '.join(items)}])"

        if isinstance(value, list):
            items = [self._summarize_runtime_value(item, max_items=max_items) for item in value[:max_items]]
            if len(value) > max_items:
                items.append("...")
            return f"list([{', '.join(items)}])"

        if isinstance(value, dict):
            items = []
            for idx, (key, item) in enumerate(value.items()):
                if idx >= max_items:
                    items.append("...")
                    break
                items.append(f"{key}={self._summarize_runtime_value(item, max_items=max_items)}")
            return "{" + ", ".join(items) + "}"

        return repr(value)

    def _trainer_or_none(self) -> Optional[Any]:
        try:
            return self.trainer
        except RuntimeError:
            return None

    @staticmethod
    def _coerce_torch_device(device: Any) -> Optional[torch.device]:
        if device is None:
            return None
        if isinstance(device, torch.device):
            return device
        try:
            return torch.device(device)
        except (TypeError, RuntimeError):
            return None

    def _active_device(self) -> torch.device:
        trainer = self._trainer_or_none()
        strategy = getattr(trainer, "strategy", None)
        strategy_device = self._coerce_torch_device(getattr(strategy, "root_device", None))
        if strategy_device is not None:
            return strategy_device
        return self._coerce_torch_device(self.device) or torch.device("cpu")

    def _active_device_type(self) -> str:
        return self._active_device().type

    def _is_xla_device(self) -> bool:
        return self._active_device_type() == "xla"

    def _is_cuda_device(self) -> bool:
        return self._active_device_type() == "cuda"

    def _cuda_memory_snapshot_enabled(self) -> bool:
        return self._is_cuda_device() and torch.cuda.is_available()

    def _cuda_memory_snapshot(self) -> str:
        """Return a concise CUDA memory summary for the current device."""
        if not self._cuda_memory_snapshot_enabled():
            return f"cuda=inactive device={self._active_device()}"

        try:
            device_index = None
            active_device = self._active_device()
            if active_device.type == "cuda":
                device_index = active_device.index
            if device_index is None:
                device_index = torch.cuda.current_device()
            allocated_mb = torch.cuda.memory_allocated(device_index) / (1024 ** 2)
            reserved_mb = torch.cuda.memory_reserved(device_index) / (1024 ** 2)
            max_allocated_mb = torch.cuda.max_memory_allocated(device_index) / (1024 ** 2)
            return (
                f"cuda:{device_index} "
                f"alloc={allocated_mb:.1f}MiB "
                f"reserved={reserved_mb:.1f}MiB "
                f"max_alloc={max_allocated_mb:.1f}MiB"
            )
        except Exception as exc:
            return f"cuda_snapshot_error={type(exc).__name__}: {exc}"

    def _torch_autocast_disabled(self) -> Any:
        device_type = self._active_device_type()
        if device_type == "xla":
            return nullcontext()
        return torch.autocast(device_type=device_type, enabled=False)

    def _torch_compile_available_for_device(self) -> bool:
        return not self._is_xla_device()

    def _mark_cudagraph_step_begin(self) -> bool:
        if not self._is_cuda_device():
            return False
        mark_step_begin = getattr(getattr(torch, "compiler", None), "cudagraph_mark_step_begin", None)
        if mark_step_begin is None:
            return False
        mark_step_begin()
        return True

    def _load_network_weights(self) -> None:
        """Load network weights from a checkpoint without restoring trainer state.

        Args:
            ckpt_path (Union[str, Path]): Path to a Lightning checkpoint or raw state dict.
            strict (bool): Whether to enforce exact key matching. Defaults to True.

        Returns:
            None: Network parameters are updated in place.
        """
        if self.network is None:
            logger.warning("Skipping weight load because no network is configured.")
            return

        resolved_path = Path(self.ckpt_path)
        logger.info("Loading network weights from '%s'.", resolved_path)
        map_location = "cpu" if self._is_xla_device() else self._active_device_type()
        checkpoint = torch.load(resolved_path, map_location=map_location)
        state_dict = checkpoint.get("state_dict", checkpoint)
        network_state = {
            key.split("network.", 1)[1]: value
            for key, value in state_dict.items()
            if key.startswith("network.")
        }
        if network_state:
            load_result = self.network.load_state_dict(network_state)
        else:
            load_result = self.network.load_state_dict(state_dict)

        if load_result.missing_keys or load_result.unexpected_keys:
            logger.warning(
                "Weight load completed with missing keys=%s and unexpected keys=%s.",
                load_result.missing_keys,
                load_result.unexpected_keys,
            )

    def setup(self, stage: str):
        """Instantiate the detector lazily once the trainer is available.

        Args:
            stage (str): Lightning stage identifier (``fit``, ``validate``, etc.).

        Returns:
            None: Detector is instantiated when entering fit stage.
        """
        self._clamp_gdn_topk_to_num_features()

        if self.network is not None and 'timesnet' not in self.model_name:
            with torch.device("meta") as meta_device:
                meta_network = copy.deepcopy(self.network).to_empty(device=meta_device)
                effective_network_window_size = getattr(meta_network, "seq_len", self.window_size)

                def sample_forward() -> torch.Tensor:
                    wd = effective_network_window_size
                    shape = [wd, self.num_features]
                    shape.insert(self.batch_dimension, self.trainer.datamodule.batch_size)
                    x = torch.randn(*shape, device=meta_device)
                    return meta_network((x,))

                def loss_forward(out: torch.Tensor) -> torch.Tensor:
                    if self.prediction_horizon:
                        wd = self.prediction_horizon
                    else:
                        wd = effective_network_window_size
                    shape = [wd, self.num_features]
                    shape.insert(self.batch_dimension, self.trainer.datamodule.batch_size)
                    target = torch.randn(*shape, device=meta_device)
                    losses_list = []
                    for loss in self.losses:
                        meta_loss = copy.deepcopy(loss).to_empty(device=meta_device)
                        losses_list.append(meta_loss((out,), (target,)))
                    return sum(losses_list)

                loss_forward_func = loss_forward
                if 'gmm_vae' in self.model_name:
                    loss_forward_func = None
                self.flops_per_batch = measure_flops(meta_network, sample_forward, loss_fn=loss_forward_func)

        if self.ckpt_path is not None:
            self._load_network_weights()

        if not isinstance(self.detector, AnomalyDetector):
            logger.info("Creating AnomalyDetector.")
            if self.network is not None:
                self.detector = self.detector(self.network)
            else:
                self.detector = self.detector()

        if self.style_transfer is not None:
            self.style_transfer.setup()

    def _process_batch_with_compile(self, b_inputs):
        """Run the network, optionally using ``torch.compile``.

        Args:
            b_inputs (Any): Batch inputs passed to the network.

        Returns:
            Any: Network outputs.

        Raises:
            RuntimeError: If compilation is requested but unsupported.
            UserError: If compilation fails and fullgraph is enforced.
        """
        if not self._compile_torch_model:
            return self.network(b_inputs)
        if not self._torch_compile_available_for_device():
            if self._compiled_network is not None:
                self._compiled_network = None
            if self._should_trace_compile_call(self._compile_invocation_count + 1):
                logger.debug(
                    "Skipping torch.compile for model=%s on unsupported device=%s.",
                    self.model_name,
                    self._active_device(),
                )
            return self.network(b_inputs)

        call_idx = self._compile_invocation_count + 1
        should_trace = self._should_trace_compile_call(call_idx)
        input_summary = self._summarize_runtime_value(b_inputs) if should_trace else None
        compile_kwargs: Dict[str, Any] = {}
        if self._compile_torch_model_mode:
            compile_kwargs["mode"] = self._compile_torch_model_mode
        compile_kwargs['fullgraph'] = self._compile_torch_model_fullgraph
        try:
            if self._compiled_network is None:
                if should_trace:
                    logger.debug(
                        "Compile call %d: creating torch.compile wrapper for %s "
                        "(mode=%s, fullgraph=%s, inputs=%s, %s).",
                        call_idx,
                        type(self.network).__name__ if self.network is not None else None,
                        self._compile_torch_model_mode,
                        self._compile_torch_model_fullgraph,
                        input_summary,
                        self._cuda_memory_snapshot(),
                    )
                compile_start = time.perf_counter()
                if not hasattr(torch, "compile"):
                    raise RuntimeError("compile_torch_model=True requires torch.compile support")
                self._compiled_network = torch.compile(self.network, **compile_kwargs)
                if should_trace:
                    logger.debug(
                        "Compile call %d: torch.compile wrapper created in %.3fs.",
                        call_idx,
                        time.perf_counter() - compile_start,
                    )
            if should_trace:
                logger.debug(
                    "Compile call %d: invoking compiled network (inputs=%s, %s).",
                    call_idx,
                    input_summary,
                    self._cuda_memory_snapshot(),
                )
            forward_start = time.perf_counter()
            result = self._compiled_network(b_inputs)
            self._compile_invocation_count = call_idx
            if should_trace:
                logger.debug(
                    "Compile call %d: compiled network returned in %.3fs "
                    "(outputs=%s, %s).",
                    call_idx,
                    time.perf_counter() - forward_start,
                    self._summarize_runtime_value(result),
                    self._cuda_memory_snapshot(),
                )
            return result
        except (TorchDynamoException, FailOnRecompileLimitHit, UserError, torch.AcceleratorError) as e:
            self._compile_invocation_count = call_idx
            if self._diagnostic_logging_enabled:
                logger.debug(
                    "Compile call %d failed (mode=%s, fullgraph=%s, inputs=%s, %s).",
                    call_idx,
                    self._compile_torch_model_mode,
                    self._compile_torch_model_fullgraph,
                    input_summary,
                    self._cuda_memory_snapshot(),
                    exc_info=True,
                )
            if self._compile_torch_model_fullgraph:
                logger.error("Fullgraph compilation is forced, but network is not compiled.")
                raise
            else:
                logger.warning(e)
                logger.warning("Compilation failed. Switching to eager mode.")
                self._compile_torch_model = False
                self._compiled_network = None
                return self.network(b_inputs)


    def configure_optimizers(self):
        """Create optimizers and schedulers for each parameter group.

        Returns:
            Any: Optimizers and schedulers tuple or None if no network.
        """
        if self.network is not None:
            parameters = pack_tuple(self.network.grouped_parameters())
            optimizers = [self.optimizer(params) for params in parameters]
            # if self.lr_scheduler is not None:
            #    schedulers = [self.lr_scheduler(optimizer) for optimizer in optimizers]
            # else:
            def _infer_num_devices() -> int:
                # Works across 1 GPU, DDP, etc.
                # In DDP, world_size == number of processes == number of GPUs used.
                if getattr(self.trainer, "world_size", None):
                    return int(self.trainer.world_size)
                # Fallback (single process)
                if getattr(self.trainer, "num_devices", None):
                    return int(self.trainer.num_devices)
                return 1

            per_gpu_batch = self.trainer.datamodule.batch_size
            accumulate = int(getattr(self.trainer, "accumulate_grad_batches", 1))
            num_devices = _infer_num_devices()
            global_batch = per_gpu_batch * num_devices * accumulate
            scale = global_batch / per_gpu_batch

            warmup_steps = 5000
            linear_factor = 100

            total_steps = int(self.trainer.estimated_stepping_batches)
            # Keep warmup step-based and bounded
            use_linear = total_steps >= linear_factor
            warmup_steps = max(1, min(warmup_steps, max(1, total_steps // linear_factor))) if use_linear else 0

            schedulers = []
            for optimizer in optimizers:
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= scale

                cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps))
                if use_linear:
                    warmup = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)
                    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
                else:
                    scheduler = cosine
                schedulers.append({"scheduler": scheduler, "interval": "step"})

            return optimizers, schedulers
        return None

    def forward(self, inputs: torch.FloatTensor) -> torch.FloatTensor:
        """Forward pass through the wrapped network.

        Args:
            inputs (torch.FloatTensor): Input tensor.

        Returns:
            torch.FloatTensor: Network output tensor.
        """
        return self.network((inputs,))

    def on_fit_start(self):
        """Ensure the detector has access to the network when fit starts.

        Returns:
            None: Updates detector model reference if present.
        """
        if self._diagnostic_logging_enabled:
            logger.debug(
                "Fit start: model=%s compile_enabled=%s compile_mode=%s fullgraph=%s "
                "device=%s batch_size=%s num_workers=%s %s",
                self.model_name,
                self._compile_torch_model,
                self._compile_torch_model_mode,
                self._compile_torch_model_fullgraph,
                self.device,
                getattr(self.trainer.datamodule, "batch_size", None),
                getattr(self.trainer.datamodule, "num_workers", None),
                self._cuda_memory_snapshot(),
            )
        if hasattr(self.detector, 'model') and self.network is not None:
            self.detector.model = self.network

    def training_step(self, batch, batch_idx):
        """Run a manual optimization step for each configured loss.

        Args:
            batch (Any): Tuple of inputs and targets from the dataloader.
            batch_idx (int): Training batch index.

        Returns:
            dict: Dictionary of logged training losses.
        """
        b_inputs, b_targets = batch
        optimizers = self.optimizers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        batch_loss = {}
        trace_batch = self._should_trace_training_batch(batch_idx)
        if trace_batch:
            logger.debug(
                "training_step start: epoch=%d batch_idx=%d global_step=%d compile_enabled=%s "
                "inputs=%s targets=%s %s",
                self.current_epoch,
                batch_idx,
                self.global_step,
                self._compile_torch_model,
                self._summarize_runtime_value(b_inputs),
                self._summarize_runtime_value(b_targets),
                self._cuda_memory_snapshot(),
            )
        cudagraph_start = time.perf_counter()
        marked_cudagraph_step = self._mark_cudagraph_step_begin()
        if trace_batch:
            logger.debug(
                "training_step batch_idx=%d: cudagraph_mark_step_begin enabled=%s returned in %.3fs.",
                batch_idx,
                marked_cudagraph_step,
                time.perf_counter() - cudagraph_start,
            )
        for i, (optimizer, loss) in enumerate(zip(optimizers, self.losses)):
            # Forward pass through the wrapped network. pack_tuple guarantees
            # consistent tuple output regardless of the underlying model
            # signature, which keeps loss functions simple.
            if trace_batch:
                logger.debug(
                    "training_step batch_idx=%d optimizer_idx=%d: forward start.",
                    batch_idx,
                    i,
                )
            forward_start = time.perf_counter()
            res = pack_tuple(self._process_batch_with_compile(b_inputs))
            if trace_batch:
                logger.debug(
                    "training_step batch_idx=%d optimizer_idx=%d: forward returned in %.3fs "
                    "(outputs=%s, %s).",
                    batch_idx,
                    i,
                    time.perf_counter() - forward_start,
                    self._summarize_runtime_value(res),
                    self._cuda_memory_snapshot(),
                )

            # Compute the loss for this optimizer/loss pair. We pass epoch
            # information so losses can implement curriculum-like behavior
            # if desired.
            loss_start = time.perf_counter()
            loss_value = loss(res, b_targets, epoch=self.current_epoch, num_epochs=self.trainer.max_epochs)
            if trace_batch:
                logger.debug(
                    "training_step batch_idx=%d optimizer_idx=%d: loss computed in %.3fs "
                    "(loss=%s).",
                    batch_idx,
                    i,
                    time.perf_counter() - loss_start,
                    self._summarize_runtime_value(loss_value),
                )

            # Manual optimization: Lightning does not manage gradients for us
            # when ``automatic_optimization`` is False. We therefore clear
            # gradients, backpropagate only through the parameters owned by
            # this optimizer, and then apply the update step manually.
            optimizer.zero_grad(True)
            opt_params = list(itertools.chain(*(group['params'] for group in optimizer.param_groups)))
            if trace_batch:
                logger.debug(
                    "training_step batch_idx=%d optimizer_idx=%d: backward start.",
                    batch_idx,
                    i,
                )
            backward_start = time.perf_counter()
            self.manual_backward(loss_value, inputs=opt_params)
            if trace_batch:
                logger.debug(
                    "training_step batch_idx=%d optimizer_idx=%d: backward returned in %.3fs.",
                    batch_idx,
                    i,
                    time.perf_counter() - backward_start,
                )
            step_start = time.perf_counter()
            self.clip_gradients(
                optimizer,
                gradient_clip_val=1.0,
                gradient_clip_algorithm="norm",
            )
            optimizer.step()
            if trace_batch:
                logger.debug(
                    "training_step batch_idx=%d optimizer_idx=%d: optimizer step completed in %.3fs "
                    "(%s).",
                    batch_idx,
                    i,
                    time.perf_counter() - step_start,
                    self._cuda_memory_snapshot(),
                )

            # Track scaled losses (scaled by batch size along the configured
            # batch dimension) so that epoch-level aggregation matches the
            # effective number of samples seen.
            batch_loss["train_loss_{}".format(i)] = loss_value.detach()
        batch_loss['train_loss'] = sum(val for name, val in batch_loss.items() if 'train_loss' in name)
        self.log_dict(batch_loss,
            prog_bar=True,  # show in progress bar
            logger=True  # send to logger (TensorBoard, etc.)
        )
        schedulers = self.lr_schedulers()
        if not isinstance(schedulers, list):
            schedulers = [schedulers]
        for scheduler in schedulers:
            scheduler.step()
        if trace_batch:
            logger.debug(
                "training_step end: batch_idx=%d logged_keys=%s %s",
                batch_idx,
                sorted(batch_loss.keys()),
                self._cuda_memory_snapshot(),
            )
        return batch_loss

    def validation_step(self, batch, batch_idx):
        """Evaluate validation metrics without performing optimization.

        Args:
            batch (Any): Tuple of inputs and targets.
            batch_idx (int): Validation batch index (unused).

        Returns:
            dict: Mapping of validation metrics aggregated by Lightning.
        """
        b_inputs, b_targets = batch

        # Reuse the exact forward pass from training to ensure the same
        # preprocessing logic feeds validation metrics.
        res = pack_tuple(self._process_batch_with_compile(b_inputs))
        batch_metrics = {}
        for m_name, m in self.val_metrics.items():
            batch_metrics[m_name] = m(res, b_targets)
        batch_metrics['val_loss'] = sum(val for name, val in batch_metrics.items() if 'val_loss' in name)
        self.log_dict(batch_metrics, logger=True)

        return batch_metrics

    def _fit_detector(self):
        """Fit the detector on the training dataloader.

        Returns:
            None: Detector is fitted and cached state updated.
        """
        logger.debug("Creating dataloader for detector fit.")
        train_loader = self.trainer.datamodule.train_dataloader()
        train_samples = (
            self.trainer.datamodule.num_samples("train")
            if hasattr(self.trainer.datamodule, "num_samples")
            else len(train_loader.dataset)
        )
        logger.info("Fitting detector on %d training samples.", train_samples)
        with self._torch_autocast_disabled():
            self.detector.fit(train_loader,
                              subseq_lengths=self.trainer.datamodule.seq_len('train'),
                              window_size=self.window_size)
        self._detector_fitted = True

    def on_save_checkpoint(self, checkpoint):
        """Populate checkpoint with detector state.

        Args:
            checkpoint (dict): Checkpoint payload to update.

        Returns:
            None: Checkpoint dict is modified in place.
        """
        logger.debug("Saving checkpoint.")
        if self.network is not None:
            for k in list(checkpoint['state_dict'].keys()):
                if k.startswith('detector.model') or k.startswith('_compiled_network'):
                    del checkpoint['state_dict'][k]
        elif hasattr(self.detector, 'model') and self.detector.model is not None:
            checkpoint['detector_model'] = pickle.dumps(self.detector.model, pickle.HIGHEST_PROTOCOL)
        checkpoint['detector_fitted'] = self._detector_fitted

    def on_load_checkpoint(self, checkpoint):
        """Restore detector fit state from the checkpoint payload.

        Args:
            checkpoint (dict): Loaded checkpoint payload.

        Returns:
            None: Detector state flags are updated.
        """
        logger.debug("Loading checkpoint.")
        if hasattr(self.detector, 'model'):
            if self.network is not None:
                self.detector.model = None
            elif 'detector_model' in checkpoint:
                self.detector.model = pickle.loads(checkpoint['detector_model'])
        self._detector_fitted = checkpoint.get('detector_fitted', False)
        if self.style_transfer is not None and not any('style_transfer' in k for k in checkpoint['state_dict']):
            st_trans_dict = self.style_transfer.state_dict(prefix='style_transfer.')
            checkpoint['state_dict'] |= st_trans_dict

    def on_train_end(self):
        """Ensure the anomaly detector is trained before running prediction."""
        if hasattr(self.detector, 'model') and self.network is not None:
            self.detector.model = self.network
        logger.info("Starting anomaly detector fitting.")
        self._fit_detector()
        logger.info("Detector fit completed.")
        ckpt_cb = self.trainer.checkpoint_callback
        logger.info(f"Saving checkpoint to {ckpt_cb}")
        if ckpt_cb is not None:
            ckpt_cb._last_checkpoint_saved = ""
            ckpt_cb.on_train_end(self.trainer, self)

    def _validate_prediction_streaming(self) -> None:
        if not self.predict_on_end_streaming:
            return
        if not self.predict_on_end:
            raise ValueError("predict_on_end_streaming=True requires predict_on_end=True.")
        if self.predict_on_end_stream_adapter is not None:
            raise ValueError(
                "model.prediction_stream.adapter is no longer supported for "
                "predict_on_end_streaming=True. Use model.prediction_stream.class_path."
            )
        if self.predict_on_end_stream_policy_name is not None:
            raise ValueError(
                "model.prediction_stream.policy_name is no longer supported for "
                "predict_on_end_streaming=True. Use model.prediction_stream.class_path."
            )
        if not self.predict_on_end_stream_class_path:
            raise ValueError(
                "predict_on_end_streaming=True requires model.prediction_stream.class_path."
            )

        world_size = int(getattr(self.trainer, "world_size", 1) or 1)
        if world_size != 1:
            raise RuntimeError(
                "predict_on_end_streaming=True is only supported for single-process prediction. "
                f"Got trainer.world_size={world_size}."
            )

    def _create_prediction_stream_state(self):
        from .prediction_stream import PredictionStreamConfig, create_prediction_stream_state

        stream_state = create_prediction_stream_state(
            detector=self.detector,
            device_type=self._active_device_type(),
            config=PredictionStreamConfig(
                adapter=self.predict_on_end_stream_adapter,
                class_path=self.predict_on_end_stream_class_path,
                init_args=self.predict_on_end_stream_init_args,
                policy_name=self.predict_on_end_stream_policy_name,
                window_size=self.window_size,
                patch_size=self.window_size,
            ),
        )
        self._prediction_stream_policy = getattr(stream_state, "policy", stream_state)
        return stream_state

    def _close_prediction_stream_state(self) -> None:
        stream_state = self._prediction_stream_state
        self._prediction_stream_state = None
        self._prediction_stream_policy = None
        if stream_state is not None and hasattr(stream_state, "close"):
            stream_state.close()

    def on_predict_start(self):
        """Prepare the detector before prediction.

        Returns:
            None: Fits detector if needed and updates model reference.
        """
        logger.debug("Starting prediction.")

        if self.network is not None and hasattr(self.detector, 'model'):
            self.detector.model = self.network
        if not self.detector_fitted:
            self.on_train_end()
        self._validate_prediction_streaming()
        self._close_prediction_stream_state()
        if self.predict_on_end_streaming:
            predict_loop = getattr(self.trainer, "predict_loop", None)
            if predict_loop is not None and hasattr(predict_loop, "return_predictions"):
                predict_loop.return_predictions = False
            self._prediction_stream_state = self._create_prediction_stream_state()

    def predict_step(self, batch, batch_idx):
        """Compute anomaly scores for a single batch during prediction.

        Args:
            batch (Any): Mini-batch containing inputs and targets.
            batch_idx (int): Prediction batch index (unused).

        Returns:
            torch.Tensor: Scores tensor for the batch, or empty tensor on non-zero ranks.
        """
        b_inputs, b_targets = batch
        if self.style_transfer is not None:
            b_inputs = pack_tuple(self.style_transfer(b_inputs[0]))
        with self._torch_autocast_disabled():
            if self.predict_on_end and self.predict_on_end_streaming:
                if self._prediction_stream_state is None:
                    raise RuntimeError("Prediction stream state was not initialized before predict_step.")
                self._prediction_stream_state.add(b_inputs[0])
                return torch.empty(0)
            if self.predict_on_end:
                return b_inputs[0]
            scores = self.detector(b_inputs)
        if scores is None:
            return torch.empty(0)
        return scores.detach().cpu()

    def align_prediction_outputs(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        orig_seq_len: List[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pad raw detector scores back to row-aligned sequence outputs."""
        pred_offset = self.prediction_horizon if self.prediction_horizon is not None else 0
        start_offset = self.window_size + pred_offset - 1 + self.label_index_offset
        end_offset = max(0, pred_offset - 1)
        effective_seq_len = [sl - start_offset - end_offset for sl in orig_seq_len]
        padded_scores = pad_each_subsequence(
            scores,
            effective_seq_len,
            pad_prefix=start_offset,
            pad_suffix=end_offset,
            value=-torch.inf,
        )
        if padded_scores.shape != labels.shape:
            raise RuntimeError(
                f"Aligned score shape {padded_scores.shape} does not match labels {labels.shape}."
            )
        return self._tensor_to_numpy(padded_scores), self._tensor_to_numpy(labels)

    def _scores_to_cpu_tensor(self, scores: Any) -> torch.Tensor:
        if isinstance(scores, torch.Tensor):
            return scores.detach().cpu()
        return torch.as_tensor(scores).detach().cpu()

    @staticmethod
    def _tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy()

    def on_predict_epoch_end(self):
        """Aggregate predictions and compute alarm-based evaluation metrics.

        Returns:
            None: Updates cached metrics and anomaly scores/labels.
        """
        with self._torch_autocast_disabled():
            if self.predict_on_end and self.predict_on_end_streaming:
                if self._prediction_stream_state is None:
                    raise RuntimeError("Prediction stream state was not initialized before epoch end.")
                try:
                    scores = self._prediction_stream_state.finalize()
                finally:
                    self._close_prediction_stream_state()
            elif self.predict_on_end:
                inputs = self.trainer.predict_loop.predictions
                inputs = torch.cat(inputs, dim=0).detach().cpu()
                scores = self.detector((inputs,))
            else:
                scores = self.trainer.predict_loop.predictions
                scores = torch.cat(scores, dim=0)
            if scores is not None:
                scores = self._scores_to_cpu_tensor(scores)
            if scores is None or scores.numel() == 0:
                # TODO: Rework
                labels, scores = self.detector.get_labels_and_scores(self.trainer.datamodule.predict_dataloader(),
                                                                     subseq_lengths=self.trainer.datamodule.seq_len('test'),
                                                                     window_size=self.window_size)
                scores = self._scores_to_cpu_tensor(scores)
        labels = []
        for _, label in self.trainer.datamodule.predict_orig_dataloader():
            labels.append(label[0].select(self.batch_dimension, 0))
        labels = torch.cat(labels, dim=0)
        orig_seq_len = self.trainer.datamodule.seq_len('test')
        pred_offset = self.prediction_horizon if self.prediction_horizon is not None else 0

        start_offset = self.window_size + pred_offset - 1 + self.label_index_offset
        end_offset = max(0, pred_offset - 1)
        seq_len = [sl - start_offset - end_offset for sl in orig_seq_len]

        if self.use_full_seq_prediction:
            labels_formatted = labels.clone()
            scores = pad_each_subsequence(
                scores,
                seq_len,
                pad_prefix=start_offset,
                pad_suffix=end_offset,
                value=-torch.inf,
            )
            seq_len = orig_seq_len
        else:
            labels_formatted = torch.cat(
                [t[start_offset:-end_offset or None] for t in torch.split(labels, orig_seq_len)]
            )

        assert scores.shape == labels_formatted.shape and scores.numel() == labels_formatted.numel(), \
            f"{scores.shape} != {labels_formatted.shape}"

        score_sequences = [self._tensor_to_numpy(segment) for segment in torch.split(scores, seq_len)]
        label_sequences = [self._tensor_to_numpy(segment) for segment in torch.split(labels_formatted, seq_len)]

        raw_scores = self._tensor_to_numpy(insert_before_subsequences(scores, seq_len, insert_value=-torch.inf))
        labels_formatted = self._tensor_to_numpy(insert_before_subsequences(labels_formatted, seq_len, insert_value=0))
        labels = self._tensor_to_numpy(insert_before_subsequences(labels, orig_seq_len, insert_value=0))
        labels_bin = (labels != 0).astype(np.int32, copy=False)
        labels_bin_formatted = (labels_formatted != 0).astype(np.int32, copy=False)

        def get_labels(metric_name: str) -> np.ndarray:
            if is_metric_binary(metric_name):
                return labels_bin_formatted
            return labels_formatted

        self._anomaly_scores = raw_scores
        self._anomaly_labels = labels

        if self._evaluation_postprocessing.enabled:
            logger.info("Selecting threshold with event-style evaluation postprocessing.")
            postprocessed = evaluate_sequences_with_optional_postprocessing(
                score_sequences,
                label_sequences,
                metric_names=tuple(self._test_metrics.keys()),
                config=self._evaluation_postprocessing,
                fix_threshold=self.fix_threshold,
                dataset_name=self.trainer.datamodule.dataset_name,
            )
            predictions = postprocessed.predictions
            metric_results = dict(postprocessed.metric_values)
        else:
            logger.info('Selecting threshold based on %s metric:', self._target_metric)
            if self.fix_threshold:
                threshold = np.quantile(
                    raw_scores,
                    1 - DATASET_FRACTIONS[self.trainer.datamodule.dataset_name],
                )
            else:
                threshold = select_threshold(raw_scores, get_labels(self._target_metric)).best_threshold
            predictions = (raw_scores > threshold).astype(np.int32)
            metric_results = {
                name: metric(predictions, get_labels(name))
                for name, metric in self._test_metrics.items()
            }

        for name, metric in self._test_metrics.items():
            c_metric = metric_results[name]
            best_metric = metric(labels_bin, labels_bin if is_metric_binary(name) else labels)
            self._test_metrics_results[name] = (c_metric, best_metric)
            # The label-mask reference is not a valid bound for alarm-frequency and
            # detection-timing metrics, so only assert for metrics with comparable
            # reference predictions.
            if name not in ['aaf', 'edf', 'ldf']:
                assert c_metric <= best_metric, f"{name}: {c_metric} > {best_metric}"
        logger.info("Finished prediction.")

    def teardown(self, stage: str) -> None:
        if stage == "predict":
            self._close_prediction_stream_state()
        super().teardown(stage)
