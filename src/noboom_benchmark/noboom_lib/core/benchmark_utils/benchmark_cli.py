"""Helper around :class:`~lightning.pytorch.cli.LightningCLI` for the benchmark."""

from pathlib import Path
from typing import Optional

from lightning.pytorch.callbacks import ModelCheckpoint, ThroughputMonitor
from lightning.pytorch.cli import LightningCLI

from .benchmark_dataset import BenchmarkDataset
from .benchmark_model import BenchmarkModel
from .transform_compiler import DatasetTransformCompiler
from .model_utils import get_args
from ..tune_constants import CPU_ONLY_MODELS, uses_lightning_optimizer


class BatchDimThroughputMonitor(ThroughputMonitor):
    def __init__(self, batch_dim: int):
        super().__init__(batch_size_fn=lambda batch: batch[0][0].size(batch_dim))


def _effective_model_window_size(model_name: str, window_size: Optional[int]) -> Optional[int]:
    if window_size is None or model_name not in {"energyad_autoformer", "energyad_fedformer"}:
        return window_size
    if window_size < 2:
        raise ValueError("EnergyAD models require window_size >= 2 when using finite differences.")
    return window_size - 1


def _effective_transform_window_size(
    model_name: str, split_name: str, window_size: Optional[int]
) -> Optional[int]:
    return window_size


class BenchmarkCLI(LightningCLI):
    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        data_manifest_path: Optional[str],
        config_dir: str | Path,
        ckpt_dir: str | Path,
        requested_dataset_name: Optional[str] = None,
        hp_cfg: str = None,
        lr_scale: float = 1.0,
        max_epochs: Optional[int] = None,
    ):
        """Wire together model and dataset configs for the Lightning CLI.

        Args:
            model_name (str): Benchmark model identifier used to load configuration.
            dataset_name (str): Base dataset name passed through to the data module.
            data_manifest_path (Optional[str]): Optional pre-resolved prepared manifest.
            config_dir (str | Path): Directory containing model YAML configs.
            ckpt_dir (str | Path): Folder where checkpoints are written by callbacks.
            hp_cfg (str | None): Optional hyperparameter override config. Defaults to None.
        """
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.requested_dataset_name = requested_dataset_name or dataset_name
        self.data_manifest_path = data_manifest_path
        self.ckpt_dir = ckpt_dir
        self.lr_scale = lr_scale
        self.max_epochs = max_epochs
        self._window_size_deps = []

        args = ['--config', f'{config_dir}/models/common.yaml', '--config', f'{config_dir}/models/{model_name}.yaml',
                '--requested_dataset_name', self.requested_dataset_name]

        if data_manifest_path is not None:
            args.extend(['--data.data_manifest_path', data_manifest_path])

        if hp_cfg is not None:
            args.append('--config')
            args.append(hp_cfg)

        super().__init__(BenchmarkModel, BenchmarkDataset, run=False, save_config_callback=None,
                       auto_configure_optimizers=False, parser_kwargs={"parser_mode": "omegaconf"},
                       args=args)

    def add_arguments_to_parser(self, parser):
        """Expose CLI parameters and link them into the configuration tree.

        Args:
            parser (Any): LightningCLI parser instance to enrich.

        Returns:
            None: Parser is modified in place.
        """
        parser.add_lightning_class_args(ModelCheckpoint, "model_checkpoint")
        parser.set_defaults({"model_checkpoint.dirpath": self.ckpt_dir,
                             "model_checkpoint.save_last": True,
                             "model_checkpoint.save_top_k": 0,
                             "model_checkpoint.verbose": True
                             })

        if self.model_name not in CPU_ONLY_MODELS and 'timesnet' not in self.model_name:
            parser.add_lightning_class_args(BatchDimThroughputMonitor, "throughput_monitor")
            parser.link_arguments('data.batch_dim', 'throughput_monitor.batch_dim', apply_on='parse')

        parser.add_argument('--model_name', type=str, default=self.model_name, help='Model name.')
        parser.add_argument('--dataset_name', type=str, default=self.dataset_name, help='Dataset name.')
        parser.add_argument('--requested_dataset_name', type=str, default=self.requested_dataset_name)
        parser.add_argument('--window_size', type=int, help='Window size.')
        parser.add_argument('--prediction_horizon', type=Optional[int], help='Prediction horizon. Used in some models')

        parser.link_arguments('dataset_name', 'data.dataset_name', apply_on='parse')
        parser.link_arguments('requested_dataset_name', 'data.requested_dataset_name', apply_on='parse')
        parser.link_arguments('data.batch_dim', 'model.batch_dim', apply_on='parse')
        parser.link_arguments('model_name', 'model.model_name', apply_on='parse')
        parser.link_arguments('window_size', 'model.window_size', apply_on='parse')
        parser.link_arguments('prediction_horizon', 'model.prediction_horizon', apply_on='parse')
        parser.link_arguments('data.num_features', 'model.num_features', apply_on='instantiate')
        for from_name, to_name, apply_on in get_args(self.model_name):
            parser.link_arguments(f'{from_name}', f'{to_name}', apply_on=apply_on)
            if from_name == 'window_size':
                self._window_size_deps.append(to_name)

    def before_instantiate_classes(self):
        """Prepare transforms and defaults before model and data creation.

        Returns:
            None: Updates the configuration in place.
        """
        cfg = self.config

        if uses_lightning_optimizer(self.model_name):
            cfg['model']['optimizer']['init_args']['lr'] *= self.lr_scale
        if self.max_epochs is not None:
            cfg['trainer']['max_epochs'] = min(cfg['trainer']['max_epochs'], self.max_epochs)

        effective_model_window_size = _effective_model_window_size(self.model_name, cfg.window_size)
        cfg.model.window_size = cfg.window_size
        for to_name in self._window_size_deps:
            cfg[to_name] = effective_model_window_size

        # Create aug transforms
        for split_name in ['train', 'val', 'test']:
            split = f'{split_name}_transform'

            if split not in cfg.data:
                cfg.data[split] = None
            else:
                transform_list = cfg.data[split]
                if transform_list:
                    transform_window_size = _effective_transform_window_size(
                        self.model_name, split_name, cfg.window_size
                    )
                    cfg.data[split] = DatasetTransformCompiler.compile(
                        steps=transform_list,
                        window_size=transform_window_size,
                        prediction_horizon=cfg.prediction_horizon,
                    )

        if not getattr(cfg.data, "data_manifest_path", None) and not getattr(cfg.data, "inference_dataset_path", None):
            raise ValueError(
                "data.data_manifest_path must be set before BenchmarkCLI runs. "
                "Prepare dataset manifests outside Ray Tune trial threads and pass them into the run config."
            )

        # Ensure callback outputs (e.g., ModelCheckpoint) land in a predictable
        # directory so worker nodes and the head node agree on where Lightning
        # artifacts live.
        if not getattr(cfg.trainer, "default_root_dir", None):
            cfg.trainer.default_root_dir = str(
                Path.home()
                / ".cache"
                / "noboom_benchmark"
                / "lightning_logs"
                / f"{cfg.model_name}__{cfg.dataset_name}"
            )

        for callback in getattr(cfg.trainer, "callbacks", []):
            if getattr(callback, "class_path", "") == "lightning.pytorch.callbacks.ModelCheckpoint":
                callback.init_args.setdefault(
                    "dirpath", str(Path(cfg.trainer.default_root_dir) / "checkpoints")
                )
