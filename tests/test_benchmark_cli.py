import importlib
from pathlib import Path
import sys
import types


PACKAGE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "noboom_benchmark"
    / "noboom_lib"
    / "core"
    / "benchmark_utils"
)
PACKAGE_NAME = "noboom_benchmark.noboom_lib.core.benchmark_utils"


def _install_benchmark_cli_stubs() -> None:
    package = types.ModuleType(PACKAGE_NAME)
    package.__path__ = [str(PACKAGE_ROOT)]
    sys.modules[PACKAGE_NAME] = package

    benchmark_dataset_module = types.ModuleType(f"{PACKAGE_NAME}.benchmark_dataset")
    benchmark_dataset_module.BenchmarkDataset = type("BenchmarkDataset", (), {})
    sys.modules[benchmark_dataset_module.__name__] = benchmark_dataset_module

    benchmark_model_module = types.ModuleType(f"{PACKAGE_NAME}.benchmark_model")
    benchmark_model_module.BenchmarkModel = type("BenchmarkModel", (), {})
    sys.modules[benchmark_model_module.__name__] = benchmark_model_module

    transform_compiler_module = types.ModuleType(f"{PACKAGE_NAME}.transform_compiler")

    class _DatasetTransformCompiler:
        @staticmethod
        def compile(*args, **kwargs):
            return None

    transform_compiler_module.DatasetTransformCompiler = _DatasetTransformCompiler
    sys.modules[transform_compiler_module.__name__] = transform_compiler_module


_install_benchmark_cli_stubs()
BenchmarkCLI = importlib.import_module(f"{PACKAGE_NAME}.benchmark_cli").BenchmarkCLI
for module_name in [
    PACKAGE_NAME,
    f"{PACKAGE_NAME}.benchmark_cli",
    f"{PACKAGE_NAME}.benchmark_dataset",
    f"{PACKAGE_NAME}.benchmark_model",
    f"{PACKAGE_NAME}.transform_compiler",
]:
    sys.modules.pop(module_name, None)


class _FakeParser:
    def __init__(self) -> None:
        self.defaults = {}
        self.links = []

    def add_lightning_class_args(self, *_args, **_kwargs) -> None:
        return None

    def set_defaults(self, defaults) -> None:
        self.defaults.update(defaults)

    def add_argument(self, *_args, **_kwargs) -> None:
        return None

    def link_arguments(self, *args, **kwargs) -> None:
        self.links.append((args, kwargs))


def test_benchmark_cli_configures_last_checkpoint_only() -> None:
    cli = BenchmarkCLI.__new__(BenchmarkCLI)
    cli.model_name = "neutralad"
    cli.dataset_name = "cont_reactive_ome"
    cli.requested_dataset_name = "cont_reactive_ome"
    cli.ckpt_dir = "/tmp/checkpoints"
    cli._window_size_deps = []

    parser = _FakeParser()

    BenchmarkCLI.add_arguments_to_parser(cli, parser)

    assert parser.defaults["model_checkpoint.dirpath"] == "/tmp/checkpoints"
    assert parser.defaults["model_checkpoint.save_last"] is True
    assert parser.defaults["model_checkpoint.save_top_k"] == 0


def test_energyad_filonovlstm_does_not_link_window_size_into_network() -> None:
    cli = BenchmarkCLI.__new__(BenchmarkCLI)
    cli.model_name = "energyad_filonovlstm"
    cli.dataset_name = "cont_reactive_ome_ext"
    cli.requested_dataset_name = "cont_reactive_ome_ext"
    cli.ckpt_dir = "/tmp/checkpoints"
    cli._window_size_deps = []

    parser = _FakeParser()

    BenchmarkCLI.add_arguments_to_parser(cli, parser)

    linked_window_targets = {
        args[1]
        for args, _kwargs in parser.links
        if len(args) >= 2 and args[0] == "window_size"
    }

    assert "model.network.init_args.window_size" not in linked_window_targets


def test_hpad_links_window_size_and_num_features_into_network() -> None:
    cli = BenchmarkCLI.__new__(BenchmarkCLI)
    cli.model_name = "hpad"
    cli.dataset_name = "industry_process"
    cli.requested_dataset_name = "industry_process"
    cli.ckpt_dir = "/tmp/checkpoints"
    cli._window_size_deps = []

    parser = _FakeParser()

    BenchmarkCLI.add_arguments_to_parser(cli, parser)

    link_pairs = {
        (args[0], args[1])
        for args, _kwargs in parser.links
        if len(args) >= 2
    }
    assert ("window_size", "model.network.init_args.window_size") in link_pairs
    assert ("data.num_features", "model.network.init_args.input_dim") in link_pairs
