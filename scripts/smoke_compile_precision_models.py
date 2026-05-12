#!/usr/bin/env python3
"""Smoke-test model compile and mixed-precision compatibility.

This script is intentionally small-batch and config-driven. It is designed for
server use after checking GPU availability, but also runs on CPU for Dynamo and
autocast coverage in lightweight local environments.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import json
import os
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "src" / "noboom_cluster" / "cluster_files" / "configs" / "models"
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


@dataclass(frozen=True)
class CompileSpec:
    fullgraph: bool
    mode: Optional[str]

    @property
    def label(self) -> str:
        mode = self.mode if self.mode is not None else "default"
        return f"fullgraph={self.fullgraph}, mode={mode}"


COMPILE_CANDIDATES = (
    CompileSpec(fullgraph=True, mode="max-autotune"),
    CompileSpec(fullgraph=True, mode="max-autotune-no-cudagraphs"),
    CompileSpec(fullgraph=True, mode=None),
    CompileSpec(fullgraph=False, mode="max-autotune"),
    CompileSpec(fullgraph=False, mode="max-autotune-no-cudagraphs"),
    CompileSpec(fullgraph=False, mode=None),
)


def _install_timesead_stubs_if_missing() -> None:
    try:
        timesead_spec = importlib.util.find_spec("timesead")
    except ValueError:
        timesead_spec = None
    if timesead_spec is not None:
        return

    timesead_module = sys.modules.setdefault("timesead", types.ModuleType("timesead"))
    models_module = sys.modules.setdefault("timesead.models", types.ModuleType("timesead.models"))
    common_module = sys.modules.setdefault("timesead.models.common", types.ModuleType("timesead.models.common"))
    anomaly_detector_module = sys.modules.setdefault(
        "timesead.models.common.anomaly_detector",
        types.ModuleType("timesead.models.common.anomaly_detector"),
    )
    optim_module = sys.modules.setdefault("timesead.optim", types.ModuleType("timesead.optim"))
    loss_module = sys.modules.setdefault("timesead.optim.loss", types.ModuleType("timesead.optim.loss"))
    layers_module = sys.modules.setdefault("timesead.models.layers", types.ModuleType("timesead.models.layers"))
    autoformer_module = sys.modules.setdefault(
        "timesead.models.layers.autoformer_encdec",
        types.ModuleType("timesead.models.layers.autoformer_encdec"),
    )
    autocorrelation_module = sys.modules.setdefault(
        "timesead.models.layers.autocorrelation",
        types.ModuleType("timesead.models.layers.autocorrelation"),
    )
    embed_module = sys.modules.setdefault(
        "timesead.models.layers.embed",
        types.ModuleType("timesead.models.layers.embed"),
    )
    fourier_module = sys.modules.setdefault(
        "timesead.models.layers.fourier_correlation",
        types.ModuleType("timesead.models.layers.fourier_correlation"),
    )

    class BaseModel(nn.Module):
        pass

    class AnomalyDetector(nn.Module):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            super().__init__()
            self.register_buffer("dummy", torch.tensor([]), persistent=False)

        def forward(self, inputs: Any) -> torch.Tensor:
            return self.compute_online_anomaly_score(inputs)

    class PredictionAnomalyDetector(AnomalyDetector):
        pass

    class Loss(nn.Module):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__()
            self.reduction = kwargs.get("reduction", "mean")

    class DataEmbedding(nn.Module):
        def __init__(self, c_in: int, d_model: int, dropout: float = 0.0, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            super().__init__()
            self.linear = nn.Linear(c_in, d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.dropout(self.linear(x))

    class FourierBlock(nn.Module):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            super().__init__()

        def forward(
            self,
            queries: torch.Tensor,
            keys: torch.Tensor,
            values: torch.Tensor,
            mask: Optional[torch.Tensor],
        ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
            del queries, keys, mask
            return values, None

    class AutoCorrelationLayer(nn.Module):
        def __init__(self, correlation: nn.Module, d_model: int, n_heads: int, *args: Any, **kwargs: Any) -> None:
            del d_model, n_heads, args, kwargs
            super().__init__()
            self.inner = correlation

        def forward(
            self,
            queries: torch.Tensor,
            keys: torch.Tensor,
            values: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
        ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
            return self.inner(queries, keys, values, attn_mask)

    class EncoderLayer(nn.Module):
        def __init__(self, attention: nn.Module, d_model: int, d_ff: Optional[int] = None, *args: Any, **kwargs: Any) -> None:
            del d_ff, args, kwargs
            super().__init__()
            self.attention = attention
            self.feed_forward = nn.Linear(d_model, d_model)

        def forward(
            self,
            x: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
        ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
            attended, attn = self.attention(x, x, x, attn_mask)
            return self.feed_forward(attended), attn

    class Encoder(nn.Module):
        def __init__(
            self,
            attn_layers: Sequence[nn.Module],
            conv_layers: Optional[Sequence[nn.Module]] = None,
            norm_layer: Optional[nn.Module] = None,
        ) -> None:
            del conv_layers
            super().__init__()
            self.attn_layers = nn.ModuleList(attn_layers)
            self.norm = norm_layer

        def forward(
            self,
            x: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
        ) -> tuple[torch.Tensor, list[Optional[torch.Tensor]]]:
            attns = []
            for layer in self.attn_layers:
                x, attn = layer(x, attn_mask=attn_mask)
                attns.append(attn)
            if self.norm is not None:
                x = self.norm(x)
            return x, attns

    class CustomLayerNorm(nn.LayerNorm):
        pass

    models_module.BaseModel = BaseModel
    common_module.AnomalyDetector = AnomalyDetector
    anomaly_detector_module.AnomalyDetector = AnomalyDetector
    anomaly_detector_module.PredictionAnomalyDetector = PredictionAnomalyDetector
    loss_module.Loss = Loss
    embed_module.DataEmbedding = DataEmbedding
    fourier_module.FourierBlock = FourierBlock
    autocorrelation_module.AutoCorrelationLayer = AutoCorrelationLayer
    autoformer_module.CustomLayerNorm = CustomLayerNorm
    autoformer_module.Encoder = Encoder
    autoformer_module.EncoderLayer = EncoderLayer

    timesead_module.models = models_module
    timesead_module.optim = optim_module
    models_module.common = common_module
    models_module.layers = layers_module
    common_module.anomaly_detector = anomaly_detector_module
    optim_module.loss = loss_module
    layers_module.autoformer_encdec = autoformer_module
    layers_module.autocorrelation = autocorrelation_module
    layers_module.embed = embed_module
    layers_module.fourier_correlation = fourier_module


def _resolve_class(class_path: str) -> type:
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _filter_init_args(cls: type, init_args: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(cls)
    parameters = signature.parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return init_args
    return {key: value for key, value in init_args.items() if key in parameters}


def _load_config(model_name: str) -> dict[str, Any]:
    with (CONFIG_ROOT / f"{model_name}.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _iter_model_names(requested: Optional[Sequence[str]]) -> list[str]:
    if requested:
        return list(requested)
    names = []
    for path in sorted(CONFIG_ROOT.glob("*.yaml")):
        config = _load_config(path.stem)
        network = config.get("model", {}).get("network")
        if not isinstance(network, dict):
            continue
        class_path = str(network.get("class_path", ""))
        if class_path.startswith("noboom_benchmark.noboom_lib.core.models."):
            names.append(path.stem)
    return names


def _compact_error(error: BaseException) -> str:
    return f"{type(error).__name__}: {str(error).splitlines()[0][:240]}"


def _small_init_args(model_name: str, init_args: dict[str, Any]) -> dict[str, Any]:
    args = {key: value for key, value in init_args.items() if value is not None}
    for key in ("input_dim", "enc_in", "c_out", "feats"):
        if key in init_args:
            args[key] = 3
    for key in ("window_size", "win_size", "seq_len", "n_window"):
        if key in init_args:
            args[key] = 16

    if model_name in {"dlinear", "nlinear"}:
        args.update(input_dim=3, seq_len=16)
        if model_name == "dlinear":
            args["moving_avg"] = 3
    elif model_name == "autoencoder":
        args.update(input_dim=3, seq_len=16, hidden_dims=[8], latent_dim=4, dropout=0.0)
    elif model_name in {"itransformer", "patchtst", "dualtf"}:
        args.update(input_dim=3, seq_len=16, d_model=8, e_layers=1, n_heads=1, d_ff=16, dropout=0.0)
        if model_name == "patchtst":
            args.update(patch_len=4, stride=2)
    elif model_name == "modern_tcn":
        args.update(input_dim=3, seq_len=16, dims=[8], num_blocks=[1], large_size=[3], small_size=[3], dropout=0.0)
    elif model_name == "tfad":
        args.update(input_dim=3, seq_len=16, n_window=16, embedding_rep_dim=8, tcn_kernel_size=3, tcn_layers=1, dropout=0.0)
    elif model_name == "kan_ad":
        args.update(window_size=16, input_dim=3, prediction_horizon=1)
    elif model_name == "igad":
        args.update(window_size=16, input_dim=3, hidden_dims=[8], latent_dim=4, dropout=0.0)
    elif model_name == "alora":
        args.update(win_size=16, input_dim=3, d_model=3, n_heads=1, e_layers=1, dropout=0.0, top_k_limit=8)
    elif model_name == "scatterad":
        args.update(input_dim=3, win_size=16, hidden_dim=8, num_layers=1, heads=1, dropout=0.0)
    elif model_name == "anomaly_transformer":
        args.update(win_size=16, enc_in=3, c_out=3, d_model=8, n_heads=1, e_layers=1, d_ff=16, dropout=0.0)
    elif model_name == "dcdetector":
        args.update(win_size=16, input_dim=3, patch_size=[4], n_heads=1, e_layers=1, d_model=8)
    elif model_name == "catch":
        args.update(
            input_dim=3,
            seq_len=16,
            cf_dim=8,
            d_ff=16,
            d_model=8,
            e_layers=1,
            head_dim=8,
            inference_patch_size=4,
            n_heads=1,
            patch_size=4,
            patch_stride=2,
            dropout=0.0,
            head_dropout=0.0,
        )
    elif model_name == "rtdetector":
        args.update(feats=3, window_size=16, dropout=0.0)
    elif model_name == "carots":
        args.update(
            input_dim=3,
            win_size=16,
            input_step=15,
            pred_step=1,
            hidden_dim=8,
            projector_hidden_dim=8,
            projector_output_dim=8,
            cuts_hidden_dim=4,
            dropout=0.0,
        )
    elif model_name == "paano":
        args.update(input_dim=3, patch_size=16, layers=[8], kernel_sizes=[3], projection_dim=8, use_revin=False)
    elif model_name == "oraclead":
        args.update(input_dim=3, win_size=16, hidden_dim=8, num_layers=1, num_heads=1, dropout=0.0)
    elif model_name == "hpad":
        args.update(
            window_size=16,
            input_dim=3,
            model_dim=8,
            num_heads=1,
            fcn_dim=16,
            encoder_layers=1,
            modes=4,
            patch_scales=[1, 2],
            num_patch_prototypes=4,
            top_k_periods=2,
            num_period_prototypes=3,
        )
    return {key: value for key, value in args.items() if not (isinstance(value, str) and value.startswith("${"))}


def _instantiate_model(model_name: str) -> nn.Module:
    config = _load_config(model_name)
    network_config = config["model"]["network"]
    cls = _resolve_class(network_config["class_path"])
    init_args = _small_init_args(model_name, dict(network_config.get("init_args", {})))
    return cls(**_filter_init_args(cls, init_args))


def _instantiate_loss(model_name: str) -> nn.Module:
    config = _load_config(model_name)
    loss_config = config["model"].get("losses")
    if not loss_config:
        return nn.MSELoss()
    cls = _resolve_class(loss_config["class_path"])
    init_args = dict(loss_config.get("init_args") or {})
    return cls(**_filter_init_args(cls, init_args))


def _make_batch(model_name: str, device: torch.device) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    if model_name == "kan_ad":
        inputs = torch.randn(16, 2, 3, device=device)
        target = torch.randn(1, 2, 3, device=device)
        return (inputs,), (target,)
    inputs = torch.randn(2, 16, 3, device=device)
    return (inputs,), (inputs.detach().clone(),)


def _first_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, dict):
        value = output.get("reconstruction")
        if isinstance(value, torch.Tensor):
            return value
        raise TypeError("Dictionary output has no tensor reconstruction entry.")
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError("Could not find tensor output for generic loss.")


def _call_loss(loss: nn.Module, output: Any, targets: tuple[torch.Tensor, ...]) -> torch.Tensor:
    if type(loss).__module__.startswith("torch.nn"):
        return loss(_first_tensor(output), targets[0])
    predictions = output if isinstance(output, tuple) else (output,)
    return loss.forward(predictions, targets)


def _finite_gradients(parameters: Iterable[nn.Parameter]) -> bool:
    for parameter in parameters:
        if parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all():
            return False
    return True


def _compile(model: nn.Module, spec: Optional[CompileSpec]) -> nn.Module:
    if spec is None:
        return model
    kwargs: dict[str, Any] = {"fullgraph": spec.fullgraph}
    if spec.mode is not None:
        kwargs["mode"] = spec.mode
    return torch.compile(model, **kwargs)


def _training_smoke(
    model_name: str,
    compile_spec: Optional[CompileSpec],
    precision: str,
    device: torch.device,
    steps: int,
) -> tuple[bool, str, Optional[float]]:
    torch.manual_seed(17)
    if hasattr(torch, "_dynamo"):
        torch._dynamo.reset()
    model = _instantiate_model(model_name).to(device=device)
    loss = _instantiate_loss(model_name).to(device=device)
    model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=1.0e-3) if trainable else None
    compiled_model = _compile(model, compile_spec)
    autocast_enabled = precision == "bf16-mixed"
    device_type = "cuda" if device.type == "cuda" else "cpu"
    losses = []
    for _step in range(steps):
        inputs, targets = _make_batch(model_name, device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=autocast_enabled):
            output = compiled_model(inputs)
            value = _call_loss(loss, output, targets)
        if not torch.isfinite(value.detach()):
            return False, "non-finite loss", None
        value.backward()
        if not _finite_gradients(trainable):
            return False, "non-finite gradient", float(value.detach().cpu())
        if optimizer is not None:
            optimizer.step()
        losses.append(float(value.detach().cpu()))
    if len(losses) >= 2 and losses[-1] > max(losses[0] * 10.0, losses[0] + 100.0):
        return False, f"loss blew up from {losses[0]:.6g} to {losses[-1]:.6g}", losses[-1]
    return True, "ok", losses[-1] if losses else None


def _select_device(force_cpu: bool = False) -> torch.device:
    if force_cpu or not torch.cuda.is_available():
        print(json.dumps({"device": "cpu", "reason": "CUDA unavailable or --cpu set"}))
        return torch.device("cpu")
    try:
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        visible_physical_indices: Optional[list[int]] = None
        if visible_devices:
            parsed_indices = []
            for part in visible_devices.split(","):
                part = part.strip()
                if part.isdigit():
                    parsed_indices.append(int(part))
            if parsed_indices:
                visible_physical_indices = parsed_indices
        query = [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        completed = subprocess.run(query, check=True, capture_output=True, text=True, timeout=10)
        rows = []
        for line in completed.stdout.strip().splitlines():
            index, name, used, total, util = [part.strip() for part in line.split(",")]
            rows.append(
                {
                    "index": int(index),
                    "name": name,
                    "memory_used_mb": int(used),
                    "memory_total_mb": int(total),
                    "utilization_gpu_percent": int(util),
                }
            )
        if not rows:
            print(json.dumps({"device": "cuda:0", "reason": "nvidia-smi returned no rows"}))
            return torch.device("cuda:0")
        candidate_rows = rows
        if visible_physical_indices is not None:
            candidate_rows = [row for row in rows if row["index"] in visible_physical_indices]
        if not candidate_rows:
            candidate_rows = rows
        selected = min(candidate_rows, key=lambda row: row["memory_used_mb"])
        cuda_index = selected["index"]
        if visible_physical_indices is not None and selected["index"] in visible_physical_indices:
            cuda_index = visible_physical_indices.index(selected["index"])
        print(
            json.dumps(
                {
                    "gpu_inventory": rows,
                    "visible_physical_indices": visible_physical_indices,
                    "selected_gpu": selected,
                    "torch_device": f"cuda:{cuda_index}",
                },
                indent=2,
            )
        )
        return torch.device(f"cuda:{cuda_index}")
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        print(json.dumps({"device": "cuda:0", "reason": f"nvidia-smi unavailable: {_compact_error(error)}"}))
        return torch.device("cuda:0")


def run_model(model_name: str, device: torch.device, steps: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "model": model_name,
        "compile_enabled": False,
        "compile_mode": None,
        "compile_fullgraph": None,
        "precision": "fp32",
        "limitation": None,
    }
    compile_errors = []
    best_spec: Optional[CompileSpec] = None
    for spec in COMPILE_CANDIDATES:
        try:
            ok, message, loss_value = _training_smoke(model_name, spec, "fp32", device, steps)
        except Exception as error:
            ok = False
            message = _compact_error(error)
            loss_value = None
        if ok:
            best_spec = spec
            result.update(
                compile_enabled=True,
                compile_mode=spec.mode if spec.mode is not None else "default",
                compile_fullgraph=spec.fullgraph,
                fp32_loss=loss_value,
            )
            break
        compile_errors.append({"setting": spec.label, "error": message})

    precision_spec = best_spec
    try:
        ok, message, loss_value = _training_smoke(model_name, precision_spec, "bf16-mixed", device, steps)
    except Exception as error:
        ok = False
        message = _compact_error(error)
        loss_value = None
    if ok:
        result.update(precision="bf16-mixed", bf16_loss=loss_value)
    else:
        result.update(precision="fp32", bf16_error=message)

    if best_spec is None:
        result["compile_errors"] = compile_errors
        result["limitation"] = "no compile candidate passed smoke"
    return result


def _print_markdown(results: Sequence[dict[str, Any]]) -> None:
    print("\n| model | compile | fullgraph | precision | limitation |")
    print("|---|---:|---:|---:|---|")
    for item in results:
        compile_label = "disabled"
        if item["compile_enabled"]:
            compile_label = str(item["compile_mode"])
        print(
            "| {model} | {compile_label} | {fullgraph} | {precision} | {limitation} |".format(
                model=item["model"],
                compile_label=compile_label,
                fullgraph=item["compile_fullgraph"],
                precision=item["precision"],
                limitation=item.get("limitation") or "",
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    _install_timesead_stubs_if_missing()
    device = _select_device(force_cpu=args.cpu)
    model_names = _iter_model_names(args.models)
    results = []
    for model_name in model_names:
        print(f"==> {model_name}", flush=True)
        result = run_model(model_name, device=device, steps=args.steps)
        print(json.dumps(result, sort_keys=True), flush=True)
        results.append(result)
    _print_markdown(results)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
