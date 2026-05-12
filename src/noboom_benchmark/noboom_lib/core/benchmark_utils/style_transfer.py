from contextlib import nullcontext
import logging
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..diffstylets.inference_utils import convert_to_tensor, normalize
from ..diffstylets.model import DiffusionTransformer
from ..diffstylets.tsst import one_to_one

logger = logging.getLogger(__name__)


class StyleTransfer(nn.Module):
    def __init__(
        self,
        ckpt_path: str,
        batch_dim: int,
        patch_size: int = 8,
        compile_model: bool = True,
        compile_mode: Optional[str] = None,
        compile_fullgraph: bool = False,
    ):
        super(StyleTransfer, self).__init__()
        self.ckpt_path = ckpt_path
        self.batch_dim = batch_dim
        self.patch_size = patch_size
        self.model = DiffusionTransformer(patch_size=patch_size)
        self.compile_model = compile_model
        self.compile_mode = compile_mode
        self.compile_fullgraph = compile_fullgraph
        self._weights_loaded = False
        self._compiled_retrieve_cache = None
        self._compiled_ddim_step = None
        self._compiled_helpers_device: Optional[str] = None
        self._compile_failed_device: Optional[str] = None

    @staticmethod
    def _generation_autocast_context(device: torch.device) -> Any:
        if device.type == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _current_model_device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _should_compile_helpers(self, device: torch.device) -> bool:
        return (
            self.compile_model
            and device.type == "cuda"
            and hasattr(torch, "compile")
        )

    def _compiled_helpers(self):
        device = self._current_model_device()
        device_key = str(device)
        if not self._should_compile_helpers(device):
            return self.model.retrieve_cache, self.model.ddim_step
        if (
            self._compiled_retrieve_cache is not None
            and self._compiled_ddim_step is not None
            and self._compiled_helpers_device == device_key
        ):
            return self._compiled_retrieve_cache, self._compiled_ddim_step
        if self._compile_failed_device == device_key:
            return self.model.retrieve_cache, self.model.ddim_step

        compile_kwargs: Dict[str, Any] = {
            "fullgraph": self.compile_fullgraph,
        }
        if self.compile_mode is not None:
            compile_kwargs["mode"] = self.compile_mode

        try:
            self._compiled_retrieve_cache = torch.compile(self.model.retrieve_cache, **compile_kwargs)
            self._compiled_ddim_step = torch.compile(self.model.ddim_step, **compile_kwargs)
            self._compiled_helpers_device = device_key
            logger.info(
                "Enabled torch.compile for DiffStyle retrieve_cache and ddim_step on %s (mode=%s, fullgraph=%s).",
                device_key,
                self.compile_mode,
                self.compile_fullgraph,
            )
            return self._compiled_retrieve_cache, self._compiled_ddim_step
        except Exception:
            self._compile_failed_device = device_key
            logger.warning(
                "Failed to compile DiffStyle retrieve_cache/ddim_step on %s; falling back to eager execution.",
                device_key,
                exc_info=True,
            )
            return self.model.retrieve_cache, self.model.ddim_step

    @staticmethod
    def _resolve_ckpt_path(ckpt_path: str) -> Path:
        candidate = Path(ckpt_path).expanduser()
        if candidate.exists():
            return candidate.resolve()

        package_root = Path(__file__).resolve().parents[3]
        resolved_candidate = package_root / candidate
        if resolved_candidate.exists():
            return resolved_candidate.resolve()

        raise FileNotFoundError(
            "Style transfer checkpoint was not found. "
            f"Tried '{candidate}' and '{resolved_candidate}'."
        )

    def resolved_ckpt_path(self) -> Path:
        return self._resolve_ckpt_path(self.ckpt_path)

    @staticmethod
    def _file_signature(path: Path) -> Dict[str, Any]:
        resolved_path = path.resolve()
        stat = resolved_path.stat()
        return {
            "path": str(resolved_path),
            "mtime_ns": int(stat.st_mtime_ns),
            "size": int(stat.st_size),
        }

    def cache_signature(self) -> Dict[str, Any]:
        resolved_ckpt_path = self.resolved_ckpt_path()
        model_module_path = Path(sys.modules[DiffusionTransformer.__module__].__file__).resolve()
        inference_module_path = Path(sys.modules[normalize.__module__].__file__).resolve()
        return {
            "checkpoint": self._file_signature(resolved_ckpt_path),
            "patch_size": int(self.patch_size),
            "inference_timesteps": int(self.model.inference_timesteps),
            "ddim_eta": float(self.model.ddim_eta),
            "implementation": {
                "style_transfer": self._file_signature(Path(__file__)),
                "diffusion_model": self._file_signature(model_module_path),
                "inference_utils": self._file_signature(inference_module_path),
            },
        }

    def _prepare_batch(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        if self.batch_dim != 0:
            x = x.transpose(0, self.batch_dim)
        seq_len = x.size(1)
        pad = (self.patch_size - (seq_len % self.patch_size)) % self.patch_size
        x = F.pad(x, (0, 0, 0, pad), mode='replicate')
        return x, seq_len

    @staticmethod
    def _flatten_feature_batch(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        batch_size, seq_len, num_features = x.shape
        return x.permute(0, 2, 1).reshape(batch_size * num_features, seq_len).contiguous()

    def setup(self):
        if self._weights_loaded:
            return
        resolved_ckpt_path = self.resolved_ckpt_path()
        ckpt = torch.load(resolved_ckpt_path, map_location='cpu')
        self.model.load_state_dict(ckpt['model_state_dict'])
        self._weights_loaded = True

    def probe_generation_memory(self, x: torch.Tensor) -> Optional[Dict[str, float]]:
        device = x.device
        if device.type != "cuda" or not torch.cuda.is_available():
            return None

        prepared_x, _ = self._prepare_batch(x)
        flat_inputs = self._flatten_feature_batch(prepared_x)
        content_tensor = convert_to_tensor(flat_inputs, device)
        style_tensor = convert_to_tensor(flat_inputs, device)
        content_tensor_normalized, _ = normalize(content_tensor)
        style_tensor_normalized, _ = normalize(style_tensor)

        total_timesteps = 500
        _, alpha_cumprod = self.model.get_noise_schedule(device, T=total_timesteps)
        x_c3 = torch.cat(
            [
                torch.zeros_like(content_tensor_normalized),
                content_tensor_normalized,
                torch.zeros_like(content_tensor_normalized),
            ],
            dim=0,
        )
        x_s3 = torch.cat(
            [
                torch.zeros_like(style_tensor_normalized),
                torch.zeros_like(style_tensor_normalized),
                style_tensor_normalized,
            ],
            dim=0,
        )
        timesteps = self.model._sample_schedule(total_timesteps, self.model.inference_timesteps, device)
        current_t = int(timesteps[-1].item())
        prev_t = int(timesteps[-2].item()) if timesteps.numel() > 1 else None

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        with torch.inference_mode(), self._generation_autocast_context(device):
            cached_values = self.model.retrieve_cache(x_c3, x_s3)
            x_t = torch.randn(
                (content_tensor_normalized.shape[0], 1, content_tensor_normalized.shape[-1]),
                device=device,
                dtype=content_tensor_normalized.dtype,
            )
            _ = self.model.ddim_step(x_t, current_t, prev_t, alpha_cumprod, cached_values)
        torch.cuda.synchronize(device)
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        return {
            "peak_allocated_bytes": float(peak_allocated),
            "peak_reserved_bytes": float(peak_reserved),
            "effective_series_batch": float(content_tensor_normalized.shape[0]),
            "sequence_length": float(content_tensor_normalized.shape[-1]),
        }

    @torch.no_grad()
    def sample(
        self,
        x_c: torch.Tensor,
        x_s: torch.Tensor,
        *,
        drop_content: Optional[bool] = None,
        drop_style: Optional[bool] = None,
    ) -> torch.Tensor:
        del drop_content, drop_style

        _, _, sequence_length = x_c.shape
        device = x_c.device
        total_timesteps = 500
        patch_size = self.model.patch_size
        if sequence_length % patch_size != 0:
            raise ValueError(
                f"Sequence length {sequence_length} must be a multiple of patch size ({patch_size})"
            )

        _, alpha_cumprod = self.model.get_noise_schedule(device, T=total_timesteps)
        retrieve_cache_fn, ddim_step_fn = self._compiled_helpers()

        x_t = torch.randn((x_c.shape[0], 1, sequence_length), device=device)
        x_c3 = torch.cat([torch.zeros_like(x_c), x_c, torch.zeros_like(x_c)], dim=0)
        x_s3 = torch.cat([torch.zeros_like(x_s), torch.zeros_like(x_s), x_s], dim=0)
        cached_values = retrieve_cache_fn(x_c3, x_s3)

        timesteps = self.model._sample_schedule(total_timesteps, self.model.inference_timesteps, device)
        for index in range(timesteps.numel() - 1, -1, -1):
            current_t = int(timesteps[index].item())
            prev_t = int(timesteps[index - 1].item()) if index > 0 else None
            x_t = ddim_step_fn(x_t, current_t, prev_t, alpha_cumprod, cached_values)

        return x_t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, seq_len = self._prepare_batch(x)
        x = one_to_one(self, x, x)
        x = x.narrow(1, 0, seq_len)
        if self.batch_dim != 0:
            x = x.transpose(0, self.batch_dim)
        x = x.contiguous()
        return x
