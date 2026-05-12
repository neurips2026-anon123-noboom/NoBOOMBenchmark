from contextlib import nullcontext

import torch
import numpy as np

from .inference_utils import normalize, denormalize, convert_to_tensor


def _resolve_sample_fn(model_or_sampler):
    if hasattr(model_or_sampler, "sample"):
        return model_or_sampler.sample
    if callable(model_or_sampler):
        return model_or_sampler
    raise TypeError("Expected a sampler callable or an object exposing a 'sample' method.")

def style_transfer (model, contents, styles, device, denorm, sequence_length):
    """
    Perform time-series style-transfer using DiffusionTransformer model
    
    Args:
        model: DiffusionTransformer
        contents: torch.tensor/np.array
        styles: torch.tensor/np.array
        device: torch.device cpu/cuda
        sequence_length: int T
    Returns:
        the generation of tsst as torch.tensor of shape (B, T) 
    """
    content_tensor = convert_to_tensor(contents, device)
    style_tensor = convert_to_tensor(styles, device)
    
    content_tensor_normalized, content_stats = normalize(content_tensor)
    style_tensor_normalized, style_stats = normalize(style_tensor)
    sample_fn = _resolve_sample_fn(model)
    autocast_context = nullcontext()
    if device.type == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    with autocast_context:
        generation = sample_fn(content_tensor_normalized, style_tensor_normalized)
    generation = generation.to(torch.float32)
    if denorm:
        generation = denormalize(generation, content_stats)
    generation = generation.squeeze(1)
    
    return generation


def one_to_one(
    model,
    contents_batch: torch.Tensor,
    styles_batch: torch.Tensor,
    device: torch.device = torch.device("cpu"),
    save_file: str = None,
    labels: torch.Tensor = None
):
    """
    Pair c_i with s_i (row-wise). Generates batched style-transfer outputs.

    - contents_batch: tensor containing content references (B, L) or (B, L, F)
    - styles_batch:   tensor containing style references (same number of contents)
    - device:         torch device for compute
    - save_file:      if provided, save each batch to disk with this prefix instead of returning.
    - labels:         optional tensor of labels (B, L) or (B, L, F) to save alongside data.
    
    Returns:
        If save_file is None:
            Tensor of shape (B, L, F).
        If save_file is provided:
            None.
    """
    # Convert to tensor manually to handle (B, L, F) correctly
    if not isinstance(contents_batch, torch.Tensor):
        contents_batch = torch.tensor(contents_batch, device=device)
    if not isinstance(styles_batch, torch.Tensor):
        styles_batch = torch.tensor(styles_batch, device=device)
    device = contents_batch.device

    contents_batch = contents_batch.float()
    styles_batch = styles_batch.float()
    
    # Ensure shapes are (B, L, F)
    if contents_batch.dim() == 2:
        contents_batch = contents_batch.unsqueeze(-1) # (B, L, 1)
    if styles_batch.dim() == 2:
        styles_batch = styles_batch.unsqueeze(-1) # (B, L, 1)
    
    B, L, F = contents_batch.shape
    # Handle labels
    if labels is not None:
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)
        # Keep labels on CPU or move to device? CPU is fine for saving.
        # But for slicing we might want it to be tensor.
        # Ensure shape matches B
        if labels.shape[0] != B:
             raise ValueError(f"Labels batch size {labels.shape[0]} does not match contents batch size {B}")

    flat_contents = contents_batch.permute(0, 2, 1).reshape(-1, L).contiguous()
    flat_styles = styles_batch.permute(0, 2, 1).reshape(-1, L).contiguous()
    flat_generation = style_transfer(
        model,
        flat_contents,
        flat_styles,
        device,
        True,
        L,
    )

    all_generations = flat_generation.reshape(B, F, L).permute(0, 2, 1).contiguous()

    if save_file is not None:
        batch_data = all_generations.cpu().numpy()
        filename = f"{save_file}_part_0.npy"
        np.save(filename, batch_data)

        if labels is not None:
            batch_labels = labels
            if isinstance(batch_labels, torch.Tensor):
                batch_labels = batch_labels.cpu().numpy()
            label_filename = f"{save_file}_part_0_labels.npy"
            np.save(label_filename, batch_labels)
    if save_file is not None:
        return None
    return all_generations
