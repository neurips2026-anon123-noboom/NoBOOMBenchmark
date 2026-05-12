from typing import Sequence

import torch


def insert_before_subsequences(
    x: torch.Tensor,
    lengths,
    dim: int = 0,
    insert_value=0,
):
    """Insert one slice before each subsequence along a dimension.

    Args:
        x (torch.Tensor): Tensor containing concatenated subsequences.
        lengths (Sequence[int] | torch.Tensor): Lengths of each subsequence.
        dim (int): Dimension along which to insert. Defaults to 0.
        insert_value (Any): Scalar or tensor to insert. Defaults to 0.

    Returns:
        torch.Tensor: Tensor with inserted slices.

    Raises:
        ValueError: If lengths do not sum to ``x.size(dim)``.
    """
    device = x.device
    lengths_t = torch.as_tensor(lengths, device=device, dtype=torch.long)
    if lengths_t.numel() == 0:
        return x

    L = int(lengths_t.sum().item())
    if L != x.size(dim):
        raise ValueError(f"sum(lengths)={L} must equal x.size(dim)={x.size(dim)}")

    nseg = int(lengths_t.numel())
    new_L = L + nseg

    # Segment id per original position (0..nseg-1)
    seg_id = torch.repeat_interleave(torch.arange(nseg, device=device), lengths_t)  # (L,)
    # Destination positions for original elements after insertion
    dst_pos = torch.arange(L, device=device) + (seg_id + 1)  # (L,)

    # Positions where inserted values go: start_out of each segment
    start_orig = torch.cumsum(lengths_t, 0) - lengths_t                      # (nseg,)
    ins_pos = start_orig + torch.arange(nseg, device=device)                  # (nseg,)

    # Allocate output
    out_shape = list(x.shape)
    out_shape[dim] = new_L
    out = x.new_empty(out_shape)

    # Build the inserted slices tensor of shape (nseg, *slice_shape)
    slice_shape = x.shape[:dim] + x.shape[dim+1:]

    if torch.is_tensor(insert_value):
        v = insert_value.to(device=device, dtype=x.dtype)
        # Make v broadcastable to slice_shape, then expand to (nseg, *slice_shape)
        v = v.reshape((1,) + tuple(v.shape))
        ins_src = v.expand((nseg,) + slice_shape)
    else:
        ins_src = x.new_full((nseg,) + slice_shape, insert_value)

    # Fill inserted positions
    out.index_copy_(dim, ins_pos, ins_src)

    # Fill the shifted original data
    out.index_copy_(dim, dst_pos, x)

    return out


def pad_each_subsequence(
    x: torch.Tensor,
    lengths: Sequence[int] | torch.Tensor,
    *,
    pad_prefix: int = 0,
    pad_suffix: int = 0,
    value: float | int = 0.0,
    dim: int = 0,
) -> torch.Tensor:
    """Pad the start and/or end of each subsequence within a concatenated tensor.

    The input tensor `x` contains multiple subsequences concatenated along `dim`,
    with lengths provided by `lengths`. This function inserts `pad_prefix` values
    before each subsequence and `pad_suffix` values after each subsequence,
    then concatenates everything back along `dim`.

    Args:
        x: Tensor with subsequences concatenated along `dim`.
        lengths: Lengths of each subsequence.
        pad_prefix: Number of values to insert before each subsequence.
        pad_suffix: Number of values to insert after each subsequence.
        value: Fill value for padding.
        dim: Dimension along which subsequences are concatenated.

    Returns:
        Tensor with padded subsequences concatenated along `dim`.

    Raises:
        ValueError: If padding is negative or lengths mismatch the tensor size.
    """
    if pad_prefix < 0 or pad_suffix < 0:
        raise ValueError("pad_prefix and pad_suffix must be >= 0")
    if pad_prefix == 0 and pad_suffix == 0:
        return x

    if isinstance(lengths, torch.Tensor):
        lengths_t = lengths.to(device=x.device, dtype=torch.long)
    else:
        lengths_t = torch.tensor(list(lengths), device=x.device, dtype=torch.long)

    if lengths_t.numel() == 0:
        return x

    T = x.size(dim)
    total = int(lengths_t.sum().item())
    if total != T:
        raise ValueError(f"sum(lengths)={total} must equal x.size(dim)={T}")

    n = int(lengths_t.numel())

    # Move `dim` to front so we can operate on axis 0.
    xm = torch.movedim(x, dim, 0)  # [T, ...]
    device = x.device

    # boundaries = end positions of each subsequence in the input concatenation
    # e.g., lengths [3,2,1] -> boundaries [3,5,6]
    boundaries = torch.cumsum(lengths_t, dim=0)  # [n]

    pos = torch.arange(T, device=device)  # [0..T-1]

    # seq_id in [0..n-1] for each element in the concatenation
    # Use boundaries[:-1] as cut points.
    seq_id = torch.bucketize(pos, boundaries[:-1], right=False)

    # Prefix padding shifts subsequence k by pad_prefix*(k+1)
    # Suffix padding shifts subsequence k by pad_suffix*k (since earlier subsequences add suffix blocks)
    out_pos = pos + pad_prefix * (seq_id + 1) + pad_suffix * seq_id  # [T]

    out_T = T + n * (pad_prefix + pad_suffix)
    out_shape = (out_T, *xm.shape[1:])
    out = xm.new_full(out_shape, value)

    # Copy original values into shifted positions
    out.index_copy_(0, out_pos, xm)

    return torch.movedim(out, 0, dim)


