import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

class ContentLPEncoder(nn.Module):
    """
    Low-pass content encoder that preserves temporal alignment.
    Downsamples strongly (anti-aliased), processes at low res, then upsamples back.

    Args:
        ds:           int, downsample factor (4 or 8 recommended)
        hidden:       int, channel width in the low-res trunk
        blocks:       int, number of low-res conv blocks
        kernel:       int, kernel size for low-res convs (>=3)
        up_mode:      str, upsampling mode ('linear' recommended)
    Input:
        x:  (B,1,L)
    Output:
        y:  (B,1,L)  -- smooth (low-pass) version aligned to x
    """

    def __init__(
        self,
        ds: int = 8,
        hidden: int = 128,
        blocks: int = 3,
        kernel: int = 5,
        up_mode: str = "linear",
    ):
        super().__init__()
        assert ds >= 2 and isinstance(ds, int)
        self.ds = ds
        pad = (kernel - 1) // 2

        # Anti-aliased downsample: blur (conv) + stride
        self.pre = nn.Sequential(
            nn.Conv1d(
                1, hidden, kernel_size=kernel, padding=pad, stride=ds, bias=False
            ),
            nn.GELU(),
        )

        trunk = []
        for _ in range(blocks):
            trunk += [
                nn.Conv1d(hidden, hidden, kernel_size=kernel, padding=pad),
                nn.GELU(),
                nn.Conv1d(hidden, hidden, kernel_size=kernel, padding=pad),
                nn.GELU(),
            ]
        self.trunk = nn.Sequential(*trunk)

        # Project back to 1 channel at low res; then upsample to L
        self.proj = nn.Conv1d(hidden, 1, kernel_size=1)
        self.up_mode = up_mode

    def _pad_to_multiple(self, x: torch.Tensor, m: int) -> Tuple[torch.Tensor, int]:
        """
        Pad tensor length to be divisible by m.
        
        Args:
            x: Input tensor, shape (B, C, L)
            m: Multiple to pad to
            
        Returns:
            Tuple of (padded_tensor, pad_amount) where:
                - padded_tensor: Right-padded tensor
                - pad_amount: Number of elements padded (0 if no padding needed)
        """
        # Right-pad so length is divisible by m; remember pad to crop later.
        B, C, L = x.shape
        rem = L % m
        if rem == 0:
            return x, 0
        pad = m - rem
        return F.pad(x, (0, pad)), pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply content encoding with downsampling and upsampling.
        
        Args:
            x: Input tensor, shape (B, 1, L)
            
        Returns:
            Low-pass filtered (smooth) version of input, shape (B, 1, L)
        """
        # x: (B,1,L)
        x, right_pad = self._pad_to_multiple(x, self.ds)  # keep alignment
        h = self.pre(x)  # (B,H,L//ds)
        h = self.trunk(h)  # (B,H,L//ds)
        h = self.proj(h)  # (B,1,L//ds)

        # Upsample back to original (including pad), then crop pad off
        up_len = x.shape[-1]
        y = F.interpolate(h, size=up_len, mode=self.up_mode, align_corners=False)
        if right_pad:
            y = y[..., : up_len - right_pad]  # (B,1,L)
        return y


class SymDCFreeConv1d(nn.Module):
    """
    k odd, stride=1. Conv-only layer with:
      - zero-DC kernels (sum=0)
      - symmetric kernels (linear phase, no pre/post shift)
      - manual reflect padding to avoid edge artifacts
    """

    def __init__(self, in_ch, out_ch, k=3, groups=1, pad_mode="reflect"):
        """
        Initialize symmetric zero-DC convolution layer.
        
        Args:
            in_ch: Number of input channels
            out_ch: Number of output channels
            k: Kernel size (must be odd, default: 3)
            groups: Number of convolution groups (default: 1)
            pad_mode: Padding mode for reflection/replication (default: "reflect")
        """
        super().__init__()
        assert k % 2 == 1, "use odd kernel for centering"
        self.weight = nn.Parameter(torch.empty(out_ch, in_ch // groups, k))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        self.groups = groups
        self.pad = (k - 1) // 2
        self.pad_mode = pad_mode  # "reflect" or "replicate" are both fine

    def forward(self, x):
        """
        Apply symmetric zero-DC convolution.
        
        Args:
            x: Input tensor, shape (B, C, L)
            
        Returns:
            Convolved tensor with zero DC component and symmetric kernels
        """
        # 1) zero-DC
        w = self.weight - self.weight.mean(dim=2, keepdim=True)
        # 2) symmetric (linear phase)
        w = 0.5 * (w + torch.flip(w, dims=[2]))
        x = F.pad(x, (self.pad, self.pad), mode=self.pad_mode)
        return F.conv1d(x, w, bias=None, stride=1, padding=0, groups=self.groups)


class StylePureConvEncoder(nn.Module):
    """
    Local, conv-only style encoder with boundary-safe padding and linear phase.
    """

    def __init__(self, hidden=16, depth=2, out_channels=1, pad_mode="reflect"):
        """
        Initialize pure convolution style encoder.
        
        Args:
            hidden: Number of hidden channels (default: 16)
            depth: Number of convolutional layers (default: 2)
            out_channels: Number of output channels (default: 1)
            pad_mode: Padding mode for boundary handling (default: "reflect")
        """
        super().__init__()
        layers = [SymDCFreeConv1d(1, hidden, k=3, pad_mode=pad_mode), nn.GELU()]
        for _ in range(depth - 1):
            layers += [
                SymDCFreeConv1d(hidden, hidden, k=3, pad_mode=pad_mode),
                nn.GELU(),
            ]
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv1d(hidden, out_channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.head.bias)  # don’t reintroduce constant offsets

    def forward(self, x):  # (B,1,L)
        """
        Encode style features from input time series.
        
        Args:
            x: Input tensor, shape (B, 1, L)
            
        Returns:
            Style features, shape (B, 1, L)
        """
        h = self.body(x)
        return self.head(h)  # (B,1,L)


# ==================== ALiBi (additive biases) ====================


def _get_alibi_slopes(num_heads: int, device):
    """
    Compute ALiBi (Attention with Linear Biases) slope values for each head.
    
    Args:
        num_heads: Number of attention heads
        device: PyTorch device for tensor placement
        
    Returns:
        Tensor of slope values, shape (num_heads,)
    """
    base = 2 ** (-8.0 / num_heads)
    return torch.pow(base, torch.arange(1, num_heads + 1, device=device))


def alibi_bidirectional(num_heads: int, length: int, device):
    """
    Create bidirectional ALiBi attention bias for self-attention.
    
    Creates additive attention biases based on absolute position distances,
    with different slopes for each head.
    
    Args:
        num_heads: Number of attention heads
        length: Sequence length
        device: PyTorch device for tensor placement
        
    Returns:
        Attention bias tensor, shape (1, num_heads, length, length)
    """
    slopes = _get_alibi_slopes(num_heads, device)
    pos = torch.arange(length, device=device)
    rel = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs().float()  # (L,L)
    bias = -slopes.view(-1, 1, 1) * rel.unsqueeze(0)  # (1,H,L,L)
    return bias.unsqueeze(0)  # (1,H,L,L) with leading batch dim


def alibi_cross(num_heads: int, len_q: int, len_k: int, device):
    """
    Create ALiBi attention bias for cross-attention.
    
    Creates additive attention biases for cross-attention based on position
    distances between query and key sequences.
    
    Args:
        num_heads: Number of attention heads
        len_q: Query sequence length
        len_k: Key sequence length
        device: PyTorch device for tensor placement
        
    Returns:
        Attention bias tensor, shape (1, num_heads, len_q, len_k)
    """
    slopes = _get_alibi_slopes(num_heads, device)
    pos_q = torch.arange(len_q, device=device).unsqueeze(1)
    pos_k = torch.arange(len_k, device=device).unsqueeze(0)
    rel = (pos_q - pos_k).abs().float()  # (Lq,Lk)
    bias = -slopes.view(-1, 1, 1) * rel.unsqueeze(0)  # (1,H,Lq,Lk)
    return bias.unsqueeze(0)  # (1,H,Lq,Lk)


# ==================== Timestep embedding ====================


class SinusoidalEmbedding(nn.Module):
    def __init__(self, d_model):
        """
        Initialize sinusoidal positional embedding.
        
        Args:
            d_model: Embedding dimension
        """
        super().__init__()
        self.d_model = d_model

    def forward(self, t):  # (B,)
        """
        Generate sinusoidal timestep embeddings.
        
        Args:
            t: Timestep indices, shape (B,)
            
        Returns:
            Timestep embeddings, shape (B, d_model)
        """
        device = t.device
        half = self.d_model // 2
        scale = math.log(10000) / max(1, half - 1)
        freqs = torch.exp(torch.arange(half, device=device) * -scale)
        angles = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([angles.sin(), angles.cos()], dim=1)  # (B,D)


# ==================== Attention (SDPA) ====================


class SDPAAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        """
        Initialize scaled dot-product attention module.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads (must divide d_model evenly)
        """
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, attn_bias=None):
        """
        Apply multi-head scaled dot-product attention.
        
        Args:
            query: Query tensor, shape (B, Lq, d_model)
            key: Key tensor, shape (B, Lk, d_model)
            value: Value tensor, shape (B, Lk, d_model)
            attn_bias: Optional attention bias, shape (1, num_heads, Lq, Lk)
            
        Returns:
            Attention output, shape (B, Lq, d_model)
        """
        B, Lq, _ = query.shape
        Lk = key.shape[1]
        q = self.q(query).view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(key).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(value).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        if attn_bias is None:
            attn_mask = None
        else:
            attn_mask = attn_bias
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        return self.o(out)


# ==================== Simple patch embed / unpatch ====================


class PatchEmbed(nn.Module):
    """(B, C, L) -> (B, N, D), zero-pad to multiple of patch_size."""

    def __init__(self, patch_size=16, in_channels=1, d_model=256):
        """
        Initialize patch embedding layer.
        
        Args:
            patch_size: Size of each patch (default: 16)
            in_channels: Number of input channels (default: 1)
            d_model: Model embedding dimension (default: 256)
        """
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.proj = nn.Linear(patch_size * in_channels, d_model)

    def forward(self, x):
        """
        Convert time series to patch embeddings.
        
        Pads input to be divisible by patch_size, splits into patches,
        and projects to embedding dimension.
        
        Args:
            x: Input tensor, shape (B, C, L)
            
        Returns:
            Patch embeddings, shape (B, N, d_model) where N = L // patch_size
        """
        # x: (B, C, L)
        B, C, L = x.shape
        if C != self.in_channels:
            raise ValueError(
                f"PatchEmbed expected {self.in_channels} channels, got {C}"
            )
        if L % self.patch_size != 0:
            pad = self.patch_size - (L % self.patch_size)
            x = F.pad(x, (0, pad))
        N = x.shape[-1] // self.patch_size
        x = x.view(B, C, N, self.patch_size).permute(0, 2, 1, 3).flatten(2)  # (B,N,C*P)
        return self.proj(x)  # (B,N,D)


class Unpatch(nn.Module):
    """(B, N, D) -> (B, C, L) then crop to original_length."""

    def __init__(self, patch_size=16, out_channels=1, d_model=256):
        """
        Initialize unpatch layer.
        
        Args:
            patch_size: Size of each patch (default: 16)
            out_channels: Number of output channels (default: 1)
            d_model: Model embedding dimension (default: 256)
        """
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.proj = nn.Linear(d_model, patch_size * out_channels)

    def forward(self, patches, original_length):
        """
        Convert patch embeddings back to time series.
        
        Args:
            patches: Patch embeddings, shape (B, N, d_model)
            original_length: Target length for output time series
            
        Returns:
            Time series tensor, shape (B, out_channels, original_length)
        """
        B, N, _ = patches.shape
        x = self.proj(patches).view(
            B, N, self.out_channels, self.patch_size
        )  # (B,N,C,P)
        x = x.permute(0, 2, 1, 3).flatten(2)  # (B,C,N*P)
        return x[:, :, :original_length]


# ==================== Denoiser (no AdaLN; style via cross-attn only) ====================


class DenoisingBlock(nn.Module):
    """
    One block:
      - self-attn (ALiBi)
      - cross-attn to content (ALiBi cross)
      - cross-attn to style   (ALiBi cross)
      - MLP
    """

    def __init__(self, d_model, num_heads):
        """
        Initialize denoising transformer block.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
        """
        super().__init__()
        self.num_heads = num_heads

        self.norm_self = nn.LayerNorm(d_model)
        self.norm_c = nn.LayerNorm(d_model)
        self.norm_s = nn.LayerNorm(d_model)
        self.norm_mlp = nn.LayerNorm(d_model)

        self.self_attn = SDPAAttention(d_model, num_heads)
        self.cross_attn_c = SDPAAttention(d_model, num_heads)
        self.cross_attn_s = SDPAAttention(d_model, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.SiLU(), nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x, content_mem, style_mem, self_bias, cross_bias_c, cross_bias_s):
        """
        Apply denoising block with self and cross-attention.
        
        Args:
            x: Input tokens, shape (B, N, d_model)
            content_mem: Content memory for cross-attention, shape (B, N, d_model)
            style_mem: Style memory for cross-attention, shape (B, N, d_model)
            self_bias: Attention bias for self-attention
            cross_bias_c: Attention bias for content cross-attention
            cross_bias_s: Attention bias for style cross-attention
            
        Returns:
            Output tokens, shape (B, N, d_model)
        """
        xs = self.norm_self(x)
        x = x + self.self_attn(xs, xs, xs, attn_bias=self_bias)

        xc = self.norm_c(x)
        x = x + self.cross_attn_c(xc, content_mem, content_mem, attn_bias=cross_bias_c)

        xs2 = self.norm_s(x)
        x = x + self.cross_attn_s(xs2, style_mem, style_mem, attn_bias=cross_bias_s)

        xm = self.norm_mlp(x)
        x = x + self.mlp(xm)
        return x


# ==================== Full model (patch-based, ALiBi, simple MLP patch embeds) ====================


class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        d_model=256,
        num_layers=4,
        num_heads=4,
        patch_size=8,
        p_drop_content=0.10,
        p_drop_style=0.15,
        inference_timesteps: int = 200,
        ddim_eta: float = 0.0,
    ):
        """
        Initialize the diffusion transformer model for time series style transfer.
        
        This model uses:
        - Separate encoders for content (low-pass) and style (high-frequency)
        - Patch-based transformer architecture with ALiBi position biases
        - Classifier-free guidance via random dropout of content/style
        - Cross-attention to content and style representations
        
        Args:
            d_model: Model embedding dimension (default: 256)
            num_layers: Number of denoising transformer blocks (default: 4)
            num_heads: Number of attention heads (default: 4)
            patch_size: Size of time series patches (default: 8)
            p_drop_content: Probability of dropping content during training (default: 0.10)
            p_drop_style: Probability of dropping style during training (default: 0.15)
            inference_timesteps: Number of DDIM steps used at inference time (default: 200)
            ddim_eta: DDIM stochasticity parameter. Use 0.0 for deterministic sampling (default: 0.0)
        """
        super().__init__()
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.p_drop_content = p_drop_content
        self.p_drop_style = p_drop_style
        self.inference_timesteps = inference_timesteps
        self.ddim_eta = ddim_eta
        self.content_enc = ContentLPEncoder(ds=8, hidden=128, blocks=3, kernel=5)
        self.style_enc = StylePureConvEncoder(hidden=16, depth=2, out_channels=1)
        # Patch projections (linear on flattened patches)
        self.input_patch = PatchEmbed(patch_size, in_channels=1, d_model=d_model)  # x_t
        self.content_patch = PatchEmbed(
            patch_size, in_channels=1, d_model=d_model
        )  # x_c
        self.style_patch = PatchEmbed(
            patch_size, in_channels=1, d_model=d_model
        )  # cat 3 bands
        self.unpatch = Unpatch(patch_size, out_channels=1, d_model=d_model)

        # Time embedding (added to tokens)
        self.time_emb = nn.Sequential(
            SinusoidalEmbedding(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Denoiser stack
        self.blocks = nn.ModuleList(
            [DenoisingBlock(d_model, num_heads) for _ in range(num_layers)]
        )
        self._noise_schedule_cache: Dict[Tuple[str, int, float, float], Tuple[torch.Tensor, torch.Tensor]] = {}

    @staticmethod
    def _maybe_drop(
        x: torch.Tensor, drop: bool, p: float, training: bool
    ) -> tuple[torch.Tensor, bool]:
        """
        Returns (x_or_zero, actually_dropped)
        drop: if True/False forces behavior; if None, sample Bernoulli with prob p (only when training).
        """
        if drop is True:
            return torch.zeros_like(x), True
        if drop is False:
            return x, False
        # drop is None -> use random only during training
        if training and torch.rand(()) < p:
            return torch.zeros_like(x), True
        return x, False

    def forward(
        self,
        x_t: torch.Tensor,  # (B,1,L)
        t: torch.Tensor,  # (B,)
        x_c: torch.Tensor,  # (B,1,L)
        x_s: torch.Tensor,  # (B,1,L)
        *,
        drop_content: (
            bool 
        ) = None,  # NEW: None => sample (train), True/False => force
        drop_style: bool  = None,  # NEW
    ):
        """
        Forward pass for noise prediction in diffusion model.
        
        Implements classifier-free guidance by optionally dropping content/style inputs.
        During training, drops content/style with probability p_drop_content/p_drop_style.
        
        Args:
            x_t: Noised time series at timestep t, shape (B, 1, L)
            t: Diffusion timestep indices, shape (B,)
            x_c: Content reference time series, shape (B, 1, L)
            x_s: Style reference time series, shape (B, 1, L)
            drop_content: If True/False, forces content drop behavior.
                         If None, samples based on p_drop_content during training (default: None)
            drop_style: If True/False, forces style drop behavior.
                       If None, samples based on p_drop_style during training (default: None)
            
        Returns:
            Predicted noise, shape (B, 1, L)
        """
        B, _, L = x_t.shape
        device = x_t.device

        # ---------- Classifier-free dropout on raw inputs ----------
        x_c_in, c_dropped = self._maybe_drop(
            x_c, drop_content, self.p_drop_content, self.training
        )
        x_s_in, s_dropped = self._maybe_drop(
            x_s, drop_style, self.p_drop_style, self.training
        )

        # ----- Encode content and style -----
        c_feat = self.content_enc(x_c_in)  # (B,1,L)
        s_feat = self.style_enc(x_s_in)  # (B,1,L)

        # Safety: if dropped, hard-zero after encoding too (avoids encoder biases)
        if c_dropped:
            c_feat = torch.zeros_like(c_feat)
        if s_dropped:
            s_feat = torch.zeros_like(s_feat)

        # ----- Patchify -----
        x_tok = self.input_patch(x_t)  # (B,N,D)
        c_tok = self.content_patch(c_feat)  # (B,N,D)
        s_tok = self.style_patch(s_feat)  # (B,N,D)
        N = x_tok.shape[1]

        # ----- Time conditioning -----
        t_vec = self.time_emb(t).unsqueeze(1)  # (B,1,D)
        x_tok = x_tok + t_vec

        # ----- Biases -----
        self_bias = alibi_bidirectional(self.num_heads, N, device)
        cross_bias_c = alibi_cross(self.num_heads, N, c_tok.shape[1], device)
        cross_bias_s = alibi_cross(self.num_heads, N, s_tok.shape[1], device)

        # ----- Denoising stack -----
        h = x_tok
        for blk in self.blocks:
            h = blk(h, c_tok, s_tok, self_bias, cross_bias_c, cross_bias_s)

        out = self.unpatch(h, original_length=L)  # (B,1,L)
        return out

    def prepare_noise_schedule(self, T=500, beta_start=1e-4, beta_end=0.02):
        """
        Prepare diffusion noise schedule (same as module-level function).
        
        Args:
            T: Number of diffusion timesteps (default: 500)
            beta_start: Starting beta value (default: 1e-4)
            beta_end: Ending beta value (default: 0.02)
            
        Returns:
            Tuple of (betas, alpha_cumprod)
        """
        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        return betas, alpha_cumprod

    def get_noise_schedule(
        self,
        device: torch.device,
        T: int = 500,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device_obj = torch.device(device)
        device_key = str(device_obj)
        cache_key = (device_key, T, beta_start, beta_end)
        cached_schedule = self._noise_schedule_cache.get(cache_key)
        if cached_schedule is None:
            betas, alpha_cumprod = self.prepare_noise_schedule(
                T=T,
                beta_start=beta_start,
                beta_end=beta_end,
            )
            cached_schedule = (
                betas.to(device_obj),
                alpha_cumprod.to(device_obj),
            )
            self._noise_schedule_cache[cache_key] = cached_schedule
        return cached_schedule

  
    @torch.no_grad()
    def retrieve_cache(self, x_c, x_s):
        """
        Pre-compute and cache encoder outputs and attention biases for sampling.
        
        This caches values that don't change across denoising timesteps:
        - Content and style encodings
        - Patch embeddings for content and style
        - ALiBi attention biases
        - Time embeddings for all timesteps
        
        Args:
            x_c: Content reference, shape (B, 1, L)
            x_s: Style reference, shape (B, 1, L)
            
        Returns:
            Tuple of cached values: (c_tok, s_tok, self_bias, cross_bias_c, cross_bias_s, t_vec)
        """
        # ----- Encode content and style -----
        device = x_c.device

        # ----- Encode content and style -----
        c_feat = self.content_enc(x_c)  # (B,1,L)
        s_feat = self.style_enc(x_s)  # (B,1,L)
        
        # ----- Patchify -----
        c_tok = self.content_patch(c_feat)  # (B,N,D)
        s_tok = self.style_patch(s_feat)  # (B,N,D)
        N = c_tok.shape[1]

        # ----- Biases -----
        self_bias = alibi_bidirectional(self.num_heads, N, device)
        cross_bias_c = alibi_cross(self.num_heads, N, c_tok.shape[1], device)
        cross_bias_s = alibi_cross(self.num_heads, N, s_tok.shape[1], device)

        t = torch.arange(0, 500, dtype=torch.long).to(device)
        t_vec = self.time_emb(t)

        return (c_tok, s_tok, self_bias, cross_bias_c, cross_bias_s, t_vec)
    
    @torch.no_grad()
    def inference_forward(self, x_t, t_vec, cached_values):
        """
        Fast forward pass using pre-cached values for sampling.
        
        Args:
            x_t: Noised time series, shape (B, 1, L)
            t_vec: Time embedding for current timestep, shape (1, d_model)
            cached_values: Tuple of pre-cached encoder outputs and biases
            
        Returns:
            Predicted noise, shape (B, 1, L)
        """
        # ----- Patchify -----
        B, _, L = x_t.shape
        x_tok = self.input_patch(x_t)  # (B,N,D)
        
        c_tok, s_tok, self_bias, cross_bias_c, cross_bias_s, _ = cached_values
        
        # ----- Time conditioning -----
        x_tok = x_tok + t_vec

        # ----- Denoising stack -----
        h = x_tok
        for blk in self.blocks:
            h = blk(h, c_tok, s_tok, self_bias, cross_bias_c, cross_bias_s)

        out = self.unpatch(h, original_length=L)  # (B,1,L)
        return out

    @torch.no_grad()
    def guided_eps(
        self,
        x_t: torch.Tensor,
        t: int,
        cached_values,
        s_content: float = 0.9,
        s_style: float = 0.9,
    ) -> torch.Tensor:
        """
        Predict guided noise for a single diffusion timestep.

        Combines three predictions:
        - Unconditional (no content, no style)
        - Content-conditional only
        - Style-conditional only
        Using the formula: eps = eps_u + s_c*(eps_c - eps_u) + s_s*(eps_s - eps_u)

        Args:
            x_t: Current noised sample, shape (B, 1, L)
            t: Current timestep index.
            cached_values: Pre-cached encoder outputs (contains 3x batch: unconditional, content, style)
            s_content: Guidance scale for content (default: 0.9)
            s_style: Guidance scale for style (default: 0.9)

        Returns:
            Guided noise prediction with shape (B, 1, L).
        """
        B = x_t.shape[0]
        device = x_t.device

        x_t3 = x_t.repeat(3, 1, 1)

        t_vec = cached_values[-1][int(t)]
        tt = t_vec.unsqueeze(0).to(device)

        eps3 = self.inference_forward(
            x_t3, tt, cached_values
            )
        eps_u, eps_c, eps_s = eps3[:B], eps3[B : 2 * B], eps3[2 * B :]

        return eps_u + s_content * (eps_c - eps_u) + s_style * (eps_s - eps_u)

    @staticmethod
    def _sample_schedule(total_timesteps: int, inference_timesteps: int, device: torch.device) -> torch.Tensor:
        if inference_timesteps <= 0:
            raise ValueError("inference_timesteps must be positive")
        if inference_timesteps >= total_timesteps:
            return torch.arange(total_timesteps, device=device, dtype=torch.long)

        steps = torch.linspace(0, total_timesteps - 1, steps=inference_timesteps, device=device)
        timesteps = steps.round().to(torch.long).unique(sorted=True)
        if timesteps[0].item() != 0:
            timesteps = torch.cat([timesteps.new_tensor([0]), timesteps])
        if timesteps[-1].item() != total_timesteps - 1:
            timesteps = torch.cat([timesteps, timesteps.new_tensor([total_timesteps - 1])])
        return timesteps.unique(sorted=True)

    @torch.no_grad()
    def ddim_step(
        self,
        x_t: torch.Tensor,
        t: int,
        prev_t: Optional[int],
        alpha_cumprod: torch.Tensor,
        cached_values,
        s_content: float = 0.9,
        s_style: float = 0.9,
    ) -> torch.Tensor:
        """
        Single DDIM update with classifier-free guidance.

        Args:
            x_t: Current noised sample, shape (B, 1, L)
            t: Current timestep index.
            prev_t: Previous timestep index on the sparse DDIM schedule, or None for the final step.
            alpha_cumprod: Cumulative alpha product schedule.
            cached_values: Pre-cached encoder outputs for guided inference.
            s_content: Guidance scale for content (default: 0.9)
            s_style: Guidance scale for style (default: 0.9)

        Returns:
            Sample at the previous DDIM timestep.
        """
        eps = self.guided_eps(x_t, t, cached_values, s_content=s_content, s_style=s_style)

        alpha_t = alpha_cumprod[int(t)]
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(torch.clamp(1 - alpha_t, min=0.0))
        x0_pred = (x_t - sqrt_one_minus_alpha_t * eps) / sqrt_alpha_t

        if prev_t is None:
            return x0_pred

        alpha_prev = alpha_cumprod[int(prev_t)]
        sigma_t = self.ddim_eta * torch.sqrt(
            ((1 - alpha_prev) / (1 - alpha_t)) * (1 - alpha_t / alpha_prev)
        )
        direction = torch.sqrt(torch.clamp(1 - alpha_prev - sigma_t ** 2, min=0.0)) * eps
        noise = sigma_t * torch.randn_like(x_t) if self.ddim_eta > 0 else torch.zeros_like(x_t)
        return torch.sqrt(alpha_prev) * x0_pred + direction + noise

    @torch.no_grad()
    def sample(self,
        x_c: torch.Tensor,  # (B,1,L)
        x_s: torch.Tensor,  # (B,1,L)
        *,
        drop_content: (
            bool 
        ) = None,  # NEW: None => sample (train), True/False => force
        drop_style: bool  = None,
    ):
        """
        Generate time series samples via DDIM reverse diffusion.

        The process:
        1. Start with random Gaussian noise x_T ~ N(0, I)
        2. Cache encoder outputs for content and style (including unconditional)
        3. Follow a sparse DDIM schedule using classifier-free guidance
        4. Return clean generated sample x_0

        Args:
            x_c: Content reference time series, shape (B, 1, L)
            x_s: Style reference time series, shape (B, 1, L)
            drop_content: Optional override for content dropout (unused in sampling)
            drop_style: Optional override for style dropout (unused in sampling)

        Returns:
            Generated time series, shape (B, 1, L)

        Raises:
            ValueError: If sequence length L is not divisible by patch_size
        """
        B, _, L = x_c.shape
        device = x_c.device
        T = 500

        _, alpha_cumprod = self.get_noise_schedule(device, T=T)

        patch_size = self.patch_size
        if L % patch_size != 0:
            raise ValueError(
                f"Sequence length {L} must be a multiple of patch size ({patch_size})"
            )

        batch_size = x_c.shape[0]
        x_t = torch.randn((batch_size, 1, L), device=device)

        # stack inputs: [uncond | content-only | style-only]
        x_c3 = torch.cat([torch.zeros_like(x_c), x_c, torch.zeros_like(x_c)], dim=0)
        x_s3 = torch.cat([torch.zeros_like(x_s), torch.zeros_like(x_s), x_s], dim=0)

        ## cache encoder outputs, keys and values
        cached_values = self.retrieve_cache(x_c3, x_s3)

        timesteps = self._sample_schedule(T, self.inference_timesteps, device)
        for index in range(timesteps.numel() - 1, -1, -1):
            current_t = int(timesteps[index].item())
            prev_t = int(timesteps[index - 1].item()) if index > 0 else None
            x_t = self.ddim_step(x_t, current_t, prev_t, alpha_cumprod, cached_values)

        return x_t

# ==================== Smoke test ====================

if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DiffusionTransformer(patch_size=8).to(device)

    B, C, L = 128, 1, 128
    x_t = torch.randn(B, C, L, device=device)
    t = torch.randint(0, 500, (B,), device=device)
    x_c = x_t.clone()
    x_s = x_c.clone()
    import time
    start = time.perf_counter()
    out = model.sample(x_c, x_s)
    end = time.perf_counter()
    print(f"Runtime: {end - start:.6f} seconds")
