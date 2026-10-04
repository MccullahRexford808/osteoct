"""Three-dimensional vision blocks used by the perception-architecture arms."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from osteoct.utils.numeric import window_partition_3d, window_reverse_3d


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm = nn.GroupNorm(min(8, out_channels), out_channels)
        self.activation = nn.GELU()

    def forward(self, values: Tensor) -> Tensor:
        convolved: Tensor = self.conv(values)
        normalised: Tensor = self.norm(convolved)
        activated: Tensor = self.activation(normalised)
        return activated


class ConvEncoder3D(nn.Module):
    def __init__(self, in_channels: int = 1, embed_dim: int = 32, depth: int = 3) -> None:
        super().__init__()
        channels = [in_channels]
        for level in range(depth):
            channels.append(embed_dim * (2**level))
        blocks: list[nn.Module] = []
        for level in range(depth):
            blocks.append(ConvBlock3D(channels[level], channels[level + 1], stride=2))
        self.blocks = nn.ModuleList(blocks)
        self.projection = nn.Linear(channels[-1], embed_dim)
        self.embed_dim = embed_dim

    def forward(self, volume: Tensor) -> Tensor:
        values = volume
        for block in self.blocks:
            values = block(values)
        pooled: Tensor = values.mean(dim=(2, 3, 4))
        projected: Tensor = self.projection(pooled)
        return projected


class WindowAttention3D(nn.Module):
    def __init__(self, dim: int, num_heads: int, window: int, shift: int = 0) -> None:
        super().__init__()
        if shift < 0 or shift >= window:
            raise ValueError("the window shift must lie in [0, window)")
        self.dim = dim
        self.num_heads = num_heads
        self.window = window
        self.shift = shift
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, tokens: Tensor) -> Tensor:
        batch, length, channels = tokens.shape
        head_dim = channels // self.num_heads
        query: Tensor = self.q_proj(tokens)
        key: Tensor = self.k_proj(tokens)
        value: Tensor = self.v_proj(tokens)
        query = query.reshape(batch, length, self.num_heads, head_dim).transpose(1, 2)
        key = key.reshape(batch, length, self.num_heads, head_dim).transpose(1, 2)
        value = value.reshape(batch, length, self.num_heads, head_dim).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) / float(head_dim) ** 0.5
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(weights, value)
        attended = attended.transpose(1, 2).reshape(batch, length, channels)
        projected: Tensor = self.out_proj(attended)
        return projected


class ShiftedWindowEncoder3D(nn.Module):
    def __init__(self, in_channels: int = 1, dim: int = 32, depth: int = 2, window: int = 4, heads: int = 2) -> None:
        super().__init__()
        self.stem = nn.Conv3d(in_channels, dim, kernel_size=3, stride=2, padding=1)
        self.blocks = nn.ModuleList(
            WindowAttention3D(dim, heads, window, shift=(index % 2) * (window // 2)) for index in range(depth)
        )
        self.norms = nn.ModuleList(nn.GroupNorm(min(8, dim), dim) for _ in range(depth))
        self.window = window
        self.embed_dim = dim

    def forward(self, volume: Tensor) -> Tensor:
        stemmed: Tensor = self.stem(volume)
        for block, norm in zip(self.blocks, self.norms, strict=True):
            windows, grid = window_partition_3d(stemmed, self.window)
            attended = block(windows)
            merged = window_reverse_3d(attended, grid, self.window)
            stemmed = norm(merged) + stemmed
        pooled: Tensor = stemmed.mean(dim=(2, 3, 4))
        return pooled


class CompactVolumeEncoder3D(nn.Module):
    def __init__(self, in_channels: int = 1, dim: int = 32, depth: int = 2, heads: int = 2) -> None:
        super().__init__()
        self.patch = nn.Conv3d(in_channels, dim, kernel_size=4, stride=4)
        self.blocks = nn.ModuleList(WindowAttention3D(dim, heads, window=4, shift=0) for _ in range(depth))
        self.norms = nn.ModuleList(nn.GroupNorm(min(8, dim), dim) for _ in range(depth))
        self.embed_dim = dim

    def forward(self, volume: Tensor) -> Tensor:
        patched: Tensor = self.patch(volume)
        batch, channels = patched.shape[0], patched.shape[1]
        tokens = patched.reshape(batch, channels, -1).transpose(1, 2)
        for block, norm in zip(self.blocks, self.norms, strict=True):
            attended = block(tokens)
            tokens = norm((attended + tokens).transpose(1, 2)).transpose(1, 2)
        pooled: Tensor = tokens.mean(dim=1)
        return pooled


class DescriptorEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 32) -> None:
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, values: Tensor) -> Tensor:
        projected: Tensor = self.stack(values)
        return projected


ENCODER_TABLE: dict[str, type[nn.Module]] = {
    "3d_conv": ConvEncoder3D,
    "shifted_window": ShiftedWindowEncoder3D,
    "foundation_contract": CompactVolumeEncoder3D,
}
