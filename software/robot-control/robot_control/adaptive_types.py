"""Small immutable configuration types shared by candidate policies."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyConfiguration:
    cameras: tuple[str, ...]
    depth: tuple[str, ...]
    image_size: tuple[int, int]
    image_format: str
    action: str
    prefetch: bool
    dt_s: float
