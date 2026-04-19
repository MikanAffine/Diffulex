from __future__ import annotations

from enum import Enum


class DeepEPMode(Enum):
    NORMAL = "normal"
    LOW_LATENCY = "low_latency"
    AUTO = "auto"

    @classmethod
    def from_value(cls, value: str | "DeepEPMode") -> "DeepEPMode":
        if isinstance(value, cls):
            return value
        return cls(value)

    def enable_normal(self) -> bool:
        return self in {DeepEPMode.NORMAL, DeepEPMode.AUTO}

    def enable_low_latency(self) -> bool:
        return self in {DeepEPMode.LOW_LATENCY, DeepEPMode.AUTO}

    def resolve(self, phase: str) -> "DeepEPMode":
        if self is not DeepEPMode.AUTO:
            return self
        if phase == "decode":
            return DeepEPMode.LOW_LATENCY
        return DeepEPMode.NORMAL


__all__ = ["DeepEPMode"]
