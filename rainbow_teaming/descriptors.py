"""Utilities for converting symbolic descriptors into numeric coordinates."""
from __future__ import annotations

from typing import List, Tuple

RISK_CATEGORIES: List[str] = ["violence", "self-harm", "harassment", "illegal", "other"]
ATTACK_STYLES: List[str] = ["direct", "roleplay", "obfuscation", "multi-step", "other"]


def _normalize_category(category: str, categories: List[str]) -> str:
    normalized = category.strip().lower()
    if normalized not in categories:
        return "other"
    return normalized


def category_to_scalar(category: str, categories: List[str]) -> float:
    """Map a category string to a scalar in [0, 1].

    Unknown categories are mapped to the last entry ("other").
    """

    safe_category = _normalize_category(category, categories)
    idx = categories.index(safe_category)
    if len(categories) == 1:
        return 0.0
    return idx / (len(categories) - 1)


def build_descriptor(risk_category: str, attack_style: str) -> Tuple[float, float]:
    """Return a 2D descriptor vector in [0, 1] x [0, 1].

    Args:
        risk_category: Categorical risk label from :data:`RISK_CATEGORIES`.
        attack_style: Categorical attack style from :data:`ATTACK_STYLES`.

    Returns:
        Two-element tuple representing the normalized descriptor coordinates.
    """

    risk_val = category_to_scalar(risk_category, RISK_CATEGORIES)
    style_val = category_to_scalar(attack_style, ATTACK_STYLES)
    return (risk_val, style_val)


__all__ = [
    "RISK_CATEGORIES",
    "ATTACK_STYLES",
    "category_to_scalar",
    "build_descriptor",
]
