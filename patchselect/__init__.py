"""Patch selection utilities for large-scale IHC Arrow datasets."""

from patchselect.config import GlobalSelectionConfig, PatchSelectionConfig
from patchselect.descriptor_backend import available_descriptor_backends
from patchselect.pipeline import select_patches_from_image

__all__ = [
    "GlobalSelectionConfig",
    "PatchSelectionConfig",
    "available_descriptor_backends",
    "select_patches_from_image",
]
