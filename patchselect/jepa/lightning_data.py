"""PyTorch Lightning DataModule for stateful tar-stream JEPA data."""

from __future__ import annotations

import logging
from typing import Any

import lightning as pl
import torch
from torch.utils.data import IterableDataset

from patchselect.jepa.config import JEPAConfig
from patchselect.jepa.data import (
    PipelinedBatchLoader,
    TarImageStream,
    build_train_transform,
    discover_tar_manifest,
    manifest_metadata,
)

logger = logging.getLogger(__name__)


class _PipelineIterableDataset(IterableDataset):
    """Thin IterableDataset wrapper over PipelinedBatchLoader.

    Lightning expects a DataLoader, which expects an IterableDataset.
    Since PipelinedBatchLoader already handles batching, decoding, and
    prefetching internally, we yield pre-batched dicts and use
    batch_size=1 with a custom collate in the DataLoader to pass them
    through unchanged.
    """

    def __init__(self, pipeline: PipelinedBatchLoader) -> None:
        super().__init__()
        self.pipeline = pipeline

    def __iter__(self):
        loader_iter = iter(self.pipeline)
        while True:
            try:
                batch = next(loader_iter)
                yield batch
            except StopIteration:
                return


def _passthrough_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Identity collate — batches are already assembled by the pipeline."""
    return batch[0]


class JEPADataModule(pl.LightningDataModule):
    """LightningDataModule for the JEPA stateful tar-stream pipeline.

    Wraps the existing TarImageStream + PipelinedBatchLoader pipeline.
    Cursor state (tar_index, member_index) is tracked for checkpoint
    resumability.
    """

    def __init__(self, cfg: JEPAConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Cursor state for resume
        self._start_tar_index: int = 0
        self._start_member_index: int = 0

        # Stateful references set during setup
        self._dataset: TarImageStream | None = None
        self._pipeline: PipelinedBatchLoader | None = None
        self._manifest: list | None = None
        self._transform = None

    @property
    def manifest(self):
        """Access the tar manifest (lazy-discovered)."""
        if self._manifest is None:
            self._manifest = discover_tar_manifest(self.cfg.data)
        return self._manifest

    @property
    def manifest_meta(self) -> list[dict[str, Any]]:
        return manifest_metadata(self.manifest)

    @property
    def dataset(self) -> TarImageStream | None:
        return self._dataset

    @property
    def data_cursor(self) -> dict[str, int]:
        """Current data cursor for checkpoint state."""
        return {
            "tar_index": self._start_tar_index,
            "member_index": self._start_member_index,
        }

    def set_cursor(self, tar_index: int, member_index: int) -> None:
        """Restore cursor state from checkpoint."""
        self._start_tar_index = tar_index
        self._start_member_index = member_index

    def setup(self, stage: str | None = None) -> None:
        if stage is not None and stage != "fit":
            return

        self._transform = build_train_transform(self.cfg.augment)
        self._dataset = TarImageStream(
            manifest=self.manifest,
            transform=self._transform,
            start_tar_index=self._start_tar_index,
            start_member_index=self._start_member_index,
            cache_in_ram=self.cfg.data.cache_in_ram,
            local_cache_dir=self.cfg.data.local_cache_dir,
            download_cmd=self.cfg.data.download_cmd,
            prefetch_depth=self.cfg.data.prefetch_depth,
            prefetch_timeout_sec=self.cfg.data.prefetch_timeout_sec,
            extensions=self.cfg.data.extensions,
        )
        self._pipeline = PipelinedBatchLoader(
            self._dataset,
            batch_size=self.cfg.runtime.batch_size,
            transform=self._transform,
            decode_threads=self.cfg.data.decode_threads,
            prefetch_batches=self.cfg.data.prefetch_depth,
            pin_memory=self.cfg.data.pin_memory,
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        assert self._pipeline is not None, "Call setup() before train_dataloader()"
        iterable = _PipelineIterableDataset(self._pipeline)
        return torch.utils.data.DataLoader(
            iterable,
            batch_size=1,
            collate_fn=_passthrough_collate,
            num_workers=0,
            pin_memory=False,  # Pipeline handles pinning internally
        )
