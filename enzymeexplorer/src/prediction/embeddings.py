"""PLM embedding helpers for predict-time usage.

Wraps the ``esm_transformer_utils`` and ``ankh_transformer_utils`` factories
so the model is loaded once and a list of sequences is embedded in
mini-batches. Backend is auto-selected from the model name:

* names containing ``"esm"`` → ESM family (ESM-1v, ESM-1b, ESM-2, fine-tuned
  variants). ``repr_layer`` defaults to 33 (ESM-1v / ESM-1b); ``max_seq_len``
  defaults to 1022 (ESM-1v cap, applied as truncation).
* names containing ``"ankh"`` → Ankh family. No fixed input cap — Ankh's
  encoder handles arbitrary lengths up to memory limits — so
  ``max_seq_len`` defaults to ``None`` and is ignored unless the caller
  explicitly sets one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enzymeexplorer.src.embeddings_extraction.esm_transformer_utils import MODEL_LAYERS
from typing import Iterable

import numpy as np  # type: ignore
from tqdm.auto import tqdm  # type: ignore

logger = logging.getLogger(__name__)


def _model_family(model_name: str) -> str:
    if "esm" in model_name:
        return "esm"
    if "ankh" in model_name:
        return "ankh"
    raise ValueError(
        f"Unknown PLM family for model_name={model_name!r}; expected an "
        f"'esm'- or 'ankh'-prefixed name."
    )


@dataclass
class PLMEmbedder:
    """Stateful embedder bound to a single PLM checkpoint."""

    model_name: str
    repr_layer: int = 33
    max_seq_len: int | None = None
    family: str = field(init=False)
    _model: object = None
    _converter: object = None
    _padding_idx: int = 0

    def __post_init__(self) -> None:
        self.family = _model_family(self.model_name)
        if self.max_seq_len is None and self.family == "esm":
            # Default truncation cap for ESM-1v/1b. ESM-2 also accepts this
            # cap; callers wanting ESM-2's nominal infinite context can pass
            # ``max_seq_len=None`` explicitly via a custom value at load.
            self.max_seq_len = 2000

    def load(self) -> "PLMEmbedder":
        if self.family == "esm":
            from enzymeexplorer.src.embeddings_extraction.esm_transformer_utils import (
                get_model_and_tokenizer as esm_get_model_and_tokenizer,
            )
            model, converter, alphabet = esm_get_model_and_tokenizer(
                self.model_name, return_alphabet=True
            )
            self._model = model
            self._converter = converter
            self._padding_idx = alphabet.padding_idx
        else:
            from enzymeexplorer.src.embeddings_extraction.ankh_transformer_utils import (
                get_model_and_tokenizer as ankh_get_model_and_tokenizer,
            )
            model, tokenizer = ankh_get_model_and_tokenizer(self.model_name)
            self._model = model
            self._converter = tokenizer
            self._padding_idx = 0  # unused for Ankh
        logger.info("Loaded PLM embedder: %s (family=%s)", self.model_name, self.family)
        return self

    def embed(
        self,
        sequences: Iterable[str],
        *,
        batch_size: int = 4,
        progress_desc: str | None = None,
    ) -> np.ndarray:
        """Embed sequences in mini-batches and return a [N, D] float array.

        Sequences longer than ``self.max_seq_len`` are truncated when the
        cap is set (default for ESM family; ``None`` for Ankh). An empty
        ``sequences`` returns an empty 2-D array.
        """
        if self._model is None:
            self.load()
        if self.max_seq_len is not None:
            seqs = [
                (s if len(s) <= self.max_seq_len else s[: self.max_seq_len - 2])
                for s in sequences
            ]
        else:
            seqs = list(sequences)
        if not seqs:
            return np.zeros((0, 0), dtype=np.float32)

        out_chunks: list[np.ndarray] = []
        iterator = range(0, len(seqs), batch_size)
        if progress_desc is not None:
            iterator = tqdm(iterator, desc=progress_desc)

        if self.family == "esm":
            from enzymeexplorer.src.embeddings_extraction.esm_transformer_utils import (
                compute_embeddings as esm_compute_embeddings,
            )
            for start in iterator:
                chunk = seqs[start : start + batch_size]
                (encodings, _) = esm_compute_embeddings(
                    bert_model=self._model,
                    converter=self._converter,
                    padding_idx=self._padding_idx,
                    input_seqs=chunk,
                    model_repr_layer=self.repr_layer,
                    max_len=self.max_seq_len,
                )
                out_chunks.append(np.asarray(encodings))
        else:
            from enzymeexplorer.src.embeddings_extraction.ankh_transformer_utils import (
                compute_embeddings as ankh_compute_embeddings,
            )
            for start in iterator:
                chunk = seqs[start : start + batch_size]
                (encodings, _) = ankh_compute_embeddings(
                    bert_model=self._model,
                    tokenizer=self._converter,
                    input_seqs=chunk,
                )
                out_chunks.append(np.asarray(encodings))
        return np.concatenate(out_chunks, axis=0)


def load_plm_embedder(
    model_name: str = "ankh_large",
    *,
    max_seq_len: int | None = None,
) -> PLMEmbedder:
    """Build and load a :class:`PLMEmbedder`."""
    if "esm" in model_name:
        repr_layer = MODEL_LAYERS[model_name]
    else:        
        repr_layer = -1  # ignored for Ankh
    return PLMEmbedder(
        model_name=model_name, repr_layer=repr_layer, max_seq_len=max_seq_len,
    ).load()
