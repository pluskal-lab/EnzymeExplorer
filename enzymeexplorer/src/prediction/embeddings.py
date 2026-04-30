"""PLM embedding helpers for predict-time usage.

Wraps ``esm_transformer_utils.compute_embeddings`` so the model is loaded once
and a list of sequences is embedded in mini-batches. Matches the configuration
the classifiers were trained with (layer 33, max-len 1022 for ESM-1v family).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np  # type: ignore
from tqdm.auto import tqdm  # type: ignore

from enzymeexplorer.src.embeddings_extraction.esm_transformer_utils import (
    compute_embeddings,
    get_model_and_tokenizer,
)


@dataclass
class PLMEmbedder:
    """Stateful embedder bound to a single PLM checkpoint."""

    model_name: str
    repr_layer: int = 33
    max_seq_len: int = 1022
    _model: object = None
    _converter: object = None
    _padding_idx: int = 0

    def load(self) -> "PLMEmbedder":
        model, converter, alphabet = get_model_and_tokenizer(
            self.model_name, return_alphabet=True
        )
        self._model = model
        self._converter = converter
        self._padding_idx = alphabet.padding_idx
        return self

    def embed(
        self,
        sequences: Iterable[str],
        *,
        batch_size: int = 4,
        progress_desc: str | None = None,
    ) -> np.ndarray:
        """Embed sequences in mini-batches and return a [N, D] float array.

        Sequences longer than ``self.max_seq_len`` are truncated; an empty
        ``sequences`` returns an empty 2-D array.
        """
        if self._model is None:
            self.load()
        seqs = [
            (s if len(s) <= self.max_seq_len else s[: self.max_seq_len - 2])
            for s in sequences
        ]
        if not seqs:
            return np.zeros((0, 0), dtype=np.float32)

        out_chunks: list[np.ndarray] = []
        iterator = range(0, len(seqs), batch_size)
        if progress_desc is not None:
            iterator = tqdm(iterator, desc=progress_desc)
        for start in iterator:
            chunk = seqs[start : start + batch_size]
            (encodings, _) = compute_embeddings(
                bert_model=self._model,
                converter=self._converter,
                padding_idx=self._padding_idx,
                input_seqs=chunk,
                model_repr_layer=self.repr_layer,
                max_len=self.max_seq_len,
            )
            out_chunks.append(np.asarray(encodings))
        return np.concatenate(out_chunks, axis=0)


def load_plm_embedder(
    model_name: str = "esm-1v-finetuned-subseq",
    *,
    repr_layer: int = 33,
    max_seq_len: int = 1022,
) -> PLMEmbedder:
    """Build and load a :class:`PLMEmbedder`."""
    return PLMEmbedder(
        model_name=model_name, repr_layer=repr_layer, max_seq_len=max_seq_len
    ).load()
