"""This script contains utils for ESM embeddings extraction"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional, Union

import esm  # type: ignore
import gdown  # type: ignore
import numpy as np  # type: ignore
import torch  # type: ignore
from filelock import FileLock

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

CHECKPOINT_NAMES = {
    "esm-1v-finetuned-subseq": "checkpoint-tps-esm1v-t33-subseq.ckpt",
}

# Final-layer index per base PLM. The ESM-2 ablation variants name themselves
# ``esm-2-t{NN}-L{MM}`` where NN is the depth and MM is the representation
# layer (1..NN). ``MODEL_LAYERS`` is the legacy lookup used by callers that
# don't know about per-layer variants — it returns the FINAL layer for each
# canonical model name. Use ``get_repr_layer(name)`` for variant-aware
# resolution.
MODEL_LAYERS = {
    "esm-1b": 33,
    "esm-1v": 33,
    "esm-1v-finetuned-subseq": 33,
    "esm-2-t30": 30,
    "esm-2-t33": 33,
    "esm-2-t36": 36,
}

# Loader mapping for the canonical ESM-2 base models.
ESM2_BASE_LOADERS = {
    "esm-2-t30": "esm2_t30_150M_UR50D",
    "esm-2-t33": "esm2_t33_650M_UR50D",
    "esm-2-t36": "esm2_t36_3B_UR50D",
}

# Variant name pattern: ``esm-2-t30-L26``, ``esm-2-t33-L31``, etc. Layer
# index is captured as group 2.
_ESM2_VARIANT_RE = re.compile(r"^(esm-2-t\d+)-L(\d+)$")


def parse_esm2_variant(model_name: str) -> tuple[str, int | None]:
    """Split an ESM-2 variant name into (base, repr_layer).

    ``esm-2-t33-L31`` -> ``("esm-2-t33", 31)``. A bare base name like
    ``esm-2-t36`` returns ``("esm-2-t36", None)`` so the caller can fall
    back to the model's final layer via ``MODEL_LAYERS``.
    """
    m = _ESM2_VARIANT_RE.match(model_name)
    if m:
        return m.group(1), int(m.group(2))
    return model_name, None


def get_repr_layer(model_name: str) -> int:
    """Resolve the representation layer for a model name (variant-aware).

    For ablation variants like ``esm-2-t33-L31`` the parsed L<MM> wins; for
    canonical names (``esm-2-t36``, ``esm-1v``, ...) the final layer from
    ``MODEL_LAYERS`` is returned. Raises ``KeyError`` if the model isn't
    recognised — caller code that needs a soft default should pre-check.
    """
    base, layer = parse_esm2_variant(model_name)
    if layer is not None:
        return layer
    return MODEL_LAYERS[base]


def download_plm_checkpoint(checkpoint_name: str, checkpoint_dir: Optional[str] = "data/plm_checkpoints") -> None:
    logger.info("checking TPS language model checkpoint presence")
    plm_chkpt_path = Path(checkpoint_dir if checkpoint_dir is not None else "data/plm_checkpoints")
    if not plm_chkpt_path.exists():
        plm_chkpt_path.mkdir(parents=True)
    plm_path = plm_chkpt_path / checkpoint_name
    if not plm_path.exists():
        logger.info("Downloading TPS language model checkpoint..")
        url = "https://drive.google.com/uc?id=1jU76oUl0-CmiB9m3XhaKmI2HorFhyxC7"
        gdown.download(url, str(plm_path), quiet=False)


def get_model_and_tokenizer(
    model_name: str,
    checkpoint_names: Optional[dict[str, str]] = None,
    return_alphabet: bool = False,
    checkpoint_dir: Optional[str] = "data/plm_checkpoints",
) -> tuple:
    """Return ``(bert_model, batch_converter[, alphabet])`` for ``model_name``.

    Accepts both canonical names (``esm-2-t33``) and ablation variants
    (``esm-2-t33-L31``) — the variant suffix only affects which layer is
    later requested in ``compute_embeddings``; the BASE model loaded is the
    same for every layer of a given depth.
    """
    if checkpoint_names is None:
        checkpoint_names = CHECKPOINT_NAMES

    base_model, _ = parse_esm2_variant(model_name)

    if model_name in checkpoint_names:
        # Fine-tuned ESM-1v checkpoints loaded over the base 650M weights.
        checkpoint_name = checkpoint_names[model_name]
        ckpt_dir_path = Path(
            checkpoint_dir if checkpoint_dir is not None else "data/plm_checkpoints"
        )
        ckpt_dir_path.mkdir(parents=True, exist_ok=True)
        file_lock = FileLock(str(ckpt_dir_path / f"{checkpoint_name}.lock"))
        with file_lock:
            download_plm_checkpoint(checkpoint_name, checkpoint_dir)
        ckpt = torch.load(
            f"{checkpoint_dir}/{checkpoint_name}",
            map_location=torch.device("cpu"),
        )
        bert_model, bert_alphabet = getattr(esm.pretrained, "esm1v_t33_650M_UR90S_1")()
        bert_model.load_state_dict(ckpt["state_dict"])
    elif base_model in ESM2_BASE_LOADERS:
        loader = getattr(esm.pretrained, ESM2_BASE_LOADERS[base_model])
        bert_model, bert_alphabet = loader()
    elif model_name == "esm-1b":
        bert_model, bert_alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    elif model_name == "esm-1v":
        bert_model, bert_alphabet = getattr(esm.pretrained, "esm1v_t33_650M_UR90S_1")()
    else:
        raise NotImplementedError(f"ESM transformer {model_name} is not supported")
    bert_batch_converter = bert_alphabet.get_batch_converter()
    if torch.cuda.is_available():
        bert_model = bert_model.to(device="cuda:0", non_blocking=True)
    if return_alphabet:
        return bert_model, bert_batch_converter, bert_alphabet
    return bert_model, bert_batch_converter


def compute_embeddings(
    bert_model: "esm.ProteinBertModel | esm.ESM2",
    converter: esm.data.BatchConverter,
    padding_idx: int,
    input_seqs: list[str],
    model_repr_layer: Union[int, list[int]],
    max_len: int = 2000,
) -> Union[
    tuple[np.ndarray, list],
    tuple[dict[int, np.ndarray], dict[int, list]],
]:
    """Compute ESM embeddings for ``input_seqs``.

    Two return modes — toggled by ``model_repr_layer``:

    * ``int`` (legacy): returns ``(encodings_np[N, D], per_token_seqs[N])``
      for that single layer. Existing callers (training pipeline, screening)
      go through this path unchanged.
    * ``list[int]`` (multi-layer ablation): returns
      ``({L: encodings_np}, {L: per_token_seqs})`` — both dicts keyed by
      layer index. A single forward pass produces all requested layers'
      hidden states, which is what makes the layer ablation cheap.
    """
    multi = isinstance(model_repr_layer, (list, tuple))
    layers: list[int] = list(model_repr_layer) if multi else [int(model_repr_layer)]

    input_tuple_seqs = [
        (
            f"id{i}",
            "".join(amino_acid_seq.split())
            .replace('"', "")
            .replace("'", "")
            .replace("*", "")[:max_len],
        )
        for i, amino_acid_seq in enumerate(input_seqs)
    ]
    _, _, tokens = converter(input_tuple_seqs)
    batch_lens = (tokens != padding_idx).sum(1)
    if torch.cuda.is_available():
        tokens = tokens.to(device="cuda:0", non_blocking=True)
    with torch.no_grad():
        bert_embs = bert_model(tokens, repr_layers=layers)

    per_layer_np: dict[int, np.ndarray] = {}
    per_layer_seqs: dict[int, list] = {}
    for L in layers:
        token_representations = bert_embs["representations"][L].cpu().numpy()
        encodings_batch: list = []
        encoding_seqs_batch: list = []
        for i, tokens_len in enumerate(batch_lens):
            embs_per_tokens = token_representations[i, 1 : tokens_len - 1]
            encodings_batch.append(embs_per_tokens.mean(0))
            encoding_seqs_batch.append(embs_per_tokens)
        per_layer_np[L] = np.array(encodings_batch)
        per_layer_seqs[L] = encoding_seqs_batch

    if multi:
        return per_layer_np, per_layer_seqs
    only = layers[0]
    return per_layer_np[only], per_layer_seqs[only]
