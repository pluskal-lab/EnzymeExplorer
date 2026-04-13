"""This script contains utils for ESM embeddings extraction"""
import logging
from typing import Optional

import esm  # type: ignore
import numpy as np  # type: ignore
import torch  # type: ignore
import logging
from pathlib import Path
import gdown  # type: ignore
from filelock import FileLock

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

CHECKPOINT_NAMES = {
    "esm-1v-finetuned": "checkpoint-tps-esm1v-t33-finetuned.ckpt",
    "esm-1v-finetuned-subseq": "checkpoint-tps-esm1v-t33-subseq.ckpt",
}

MODEL_LAYERS = {
    "esm-1b": 33,
    "esm-1v": 33,
    "esm-1v-finetuned": 33,
    "esm-1v-finetuned-subseq": 33,
    "esm-2": 36,
    "esm-2-t30": 30,
}


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
    """
    This function returns bert model and batch converter (basically a tokenizer) based on the name
    :param model_name: model name
    :param checkpoint_names: mapping between model name and checkpoint file
    :param return_alphabet: flag to return alphabet object
    :param checkpoint_dir: directory where checkpoints are stored
    :return: a pair of the bert protein model and its batch converter
    """
    if checkpoint_names is None:
        checkpoint_names = CHECKPOINT_NAMES
    if model_name in checkpoint_names:
        checkpoint_name = checkpoint_names[model_name]
        file_lock = FileLock(f"{checkpoint_name}.lock")
        with file_lock:
            download_plm_checkpoint(checkpoint_name, checkpoint_dir)
        ckpt = torch.load(
            f"{checkpoint_dir}/{checkpoint_name}",
            map_location=torch.device("cpu"),
        )
        bert_model, bert_alphabet = getattr(esm.pretrained, "esm1v_t33_650M_UR90S_1")()
        bert_model.load_state_dict(ckpt["state_dict"])
    elif model_name == "esm-2":
        bert_model, bert_alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    elif model_name == "esm-1b":
        bert_model, bert_alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    elif model_name == "esm-1v":
        bert_model, bert_alphabet = getattr(esm.pretrained, "esm1v_t33_650M_UR90S_1")()
    elif model_name == "esm-2-t30":
        bert_model, bert_alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    else:
        raise NotImplementedError(f"ESM transformer {model_name} is not supported")
    bert_batch_converter = bert_alphabet.get_batch_converter()
    if torch.cuda.is_available():
        bert_model = bert_model.to(device="cuda:0", non_blocking=True)
    if return_alphabet:
        return bert_model, bert_batch_converter, bert_alphabet
    return bert_model, bert_batch_converter


def compute_embeddings(
    bert_model: esm.ProteinBertModel | esm.ESM2,
    converter: esm.data.BatchConverter,
    padding_idx: int,
    input_seqs: list[str],
    model_repr_layer: int,
    max_len: int = 2000,
) -> tuple[np.ndarray, list]:
    """
    This function computes bert_model embeddings of protein sequences
    :param bert_model: trained Protein Bert model
    :param converter: fair batch converter
    :param padding_idx: padding index
    :param input_seqs: list of protein sequences stored
    :param model_repr_layer: index of representation layer in the model
    :param max_len: maximum allowed length of a protein sequence
    :return: numpy array with average protein embeddings and array with a sequence of amino acid embeddings
    """
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
        bert_embs = bert_model(tokens, repr_layers=[model_repr_layer])

    token_representations = bert_embs["representations"][model_repr_layer].cpu().numpy()
    encodings_batch = []
    encoding_seqs_batch = []
    for i, tokens_len in enumerate(batch_lens):
        embs_per_tokens = token_representations[i, 1 : tokens_len - 1]
        encodings_batch.append(embs_per_tokens.mean(0))
        encoding_seqs_batch.append(embs_per_tokens)
    encodings_np_batch = np.array(encodings_batch)
    return encodings_np_batch, encoding_seqs_batch
