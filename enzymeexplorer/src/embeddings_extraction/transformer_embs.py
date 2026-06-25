"""This script extracts PLM embeddings for UniProt proteins.

For ESM models a single forward pass can return hidden states from several
representation layers — pass a comma-separated list to
``--model-repr-layer`` (e.g. ``29,30,31,32,33``) and the script writes one
per-layer output dir ``uniprot_embs_<model>-L<layer>/`` under
``--output-root-path``. The legacy single-int form keeps working and
produces the same per-layer subdir for that one layer. Ankh models ignore
``--model-repr-layer`` and write to ``uniprot_embs_<model>/`` as before.
"""
# pylint: disable=R0801
import argparse  # type: ignore
import logging  # type: ignore
import os  # type: ignore
import pickle  # type: ignore
from functools import partial
from pathlib import Path
import pandas as pd  # type: ignore
import torch  # type: ignore
from tqdm.auto import trange  # type: ignore
from tqdm.contrib.logging import logging_redirect_tqdm  # type: ignore

from enzymeexplorer.src.embeddings_extraction.ankh_transformer_utils import (
    compute_embeddings as ankh_compute_embeddings,
)
from enzymeexplorer.src.embeddings_extraction.ankh_transformer_utils import (
    get_model_and_tokenizer as ankh_get_model_and_tokenizer,
)
from enzymeexplorer.src.embeddings_extraction.esm_transformer_utils import (
    compute_embeddings as esm_compute_embeddings,
)
from enzymeexplorer.src.embeddings_extraction.esm_transformer_utils import (
    get_model_and_tokenizer as esm_get_model_and_tokenizer,
)

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def _parse_layers(spec: str) -> list[int]:
    """Parse ``"33"`` or ``"29,30,31,32,33"`` into a list of ints."""
    return [int(s) for s in str(spec).split(",") if s.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--model", type=str, default="esm-1v-1")
    parser.add_argument(
        "--model-repr-layer", type=str, default="33",
        help=(
            "Representation layer(s). Single int (legacy) or comma-separated "
            "list (e.g. '29,30,31,32,33') to extract every layer in one "
            "forward pass — outputs go to per-layer subdirs."
        ),
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=11_000)
    parser.add_argument(
        "--csv-path",
        type=str,
        default="data/EnzymeExplorer_Dataset.csv",
    )
    parser.add_argument("--id-column", type=str, default="ID")
    parser.add_argument("--seq-column", type=str, default="Aminoacid_sequence")
    parser.add_argument("--output-root-path", type=str, default="outputs/ankh_embs")

    args = parser.parse_args()
    return args


def _dump_per_layer(
    out_dir: Path,
    gpu: str,
    batch_i: int,
    ids_chunk: list,
    encodings: "object",
    seqs: "object",
) -> None:
    """Pickle the three per-batch files into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"batch_{gpu}_{batch_i}_embs_avg.pkl", "wb") as f:
        pickle.dump(encodings, f)
    with open(out_dir / f"batch_{gpu}_{batch_i}_embs_seqs.pkl", "wb") as f:
        pickle.dump(seqs, f)
    with open(out_dir / f"batch_{gpu}_{batch_i}_ids.pkl", "wb") as f:
        pickle.dump(ids_chunk, f)


def main():
    """Extract PLM embeddings for the specified proteins."""
    cli_args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = cli_args.gpu
    root_path = Path(cli_args.output_root_path)

    is_esm = "esm" in cli_args.model
    if is_esm:
        layers = _parse_layers(cli_args.model_repr_layer)
        logger.info(
            "Loading ESM-family model %s; requesting layers %s",
            cli_args.model, layers,
        )
        model, batch_converter, alphabet = esm_get_model_and_tokenizer(
            cli_args.model, return_alphabet=True
        )
        # ESM-2 has no architectural cap and we deliberately don't impose
        # one — the per-batch OOM recovery below falls back to single-
        # sequence runs if a multi-sequence batch trips OOM, so long
        # sequences don't drag the whole batch with them. ESM-1v keeps
        # its native 1022 cap (model architecture limit).
        max_len = 424242_000 if "esm-2" in cli_args.model else 1022

        # If only one layer requested, keep the legacy (single-tensor) return
        # shape so downstream code paths are identical to before.
        compute_kwargs = dict(
            bert_model=model,
            converter=batch_converter,
            padding_idx=alphabet.padding_idx,
            max_len=max_len,
        )
        compute_embeddings_partial = partial(
            esm_compute_embeddings,
            model_repr_layer=(layers if len(layers) > 1 else layers[0]),
            **compute_kwargs,
        )
        # Per-layer subdir naming: 'uniprot_embs_<model>-L<layer>'.
        out_dirs = {L: root_path / f"uniprot_embs_{cli_args.model}-L{L}" for L in layers}
    elif "ankh" in cli_args.model:
        layers = None
        logger.info("Loading Ankh-family model %s", cli_args.model)
        model, tokenizer = ankh_get_model_and_tokenizer(cli_args.model)
        compute_embeddings_partial = partial(
            ankh_compute_embeddings, bert_model=model, tokenizer=tokenizer
        )
        out_dirs = {None: root_path / f"uniprot_embs_{cli_args.model}"}
    else:
        raise NotImplementedError(
            f"Model {cli_args.model} is not supported. Currently only esm, ankh model families are supported"
        )

    logger.info("Model was loaded! Reading data...")
    df = pd.read_csv(cli_args.csv_path)
    df = df.drop_duplicates(cli_args.id_column)
    df = df.sort_values(by=cli_args.id_column)
    ids_list = df[cli_args.id_column].values[
        cli_args.start_index : cli_args.end_index + 1
    ]
    seqs_list = df[cli_args.seq_column].values[
        cli_args.start_index : cli_args.end_index + 1
    ]

    logger.info("Data are ready! Output dirs: %s", list(out_dirs.values()))
    batch_size = cli_args.batch_size
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    def _process_and_dump(batch_i: int, seqs_chunk: list, ids_chunk: list) -> bool:
        """Run one (sub-)batch and persist its pickles. Returns True on success."""
        result = compute_embeddings_partial(input_seqs=seqs_chunk)
        if is_esm and layers is not None and len(layers) > 1:
            per_layer_np, per_layer_seqs = result
            for L in layers:
                _dump_per_layer(
                    out_dirs[L], cli_args.gpu, batch_i, ids_chunk,
                    per_layer_np[L], per_layer_seqs[L],
                )
        else:
            enc, seqs = result
            target_dir = out_dirs[layers[0]] if (is_esm and layers) else out_dirs[None]
            _dump_per_layer(
                target_dir, cli_args.gpu, batch_i, ids_chunk, enc, seqs,
            )
        return True

    processed_seqs: list = []
    failed_ids: list = []
    with logging_redirect_tqdm([logger]):
        for batch_i in trange(len(seqs_list) // batch_size + 1):
            input_seq_list_batch = seqs_list[
                batch_i * batch_size : (batch_i + 1) * batch_size
            ]
            if not len(input_seq_list_batch):
                continue
            ids_chunk = ids_list[batch_i * batch_size : (batch_i + 1) * batch_size]
            try:
                _process_and_dump(batch_i, list(input_seq_list_batch), list(ids_chunk))
                processed_seqs.extend(input_seq_list_batch)
            except torch.cuda.OutOfMemoryError:
                # Fallback: retry one sequence at a time so a single offending
                # long sequence in a multi-batch doesn't take the whole batch
                # with it. Pickle filenames embed both batch_i and a sub-index
                # so per-sequence pickles don't collide with the original
                # batch's would-be filenames.
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.warning(
                    "Batch %d (range %d-%d) OOM at batch_size=%d — retrying one-at-a-time",
                    batch_i, cli_args.start_index, cli_args.end_index, batch_size,
                )
                for sub_i, (one_seq, one_id) in enumerate(zip(input_seq_list_batch, ids_chunk)):
                    sub_batch_id = batch_i * 10_000 + sub_i + 9_000_000  # disjoint from regular numbering
                    try:
                        _process_and_dump(sub_batch_id, [one_seq], [one_id])
                        processed_seqs.append(one_seq)
                    except torch.cuda.OutOfMemoryError:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        logger.error(
                            "Sequence %s (length=%d) STILL OOM at batch_size=1; "
                            "skipping permanently. Consider retrying with a "
                            "harder max_len cap or on a larger GPU.",
                            one_id, len(one_seq),
                        )
                        failed_ids.append(one_id)
                        continue

    count_of_unprocessed_entries = len(set(seqs_list).difference(set(processed_seqs)))
    if count_of_unprocessed_entries:
        logger.warning("Unprocessed seqs count: %d", count_of_unprocessed_entries)
    if failed_ids:
        logger.warning(
            "Persisted OOM sequences (skipped even at batch=1): %s",
            ", ".join(map(str, failed_ids)),
        )


if __name__ == "__main__":
    main()
