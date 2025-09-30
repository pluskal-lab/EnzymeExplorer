"""This script is used to predict the presence of terpene synthases in a given FASTA file, using TPS language model only, no domains."""

import os
import argparse
from pathlib import Path
from shutil import rmtree
import gdown  # type: ignore
import logging
from time import time
from uuid import uuid4

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def parse_args() -> argparse.Namespace:
    """
    This function parses arguments
    :return: current argparse.Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-fasta-path", type=str, default="data/uniprot_trembl.fasta"
    )
    parser.add_argument("--output-csv-path", type=str, default="trembl_screening")
    parser.add_argument("--detection-threshold", type=float, default=0.3)
    parser.add_argument("--detect-precursor-synthases", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    # checking TPS language model checkpoint presence
    plm_chkpt_path = Path("data/plm_checkpoints")
    if not plm_chkpt_path.exists():
        plm_chkpt_path.mkdir(parents=True)
    plm_path = plm_chkpt_path / "checkpoint-tps-esm1v-t33-subseq.ckpt"
    if not plm_path.exists():
        logger.info("Downloading TPS language model checkpoint..")
        url = "https://drive.google.com/uc?id=1jU76oUl0-CmiB9m3XhaKmI2HorFhyxC7"
        gdown.download(url, str(plm_path), quiet=False)
    clf_chkpt_path = Path("data/classifier_plm_checkpoints.pkl")
    if not clf_chkpt_path.exists():
        logger.info("Downloading classifier checkpoints..")
        url = "https://drive.google.com/uc?id=15_OFrrVUy9r9Urj-R2CjTRj_DHcazdAl"
        gdown.download(url, str(clf_chkpt_path), quiet=False)

    run_id = f"{int(time() * 1000)}-{uuid4()}"
    intermediate_outputs_root = f"_temp_dir_{run_id}"
    if not Path(intermediate_outputs_root).exists():
        Path(intermediate_outputs_root).mkdir(parents=True)
    os.system(
        "python -m enzymeexplorer.src.screening.tps_predict_fasta --model esm-1v-finetuned-subseq"
        f" --fasta-path {args.input_fasta_path} --output-root {intermediate_outputs_root}"
        f" --detect-precursor-synthases {args.detect_precursor_synthases}"
        f" --detection-threshold {args.detection_threshold}"
        f" --ckpt-root-path {clf_chkpt_path}"
    )
    os.system(
        f"python -m enzymeexplorer.src.screening.gather_detections_to_csv --screening-results-root {intermediate_outputs_root}/detections_plm --output-path {args.output_csv_path} --delete-individual-files"
    )
    rmtree(intermediate_outputs_root)


if __name__ == "__main__":
    main()
