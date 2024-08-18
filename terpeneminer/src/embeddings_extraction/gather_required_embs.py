"""This script stores extracted PLM embeddings of interested as h5 file with pandas dataframe"""

import argparse
import logging
import os
import pickle
from shutil import rmtree

import pandas as pd  # type: ignore

logger = logging.getLogger("Gathering embeddings into a h5 file for faster access")
logger.setLevel(logging.INFO)


def parse_args() -> argparse.Namespace:
    """
    This function parses arguments
    :return: current argparse.Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root-path",
        type=str,
        default="outputs/ankh_embs/uniprot_embs_ankh_base/",
    )
    parser.add_argument("--embs-suffix", type=str, default="embs_avg")
    parser.add_argument("--process-all", action="store_true")
    parser.add_argument("--storage-step", type=int, default=200)
    parser.add_argument("--csv-path", type=str)
    parser.add_argument("--id-column", type=str, default="Uniprot ID")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--remove-raw-files-on-end", action="store_true")
    args = parser.parse_args()
    return args


def main():
    """
    This function gathers embeddings into a h5 file for faster access
    @return:
    """
    logger.info("Reading data.")
    cli_args = parse_args()
    df = pd.read_csv(cli_args.csv_path)
    root_path = cli_args.input_root_path
    required_ids = set(df[cli_args.id_column].values)

    available_files = set(os.listdir(root_path))

    all_id_files = [
        x
        for x in available_files
        if "ids" in x and x.replace("ids", cli_args.embs_suffix) in available_files
    ]

    all_found_ids = []
    all_found_embs = []

    i = 0
    while (required_ids or cli_args.process_all) and i < len(all_id_files):
        id_file = all_id_files[i]

        with open(os.path.join(root_path, id_file), "rb") as f:
            uniprot_ids = pickle.load(f)

        embeddings = None
        for id_idx, uniprot_id in enumerate(uniprot_ids):
            if uniprot_id in required_ids or cli_args.process_all:
                if uniprot_id in required_ids:
                    required_ids.remove(uniprot_id)
                if cli_args.verbose and len(required_ids) % 500 == 0:
                    logger.info("Remains %d IDs to find", len(required_ids))
                if embeddings is None:
                    with open(
                        os.path.join(
                            root_path,
                            id_file.replace("ids", cli_args.embs_suffix),
                        ),
                        "rb",
                    ) as f:
                        embeddings = pickle.load(f)
                all_found_ids.append(uniprot_id)
                all_found_embs.append(embeddings[id_idx])
        i += 1

    if required_ids:
        logger.warning("Remains %d IDs to find", len(required_ids))

    results_df = pd.DataFrame(
        {cli_args.id_column: all_found_ids, "Emb": all_found_embs}
    )
    if cli_args.embs_suffix == "embs_seqs":
        for part_i, _ in enumerate(range(0, len(results_df), cli_args.storage_step)):
            results_df.iloc[
                part_i * cli_args.storage_step : (part_i + 1) * cli_args.storage_step
            ].to_hdf(
                f"data/gathered_embs_{root_path.split('uniprot_embs_')[-1].replace('/', '')}_{part_i}_{cli_args.embs_suffix}.h5",
                key="data",
            )
        logger.info(
            "Stored embeddings data into chunks here: data/gathered_embs_%s*_",
            root_path.split("uniprot_embs_")[-1].replace("/", ""),
        )
    else:
        results_df.to_hdf(
            f"data/gathered_embs_{root_path.split('uniprot_embs_')[-1].replace('/', '')}_{cli_args.embs_suffix}.h5",
            key="data",
        )
        logger.info(
            "Stored embeddings data into a single .h5 file here: data/gathered_embs_%s_%s",
            root_path.split("uniprot_embs_")[-1].replace("/", ""),
            cli_args.embs_suffix,
        )
        if cli_args.remove_raw_files_on_end:
            rmtree(root_path)


if __name__ == "__main__":
    main()
