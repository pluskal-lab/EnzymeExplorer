"""This script gathers the detection results into a CSV file"""

import argparse
import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import pandas as pd  # type: ignore
from tqdm import tqdm  # type: ignore

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def parse_args() -> argparse.Namespace:
    """
    This function parses arguments
    :return: current argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="A script to gather screening detections into a CSV file"
    )
    parser.add_argument(
        "--delete-individual-files",
        help="A flag to delete individual files",
        action="store_true",
    )
    parser.add_argument(
        "--screening-results-root",
        help="Root folder with screening results",
        type=str,
        default="trembl_screening/detections_plm",
    )
    parser.add_argument(
        "--output-path",
        help="A file to save the CSV with detections to",
        type=str,
        default=f"trembl_screening/detections_plm/detections_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    screening_results_root = Path(args.screening_results_root)
    assert (
        screening_results_root.exists()
    ), f"Folder {screening_results_root} does not exist"
    ids = []
    predicted_class_2_vals = defaultdict(list)

    processed_files = []
    all_files = [file for file in screening_results_root.glob("*") if file.is_file()]
    for detected_file in tqdm(all_files, desc="Processing detection files"):
        if ".csv" not in detected_file.name:
            with open(detected_file, "r", encoding="utf-8") as file:
                content = json.load(file)
            ids.append(detected_file.name)
            for class_name, prob in content.items():
                predicted_class_2_vals[class_name].append(prob)
            processed_files.append(str(detected_file))

    predicted_class_2_vals.update({"ID": ids})
    df_detections = pd.DataFrame(predicted_class_2_vals)
    if len(df_detections) and "isTPS" in df_detections.columns:
        df_detections = df_detections.sort_values("isTPS", ascending=False)
    df_detections.to_csv(args.output_path, index=False)
    logger.info(
        "Screening results gathered into %s with %d rows",
        args.output_path,
        len(df_detections),
    )

    if args.delete_individual_files:
        for file_to_delete in processed_files:
            os.remove(file_to_delete)
        logger.info("Deleted %d individual files", len(processed_files))
