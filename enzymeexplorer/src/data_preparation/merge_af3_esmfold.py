import pandas as pd
from tqdm.auto import tqdm
import argparse
import json
from shutil import copy
import os
import logging
from enzymeexplorer.src.data_preparation.utils import (
    get_residue_names_and_plddt_scores,
    calculate_metrics,
    calculate_metric_diff_thresholds,
)
from enzymeexplorer.src.data_preparation.constants import METRICS_2_FUNC

logger = logging.getLogger(__file__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Merge AF3 and ESMFold predictions based on PLDDT scores."
    )
    parser.add_argument(
        "--martsDB_csv",
        type=str,
        help="Path to martsDB CSV file containing sequence information.",
        default="../../EnzymeExplorer/data/martsDB_reactions_2026_02_22.csv",
    )
    parser.add_argument(
        "--af3_dir",
        type=str,
        help="Directory containing AF3 predictions.",
        default="martsDB_af3",
    )
    parser.add_argument(
        "--esmfold_dir",
        type=str,
        help="Directory containing ESMFold predictions.",
        default="martsDB_esmfold",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save merged predictions.",
        default="martsDB_merged",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    os.makedirs(args.output_dir, exist_ok=True)

    martsDB = pd.read_csv(args.martsDB_csv)
    marts_ids = martsDB["Enzyme_marts_ID"].unique().tolist()

    logger.info(f"Processing PLDDTs of {len(marts_ids)} marts IDs.")

    af3_residue_names, af3_plddt_scores = get_residue_names_and_plddt_scores(
        marts_ids, args.af3_dir
    )
    esmfold_residue_names, esmfold_plddt_scores = get_residue_names_and_plddt_scores(
        marts_ids, args.esmfold_dir
    )

    unmatched_marts_ids = []
    for marts_id in tqdm(marts_ids):
        if af3_residue_names[marts_id] != esmfold_residue_names[marts_id]:
            unmatched_marts_ids.append(marts_id)
    assert len(unmatched_marts_ids) == 0, f"Unmatched marts IDs: {unmatched_marts_ids}"

    logger.info(
        f"Calculating metrics and determining winners based on: {', '.join(list(METRICS_2_FUNC.keys()))}"
    )

    metrics = calculate_metrics(marts_ids, af3_plddt_scores, esmfold_plddt_scores)

    thresholds = calculate_metric_diff_thresholds(metrics, args.output_dir, percentile=90)

    logger.info(
        f"Determined metric difference thresholds for ESMFold to be considered better than AF3: {thresholds}. Histograms stored under {args.output_dir}"
    )

    esm_winners_combined = [
        marts_id
        for marts_id in marts_ids
        if sum(
            [
                metrics[marts_id][metric]["ESMFold"] - metrics[marts_id][metric]["AF3"]
                >= thresholds[metric]
                for metric in thresholds.keys()
            ]
        ) >= len(thresholds) / 2
    ]

    logger.info(
        f"For {len(esm_winners_combined)} structures, ESMFold will be preferred."
    )

    for marts_id in marts_ids:
        if marts_id in esm_winners_combined:
            copy(
                f"{args.esmfold_dir}/{marts_id}.pdb",
                f"{args.output_dir}/{marts_id}.pdb",
            )
        else:
            copy(
                f"{args.af3_dir}/{marts_id}.pdb",
                f"{args.output_dir}/{marts_id}.pdb",
            )

    marts_2_str_type: dict[str, dict] = {
        marts_id: {"source": "ESMFold" if marts_id in esm_winners_combined else "AF3"}
        for marts_id in marts_ids
    }

    for marts_id in marts_ids:
        for metric in metrics[marts_id]:
            marts_2_str_type[marts_id][metric] = metrics[marts_id][metric]

    with open(f"{args.output_dir}/_marts_2_str_type.json", "w") as f:
        json.dump(marts_2_str_type, f, indent=4)


if __name__ == "__main__":
    main()
