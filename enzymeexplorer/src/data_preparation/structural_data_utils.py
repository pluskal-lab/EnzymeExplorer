import logging
from pathlib import Path
from uuid import uuid4
import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon
from sklearn.model_selection import StratifiedGroupKFold
from enzymeexplorer.src.data_preparation.hmmer_wrapper import HMMerWrapper
from enzymeexplorer.src.data_preparation.constants import (
    PUTATIVE_TPS_IDS,
    PUTATIVE_TPS_IDS,
    TPS_ECS_BASE,
    TPS_ECS_BASE,
    TPS_GO_BLACKLIST,
    METRICS_2_FUNC,
    MAJOR_CLASSES,
)
from enzymeexplorer.src.utils.data import get_canonical_smiles
from enzymeexplorer.src.data_preparation.mmseqs2_wrapper import MMSeqs2Wrapper
import os
import tempfile
from collections import defaultdict
from goatools.obo_parser import GODag
from tqdm.auto import tqdm
import warnings
import requests
from Bio.PDB import PDBParser
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

logger = logging.getLogger(__file__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def download_af_structure(
    uniprot_id: str,
    structures_root: str,
    max_fails_count: int = 3,
    fails_count: int = 0,
) -> bool:
    save_name = uniprot_id
    try:
        if Path(f"{structures_root}/{save_name}.pdb").exists():
            return True
        URL = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v6.pdb"
        response = requests.get(URL)
        if response.status_code != 200:
            return False
        with open(Path(structures_root) / f"{save_name}.pdb", "wb") as file:
            file.write(response.content)
            return True
    except Exception as e:
        if fails_count < max_fails_count:
            return download_af_structure(
                uniprot_id, structures_root, max_fails_count, fails_count + 1
            )
        else:
            return False



def get_residue_names_and_plddt_scores_from_structure(
    structure,
) -> tuple[list[str], np.ndarray]:
    residue_names = []
    plddt_scores = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() == "UNK":
                    continue  # Skip unknown residues
                for atom in residue:
                    if atom.name == "CA":  # Only get the alpha carbon pLDDT
                        residue_name = f"{residue.get_resname()}_{residue.get_id()[1]}"  # Format: RES_123
                        residue_names.append(residue_name)
                        plddt_scores.append(atom.bfactor)
    return residue_names, np.array(plddt_scores)


def get_residue_names_and_plddt_scores(
    marts_ids: list[str], pdbs_root_dir: str
) -> tuple[defaultdict[str, list[str]], dict[str, np.ndarray]]:
    parser = PDBParser(QUIET=True)
    residue_names = defaultdict(list)
    plddt_scores: dict[str, np.ndarray] = {}
    for marts_id in tqdm(marts_ids, desc=f"Processing PLDDTs from {pdbs_root_dir}"):
        structure = parser.get_structure("structure", f"{pdbs_root_dir}/{marts_id}.pdb")
        residue_names[marts_id], plddt_scores[marts_id] = (
            get_residue_names_and_plddt_scores_from_structure(structure)
        )
    return residue_names, plddt_scores


def confidence_segment_lengths(
    plddt: np.ndarray,
    threshold: float = 70.0,
    min_len: int = 1,
) -> np.ndarray:
    lengths: List[int] = []
    spans: List[Tuple[int, int]] = []

    in_seg = False
    start: Optional[int] = None

    for i, v in enumerate(plddt):
        if v >= threshold:
            if not in_seg:
                in_seg = True
                start = i
        else:
            if in_seg:
                end = i
                seg_len = end - start  # type: ignore[arg-type]
                if seg_len >= min_len:
                    lengths.append(seg_len)
                    spans.append((start, end))  # type: ignore[arg-type]
                in_seg = False
                start = None

    # close trailing segment
    if in_seg and start is not None:
        end = len(plddt)
        seg_len = end - start
        if seg_len >= min_len:
            lengths.append(seg_len)
            spans.append((start, end))

    return np.array(lengths)


def calculate_metrics(
    marts_ids: list[str],
    af3_plddt_scores: dict[str, np.ndarray],
    esmfold_plddt_scores: dict[str, np.ndarray],
) -> dict[str, dict[str, dict[str, float]]]:
    return {
        marts_id: {
            metric: {
                "AF3": func(af3_plddt_scores[marts_id]),
                "ESMFold": func(esmfold_plddt_scores[marts_id]),
            }
            for metric, func in METRICS_2_FUNC.items()
        }
        for marts_id in marts_ids
    }


def calculate_metric_diff_thresholds(
    metrics: dict[str, dict[str, dict[str, float]]],
    output_dir: str,
    percentile: float = 95,
) -> dict[str, float]:
    metric_differences = {
        metric: np.array(
            [
                metrics[marts_id][metric]["ESMFold"] - metrics[marts_id][metric]["AF3"]
                for marts_id in metrics.keys()
            ]
        )
        for metric in METRICS_2_FUNC.keys()
    }
    thresholds = {
        metric: max(np.percentile(differences, percentile), 0.0)
        for metric, differences in metric_differences.items()
    }

    fig, axes = plt.subplots(
        1, len(METRICS_2_FUNC), figsize=(len(METRICS_2_FUNC) * 4, 4)
    )
    for ax, (metric_name, differences) in zip(axes, metric_differences.items()):  # type: ignore
        ax.hist(
            differences,
            bins=50,
        )
        ax.axvline(
            thresholds[metric_name], color="red", linestyle="dashed", linewidth=1
        )
        ax.set_title(metric_name)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/_metric_differences_histograms.png")
    plt.close()
    return thresholds  # type: ignore
