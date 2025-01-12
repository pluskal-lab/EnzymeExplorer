"""This module contains routines to access data systematically, e.g. to iterate over validation schemas"""
import pickle
from typing import Tuple, Union
import logging
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from indigo import Indigo  # type: ignore
from Bio.PDB import PDBParser, PPBuilder  # type: ignore
logging.getLogger("h5py").setLevel(logging.INFO)
import h5py  # type: ignore # pylint: disable=C0413
from pathlib import Path
import subprocess
import pickle
from typing import Dict, Tuple
import pandas as pd

logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)

triplets_dtype = [
    ("Uniprot ID", h5py.string_dtype()),
    ("Amino acid sequence", h5py.string_dtype()),
    (
        "SMILES_substrate_canonical_no_stereo",
        h5py.string_dtype(),
    ),
    ("SMILES of product (including stereochemistry)", h5py.string_dtype()),
]


def get_fold(
    fold_i: Union[int, str], path: str = "data/tps_folds.h5", split_desc: str = "all"
) -> np.ndarray:
    """
    This function returns selected fold of the specified validation split
    :param fold_i: fold index
    :param path: path to stored splits into folds
    :param split_desc: validation schema name
    :return: numpy array with fold triplets
    """
    with h5py.File(path, "r") as tps_folds_hf:
        training_folds = tps_folds_hf.get(f"{split_desc}")
        fold_triplets = np.array(training_folds.get(f"{fold_i}"))
        return fold_triplets


def get_folds(split_desc: str, path: str = "data/tps_folds_nov2023.h5") -> list[str]:
    """
    This function returns available fold names in the specified split
    :param split_desc: validation schema name
    :param path: path to stored splits into folds
    :return: list of fold names
    """
    with h5py.File(path, "r") as tps_folds_hf:
        return [
            fold_name
            for fold_name in tps_folds_hf.get(split_desc).keys()
            if fold_name != "unsplittable_target_values"
        ]


def get_unsplittable_targets(split_desc: str, path: str = "data/tps_folds.h5") -> set:
    """
    This function returns available fold names in the specified split
    :param split_desc: validation schema name
    :param path: path to stored splits into folds
    :return: set of targets which should not be predicted in the current split
    """
    with h5py.File(path, "r") as tps_folds_hf:
        if "unsplittable_target_values" in tps_folds_hf.get(split_desc):
            return {
                val[0].decode("ascii", "ignore")
                for val in tps_folds_hf.get(split_desc).get(
                    "unsplittable_target_values"
                )
            }
        return set()


def get_str(h5_val: Union[bytes, str]) -> str:
    """
    This function returns a string even if the underlying h5 value is in bytes
    :param h5_val:
    :return:
    """
    return h5_val.decode() if isinstance(h5_val, bytes) else h5_val


def get_train_val_per_fold(
    fold_i: Union[int, str],
    path: str = "data/tps_folds.h5",
    split_desc: str = "all",
    filter_val_terzyme: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function returns train and validation indices for the selected validation fold
    :param fold_i: validation fold name
    :param path: path to stored splits into folds
    :param split_desc: validation schema name
    :param filter_val_terzyme: flag indicating filtering of the original terzyme dataset
    :return: tuple of numpy arrays with training and validation indices
    """
    val_np = get_fold(fold_i, path, split_desc=split_desc)

    if (
        filter_val_terzyme
    ):  # deprecated as we re-train profileHMM on our richer data to get a stronger baseline
        terzyme_whole_df = pd.read_csv("data/terzyme_data_whole.csv")
        uniprot_ids_terzyme = set(terzyme_whole_df["Uniprot ID"].values)
        val_np = np.array(
            [
                element
                for element in val_np
                if get_str(element[0]) not in uniprot_ids_terzyme
            ]
        )

    _train_folds = [
        get_fold(fold_j, path, split_desc)
        for fold_j in get_folds(split_desc, path)
        if fold_j != fold_i
    ]
    train_np = np.concatenate(_train_folds)
    return train_np, val_np


def get_tps_df(
    path_to_file: str,
    path_to_sampled_negatives: str,
    id_col_name: str = "Uniprot ID",
    remove_fragments: bool = True,
    max_seq_len: int = 2000,
) -> pd.DataFrame:
    """
    Function to prepare a pre-processed terpene synthases dataset as a pandas DataFrame.

    :param path_to_file: Path to an xlsx file containing gathered TPS sequences.
    :param path_to_sampled_negatives: Path to a pickle file containing negative sequences sampled from Swiss-Prot.
    :param id_col_name: The name of the column containing Uniprot IDs. Defaults to "Uniprot ID".
    :param remove_fragments: Flag indicating whether to remove sequences that are not full fragments. Defaults to True.
    :param max_seq_len: Maximum allowed length of the sequence to filter out sequences that are too long. Defaults to 2000.

    :return: A pandas DataFrame containing the pre-processed terpene synthases dataset.
    """

    tps_df = pd.read_csv(path_to_file)
    tps_df[id_col_name] = tps_df[id_col_name].map(
        lambda x: x.strip() if isinstance(x, str) else "Negative"
    )
    tps_df.dropna(subset=["Amino acid sequence"], inplace=True)
    if remove_fragments:
        tps_df = tps_df[tps_df["Fragment"].isnull()]

    known_ids = set(tps_df[id_col_name].values)

    with open(path_to_sampled_negatives, "rb") as file:
        id_2_seq = pickle.load(file)

    id_seq_new = [
        (uniprot_id, seq)
        for uniprot_id, seq in id_2_seq.items()
        if uniprot_id not in known_ids
    ]

    tps_df_new_dict = {
        id_col_name: [el[0] for el in id_seq_new],
        "Amino acid sequence": [el[1] for el in id_seq_new],
    }

    tps_df_new = pd.DataFrame(
        {
            column: tps_df_new_dict.get(
                column, ["Unknown" for _ in range(len(id_seq_new))]
            )
            for column in tps_df.columns
        }
    )
    # filtering non-standard amino acids
    tps_df_new = tps_df_new[
        tps_df_new["Amino acid sequence"].map(lambda x: "U" not in x and "O" not in x)
    ]
    tps_df = pd.concat((tps_df, tps_df_new))

    tps_df = tps_df[tps_df["Amino acid sequence"].map(len) <= max_seq_len]

    return tps_df


def get_major_classes_distribution(
    dataframe: pd.DataFrame,
    target_col: str,
    major_classes: list,
) -> pd.Series:
    """
    The function returns proportions of the specified classes in the selected column of the dataframe
    :param dataframe: a dataframe
    :param target_col: the name of a target column in the dataframe df
    :param major_classes: specified target values to compute proportion for
    :return: pandas Series containing the proportions
    """
    counts_all = dataframe[target_col].value_counts()
    counts_all_major = counts_all[counts_all.index.isin(major_classes)]
    counts_all_major.index = counts_all_major.index.map(lambda x: x.copy().pop())
    for class_name in major_classes:
        class_name = class_name.copy().pop()
        if class_name not in counts_all_major.index:
            counts_all_major = pd.concat(
                (counts_all_major, pd.Series([0], index=[class_name]))
            )
    counts_all_major.sort_index(inplace=True)
    counts_all_major /= counts_all_major.sum()
    return counts_all_major


def get_canonical_smiles(smiles: str, without_stereo: bool = True):
    """
    Function to generate the canonical SMILES representation of a given SMILES string.

    :param smiles: The input SMILES string.
    :param without_stereo: A flag indicating whether to remove stereochemistry information. Defaults to True.

    :return: The canonical SMILES string or the original input if it is invalid or in a predefined set of non-processable values.
    """
    if isinstance(smiles, float) or smiles in {"Unknown", "Negative"}:
        return smiles
    indigo = Indigo()
    mol = indigo.loadMolecule(smiles.strip())
    if without_stereo:
        mol.clearCisTrans()
        mol.clearStereocenters()
    return mol.canonicalSmiles()



def compute_mmseqs2_clusters(
    sequences_df: pd.DataFrame,
    id_column: str,
    sequence_column: str,
    output_dir: Path,
    identity_threshold: float = 0.3,
    coverage_threshold: float = 0.8,
) -> Tuple[Dict[str, int], list[list[str]]]:
    """
    Create phylogenetic clusters using MMseqs2 and return id_2_group mapping.
    
    Args:
        sequences_df: DataFrame containing sequences and their IDs
        id_column: Name of the column containing sequence IDs
        sequence_column: Name of the column containing sequences
        output_dir: Directory to store temporary files
        identity_threshold: Sequence identity threshold (default: 0.3)
        coverage_threshold: Alignment coverage threshold (default: 0.8)
    
    Returns:
        Tuple containing:
        - Dictionary mapping sequence IDs to cluster groups
        - List of clusters where each cluster is a list of sequence IDs
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create FASTA file
    fasta_path: Path = output_dir / "sequences.fasta"
    with open(fasta_path, "w") as f:
        for _, row in sequences_df.iterrows():
            f.write(f">{row[id_column]}\n{row[sequence_column]}\n")
    
    # Run easy-cluster with memory-efficient parameters
    cluster_result_path: Path = output_dir / "clusterRes"
    tmp_path: Path = output_dir / "tmp"
    tmp_path.mkdir(exist_ok=True)
    
    subprocess.run([
        "mmseqs", "easy-cluster",
        str(fasta_path),
        str(cluster_result_path),
        str(tmp_path),
        "--min-seq-id", str(identity_threshold),
        "-c", str(coverage_threshold),
        "--cov-mode", "1",  # bidirectional coverage mode
        "--max-seqs", "5000",  # limit number of sequences per query
        "--split-memory-limit", "60G",  # memory limit per split
        "-s", "7.5",  # sensitivity, lower = faster but less sensitive
        "--remove-tmp-files", "1"  # clean up temporary files
    ], check=True)
    
    # Parse clustering results from the TSV file
    tsv_path: Path = output_dir / "clusterRes_cluster.tsv"
    id_2_group: dict[str, int] = {}
    clusters: list[list[str]] = []
    
    with open(tsv_path) as f:
        current_cluster: list[str] = []
        current_representative: str | None = None
        
        for line in f:
            representative, member = line.strip().split("\t")
            
            if current_representative != representative:
                if current_cluster:
                    clusters.append(current_cluster)
                current_cluster = []
                current_representative = representative
            
            current_cluster.append(member)
            id_2_group[member] = len(clusters)
    
    if current_cluster:
        clusters.append(current_cluster)
    
    
    return id_2_group, clusters