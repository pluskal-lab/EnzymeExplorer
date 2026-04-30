import logging
from uuid import uuid4
import pandas as pd
from enzymeexplorer.src.data_preparation.mmseqs2_wrapper import MMSeqs2Wrapper
from enzymeexplorer.src.utils.data import get_canonical_smiles
import os
import tempfile
import warnings
from enzymeexplorer.src.data_preparation.constants import BLACKLISTED_RHEA_MASTER_IDS
from rdkit.Chem import MolToSmiles, rdChemReactions # type: ignore
from collections import defaultdict
import numpy as np
from scipy.spatial.distance import jensenshannon
from sklearn.model_selection import StratifiedGroupKFold
from tqdm.auto import tqdm
tqdm.pandas()


logger = logging.getLogger(__file__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def get_non_tps_rhea_ids_with_tps_substrates(marts_rhea_ids: set[int], rhea_id_to_master_id: dict[int, int], rhea_reaction_smiles: pd.DataFrame, martsDB: pd.DataFrame) -> set[int]:
    rhea_reaction_smiles = add_tps_substrates_to_rhea_data(rhea_reaction_smiles, martsDB)
    rhea_reaction_smiles["MASTER_ID"] = rhea_reaction_smiles["rhea_id"].map(rhea_id_to_master_id)
    marts_rhea_master_ids = set([rhea_id_to_master_id[rhea_id] for rhea_id in marts_rhea_ids])
    blacklist_rhea_master_ids = marts_rhea_master_ids.union(set(BLACKLISTED_RHEA_MASTER_IDS))
    non_tps_rhea_ids_with_tps_substrates = set(
        rhea_reaction_smiles[
            ~(rhea_reaction_smiles["MASTER_ID"].isin(blacklist_rhea_master_ids))
            & (rhea_reaction_smiles["accepted_tps_substrates"].apply(lambda x: len(x) > 0))
        ]["MASTER_ID"].tolist()
    )
    return non_tps_rhea_ids_with_tps_substrates

def get_canonical_substrates(rxn_smiles: str):
    rxn = rdChemReactions.ReactionFromSmarts(rxn_smiles, useSmiles=True)
    trxn = rdChemReactions.ChemicalReaction(rxn)
    substrates = trxn.GetReactants()
    return {get_canonical_smiles(MolToSmiles(substr)) for substr in substrates}


def add_tps_substrates_to_rhea_data(
    rhea_reaction_smiles: pd.DataFrame,
    martsDB: pd.DataFrame,
):
    rhea_reaction_smiles["substrate_canonical_smiles_no_stereo"] = rhea_reaction_smiles[
        "reaction_smiles"
    ].progress_apply(get_canonical_substrates)
    tps_substrates = set(martsDB.SMILES_substrate_canonical_no_stereo.tolist())
    rhea_reaction_smiles["accepted_tps_substrates"] = (
        rhea_reaction_smiles.substrate_canonical_smiles_no_stereo.map(
            lambda smiles_set: smiles_set.intersection(tps_substrates)
        )
    )
    return rhea_reaction_smiles
    

def get_rhea_id_to_master_id_mappings(reaction_directions: pd.DataFrame) -> dict[int, int]:
    rhea_id_to_master_id = {}
    for _, row in reaction_directions.iterrows():
        rhea_id_to_master_id[row["RHEA_ID_LR"]] = row["RHEA_ID_MASTER"]
        rhea_id_to_master_id[row["RHEA_ID_RL"]] = row["RHEA_ID_MASTER"]
        rhea_id_to_master_id[row["RHEA_ID_BI"]] = row["RHEA_ID_MASTER"]
    return rhea_id_to_master_id

def redundancy_reduce(
    nontps_swissprot: pd.DataFrame, mmseqs: MMSeqs2Wrapper
) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as tmpdir:
        uuid = str(uuid4())
        input_fasta_path = os.path.join(tmpdir, f"sequences_{uuid}.fasta")
        with open(input_fasta_path, "w") as fasta_file:
            for _, row in nontps_swissprot.iterrows():
                fasta_file.write(f">{row['Entry']}\n{row['Sequence']}\n")
        output_prefix = os.path.join(tmpdir, f"clusters_{uuid}")
        tmpdir = os.path.join(tmpdir, f"tmp_{uuid}")
        _, representatives_df = mmseqs.easy_cluster(
            input_fasta=input_fasta_path,
            output=output_prefix,
            tmp=tmpdir,
            min_seq_id=0.95,
            coverage=0.8,
        )
    reduced_nontps_swissprot = nontps_swissprot[
        nontps_swissprot["Entry"].isin(representatives_df["Representative"].tolist())
    ]
    return reduced_nontps_swissprot


def cluster_dataset(
    dataset: pd.DataFrame,
    mmseqs: MMSeqs2Wrapper,
    id_column: str = "Enzyme_marts_ID",
    seq_column: str = "Aminoacid_sequence",
    min_seq_id: float = 0.5,
    coverage: float = 0.6,
    coverage_mode: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster the dataset sequences using MMseqs2 and return cluster assignments and representatives DataFrames"""

    with tempfile.TemporaryDirectory() as tmpdir:
        uuid = str(uuid4())
        input_fasta_path = os.path.join(tmpdir, f"dataset_{uuid}.fasta")
        dataset = dataset.drop_duplicates(subset=[id_column, seq_column])
        with open(input_fasta_path, "w") as fasta_file:
            for _, row in dataset.iterrows():
                fasta_file.write(f">{row[id_column]}\n{row[seq_column]}\n")
        output_prefix = os.path.join(tmpdir, f"dataset_clusters_{uuid}")
        tmpdir = os.path.join(tmpdir, f"tmp_{uuid}")
        clusters_df, representatives_df = mmseqs.easy_cluster(
            input_fasta=input_fasta_path,
            output=output_prefix,
            tmp=tmpdir,
            min_seq_id=min_seq_id,
            coverage=coverage,
            coverage_mode=coverage_mode,
        )

    return clusters_df, representatives_df


def get_is_splittable(
    dataset_with_clusters: pd.DataFrame,
    id_column: str,
    cluster_id_column: str,
    target_col: str,
    classes: list[str],
    n_folds: int = 5,
) -> tuple[bool, set[str]]:
    group_target_2_seq_ids: defaultdict[tuple, set] = defaultdict(set)

    target_2_partitions = defaultdict(set)
    for _, row in dataset_with_clusters.iterrows():
        cluster = row[cluster_id_column]
        target_value = row[target_col]
        group_target_2_seq_ids[(cluster, target_value)].add(row[id_column])
        target_2_partitions[target_value].add(cluster)
    unsplittable_target_values = set()
    target_val_2_total_count = dataset_with_clusters.drop_duplicates(
        [id_column, target_col]
    )[target_col].value_counts()

    group_target_2_count = {
        (cluster, target_value): len(seq_ids)
        for (cluster, target_value), seq_ids in group_target_2_seq_ids.items()
    }

    for (_, target_val), count_in_single_cc in group_target_2_count.items():
        total_target_val_occurrences = target_val_2_total_count.loc[target_val]
        if (
            count_in_single_cc / total_target_val_occurrences > 0.7
            or len(target_2_partitions[target_val]) < n_folds
        ):
            unsplittable_target_values.add(target_val)
    return (
        len(unsplittable_target_values.intersection(set(classes))) == 0,
        unsplittable_target_values,
    )


def get_class_distribution(
    dataset: pd.DataFrame,
    id_column: str,
    target_col: str,
    classes: list[str],
) -> pd.Series:
    class_dist = dataset[target_col].value_counts()
    class_dist = class_dist[class_dist.index.isin(classes)]
    class_dist = class_dist / len(
        dataset[dataset[target_col].isin(classes)][id_column].unique()
    )
    class_dist.sort_index(inplace=True)
    return class_dist


def pick_best_fold(
    dataset: pd.DataFrame,
    id_column: str,
    cluster_id_column: str,
    target_col: str,
    classes: list[str],
    n_folds: int,
    class_dist: pd.Series,
) -> list[tuple[np.ndarray, np.ndarray]]:
    min_max_jensenshannon_val = float("inf")
    final_folds = []
    invalid_fold = set()
    for random_state in tqdm(range(2000), desc="Optimizing K-Fold splits"):
        _t = []
        kfold = StratifiedGroupKFold(
            n_splits=n_folds, shuffle=True, random_state=random_state
        )

        folds = list(
            kfold.split(
                dataset,
                dataset[f"{target_col}_sorted"],
                dataset[cluster_id_column],
            )
        )
        for _, val_idx in folds:
            val_df = dataset.iloc[val_idx].copy()
            fold_class_dist = get_class_distribution(
                val_df.explode(target_col),
                id_column=id_column,
                target_col=target_col,
                classes=classes,
            )
            invalid_fold = set(classes) - set(
                val_df.explode(target_col)[target_col].unique()
            )
            if len(invalid_fold) > 0:
                break
            _t.append(jensenshannon(class_dist, fold_class_dist))
        if len(invalid_fold) > 0:
            continue
        mean_jensenshannon = np.mean(_t)
        if mean_jensenshannon < min_max_jensenshannon_val:
            min_max_jensenshannon_val = mean_jensenshannon
            final_folds = folds
    return final_folds


def get_stratified_group_kfold_splits(
    dataset_with_clusters: pd.DataFrame,
    id_column: str,
    cluster_id_column: str,
    optimize_distribution: bool,
    target_col: str,
    classes: list[str],
    n_folds: int = 5,
) -> list[set[str]]:
    """Generate stratified group k-fold splits based on MMseqs2 clusters.

    Args:
        dataset_with_clusters (pd.DataFrame): The dataset containing sequences and target labels.
        id_column (str): The name of the column in dataset containing the IDs.
        cluster_id_column (str): The name of the column in clusters containing the cluster IDs.
        optimize_distribution (bool): Whether to optimize the distribution of target labels across folds.
        target_col (str): The name of the column in dataset containing the target labels.
        classes (list[str]): List of classes to consider for stratification.
        n_folds (int, optional): Number of folds. Defaults to 5.
    Returns:
        list[list[str]]: A list of lists containing IDs for each fold.
    """

    dataset = (
        dataset_with_clusters.groupby([id_column, cluster_id_column])[[target_col]]
        .agg(set)
        .reset_index()
    )
    dataset[f"{target_col}_sorted"] = dataset[target_col].map(
        lambda targets: str(sorted(targets))
    )

    if optimize_distribution:
        class_dist = get_class_distribution(
            dataset_with_clusters, id_column, target_col, classes
        )

        warnings.filterwarnings(action="ignore", category=UserWarning)
        folds = pick_best_fold(
            dataset,
            id_column,
            cluster_id_column,
            target_col,
            classes,
            n_folds,
            class_dist,
        )
        warnings.resetwarnings()

        return [set(dataset[id_column].values[val_idx]) for _, val_idx in folds]
    else:
        kfold = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
        folds = list(
            kfold.split(
                dataset,
                dataset[f"{target_col}_sorted"],
                dataset[cluster_id_column],
            )
        )
        return [set(dataset[id_column].values[val_idx]) for _, val_idx in folds]

