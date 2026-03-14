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
from rdkit.Chem import MolToSmiles, rdChemReactions

logger = logging.getLogger(__file__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_canonical_substrates(rxn_smiles: str):
    rxn = rdChemReactions.ReactionFromSmarts(rxn_smiles, useSmiles=True)
    trxn = rdChemReactions.ChemicalReaction(rxn)
    substrates = trxn.GetReactants()
    return {get_canonical_smiles(MolToSmiles(substr)) for substr in substrates}


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


def preprocess_martsdb(martsDB: pd.DataFrame) -> tuple[pd.DataFrame, list[list[str]]]:
    """Preprocess the martsDB TPS data and store the cleaned data to a DataFrame"""

    martsDB = martsDB.copy()
    # read the raw martsDB CSV
    martsDB["Substrate_smiles"] = martsDB.Substrate_smiles.map(
        lambda s: s.replace(";", ".").upper()
    )
    martsDB["Product_smiles"] = martsDB.Product_smiles.map(
        lambda s: s.replace(";", ".").upper()
    )

    # canonical SMILES
    martsDB["SMILES_substrate_canonical_no_stereo"] = martsDB["Substrate_smiles"].map(
        get_canonical_smiles
    )

    martsDB["SMILES_product_canonical_no_stereo"] = martsDB["Product_smiles"].map(
        get_canonical_smiles
    )

    # fixing multi-molecule substrates

    def fix_multi_molecule_substrate(row):
        if row["Type"] == "pt":
            return "precursor substr"
        if (
            row["SMILES_product_canonical_no_stereo"].count("C")
            % row["SMILES_substrate_canonical_no_stereo"].count("C")
            != 0
        ):
            return row["SMILES_substrate_canonical_no_stereo"]

        mol_ctr = row["SMILES_product_canonical_no_stereo"].count("C") // row[
            "SMILES_substrate_canonical_no_stereo"
        ].count("C")
        return ".".join(
            [row["SMILES_substrate_canonical_no_stereo"] for _ in range(mol_ctr)]
        )

    martsDB["SMILES_substrate_canonical_no_stereo"] = martsDB.apply(
        fix_multi_molecule_substrate, axis=1
    )

    # cleaning the sequences
    martsDB = martsDB.loc[
        martsDB["Aminoacid_sequence"].map(lambda x: not isinstance(x, float))
    ]
    martsDB["Aminoacid_sequence"] = martsDB["Aminoacid_sequence"].map(
        lambda x: x.replace("*", "").replace('"', "").replace("'", "")
    )
    martsDB["Aminoacid_sequence"] = martsDB["Aminoacid_sequence"].map(
        lambda x: "".join(x.split())
    )

    aa_seq_2_id = defaultdict(list)
    for _, row in martsDB.iterrows():
        aa_seq = row["Aminoacid_sequence"]
        aa_seq_2_id[aa_seq].append(row["Enzyme_marts_ID"])
    for aa_seq in aa_seq_2_id:
        aa_seq_2_id[aa_seq] = sorted(list(set(aa_seq_2_id[aa_seq])))

    marts_duplicates = list([ids for ids in aa_seq_2_id.values() if len(ids) > 1])
    martsDB["Enzyme_marts_ID"] = martsDB.apply(
        lambda row: list(aa_seq_2_id[row["Aminoacid_sequence"]])[0], axis=1
    )
    assert (
        martsDB["Enzyme_marts_ID"].isna().sum() == 0
    ), "NaN values found in Enzyme_marts_ID after processing"
    martsDB.drop_duplicates(
        subset=["Aminoacid_sequence", "Substrate_marts_ID", "Product_marts_ID"],
        inplace=True,
    )

    # deriving clean kingdom info

    martsDB["Kingdom_detailed"] = martsDB.Kingdom
    for animal_init_category in [
        "Animalia (Coral)",
        "Animalia (Marine Sponge)",
        "Animalia (Mammal)",
        "Animalia (Insect)",
        "Animalia (Mammal, Human)",
        "Animalia (Bird)",
    ]:
        martsDB.loc[martsDB["Kingdom"] == animal_init_category, "Kingdom"] = "Animals"

    for plant_init_category in ["Plantae", "Plantae (Red algae)"]:
        martsDB.loc[martsDB["Kingdom"] == plant_init_category, "Kingdom"] = "Plants"

    for plant_init_category in ["Bacteria", "Cyanobacteria"]:
        martsDB.loc[martsDB["Kingdom"] == plant_init_category, "Kingdom"] = "Bacteria"

    martsDB.loc[martsDB["Kingdom"] == "Amoebozoa", "Kingdom"] = "Protists"

    martsDB["OriginalType"] = martsDB["Type"]
    # Map Squalene Synthase and Phytoene Synthases to tetra and tri TPS types
    martsDB.loc[
        (martsDB["Type"] == "sqs")
        & (martsDB["SMILES_substrate_canonical_no_stereo"].str.count("C") == 40),
        "Type",
    ] = "tetra"
    martsDB.loc[
        (martsDB["Type"] == "sqs")
        & (martsDB["SMILES_substrate_canonical_no_stereo"].str.count("C") == 30),
        "Type",
    ] = "tri"
    martsDB.loc[
        (martsDB["Type"] == "psy")
        & (martsDB["SMILES_substrate_canonical_no_stereo"].str.count("C") == 40),
        "Type",
    ] = "tetra"
    martsDB.loc[
        (martsDB["Type"] == "psy")
        & (martsDB["SMILES_substrate_canonical_no_stereo"].str.count("C") == 30),
        "Type",
    ] = "tri"

    return martsDB, marts_duplicates


def proprocess_negatives(
    swissprot_df: pd.DataFrame,
    martsdb_seqs: list,
    pfam_models_dir: str,
    supfam_models_dir: str,
    go_dag: GODag,
    mmseqs: MMSeqs2Wrapper,
    hmmer: HMMerWrapper,
    filter_by_putative_tpss: bool = True,
) -> pd.DataFrame:
    swissprot_df = swissprot_df.drop_duplicates("Sequence")

    nontps_swissprot = swissprot_df[~swissprot_df.Sequence.isin(martsdb_seqs)]

    if filter_by_putative_tpss:
        
        tps_swissprot_rhea = swissprot_df[
            swissprot_df.Sequence.isin(martsdb_seqs) & swissprot_df["Rhea ID"].notna()
        ]

        tps_swissprot_ec = swissprot_df[
            swissprot_df.Sequence.isin(martsdb_seqs) & swissprot_df["EC number"].notna()
        ]

        tps_swissprot_go = swissprot_df[
            swissprot_df.Sequence.isin(martsdb_seqs)
            & swissprot_df["Gene Ontology IDs"].notna()
        ]
        
        nontps_swissprot = filter_out_putative_tpss(nontps_swissprot, PUTATIVE_TPS_IDS)

        logger.info(
            f"Filtered out putative TPSs. Remaining non-TPS SwissProt size: {len(nontps_swissprot)}"
        )

        nontps_swissprot = filter_by_rhea(tps_swissprot_rhea, nontps_swissprot)

        logger.info(
            f"Filtered by Rhea IDs. Remaining non-TPS SwissProt size: {len(nontps_swissprot)}"
        )

        nontps_swissprot = filter_by_ec(tps_swissprot_ec, nontps_swissprot, TPS_ECS_BASE)

        logger.info(
            f"Filtered by EC numbers. Remaining non-TPS SwissProt size: {len(nontps_swissprot)}"
        )

        nontps_swissprot = filter_by_go(
            tps_swissprot_go, nontps_swissprot, TPS_GO_BLACKLIST, go_dag
        )

        logger.info(
            f"Filtered by GO terms. Remaining non-TPS SwissProt size: {len(nontps_swissprot)}"
        )

    nontps_swissprot = redundancy_reduce(nontps_swissprot, mmseqs=mmseqs)

    logger.info(
        f"95% Sequence Identity Redundancy reduced non-TPS SwissProt size: {len(nontps_swissprot)}"
    )

    if filter_by_putative_tpss:
        nontps_swissprot = filter_by_pfam_supfam(
            nontps_swissprot,
            pfam_models_dir=pfam_models_dir,
            supfam_models_dir=supfam_models_dir,
            hmmer=hmmer,
        )

        logger.info(
            f"Filtered out sequences with Pfam/Supfam hits. Remaining non-TPS SwissProt size: {len(nontps_swissprot)}"
        )

    return nontps_swissprot


def filter_out_putative_tpss(
    nontps_swiss: pd.DataFrame, putative_tps_ids: list
) -> pd.DataFrame:
    return nontps_swiss[~nontps_swiss.Entry.isin(putative_tps_ids)]  # type: ignore


def filter_by_rhea(tps_swiss, nontps_swiss):
    tps_rheas = set()
    tps_swiss["Rhea ID"].apply(
        lambda x: (
            tps_rheas.update(
                [rhea_id.split("RHEA:")[-1] for rhea_id in str(x).split(" ")]
            )
            if x is not None
            else None
        )
    )
    return nontps_swiss[
        nontps_swiss["Rhea ID"].map(
            lambda x: (
                True
                if x is None
                else all(
                    [
                        rhea.split("RHEA:")[-1] not in tps_rheas
                        for rhea in str(x).split(" ")
                    ]
                )
            )
        )
    ]


def filter_by_ec(tps_swiss, nontps_swiss, tps_ecs_base):
    tps_ecs = set().union(tps_ecs_base)
    tps_swiss[~tps_swiss["EC number"].str.contains("-")]["EC number"].apply(
        lambda x: [tps_ecs.add(ec.strip()) for ec in x.split(";")]
    )
    return nontps_swiss[
        nontps_swiss["EC number"].map(
            lambda x: (
                True
                if x is None
                else all([ec.strip() not in tps_ecs for ec in str(x).split(";")])
            )
        )
    ]


def filter_by_go(tps_swiss, nontps_swiss, tps_go_blacklist, go_dag):
    tps_gos = set()
    tps_swiss["Gene Ontology IDs"].apply(
        lambda x: [
            tps_gos.add(go.strip())
            for go in x.split(";")
            if go_dag[go.strip()].namespace == "molecular_function"
        ]
    )

    tps_gos.difference_update(tps_go_blacklist)
    tps_root_gos = set()
    for go_1 in tps_gos:
        root = True
        for go_2 in tps_gos:
            if go_1 == go_2:
                continue
            if go_dag[go_1].has_parent(go_2):
                root = False
                break
        if root:
            tps_root_gos.add(go_1)
    tps_gos = set()
    for root_go in tps_root_gos:
        tps_gos.add(root_go)
        tps_gos.update(go_dag[root_go].get_all_children())
    return nontps_swiss[
        nontps_swiss["Gene Ontology IDs"].map(
            lambda x: (
                True
                if x is None
                else all([go.strip() not in tps_gos for go in str(x).split(";")])
            )
        )
    ]


def filter_by_pfam_supfam(
    nontps_swiss: pd.DataFrame,
    pfam_models_dir: str,
    supfam_models_dir: str,
    hmmer: HMMerWrapper,
) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as tmpdir:
        uuid = str(uuid4())
        input_fasta_path = os.path.join(tmpdir, f"nontps_swiss_{uuid}.fasta")
        with open(input_fasta_path, "w") as fasta_file:
            for _, row in nontps_swiss.iterrows():
                fasta_file.write(f">{row['Entry']}\n{row['Sequence']}\n")

        pfam_db_path = os.path.join(tmpdir, f"pfam_models_{uuid}")
        hmmer.hmm_concat(pfam_models_dir, pfam_db_path)
        hmmer.hmmpress(pfam_db_path)
        pfam_hits_df = hmmer.hmmscan(
            query_fasta=input_fasta_path,
            hmm_path=pfam_db_path,
            output=os.path.join(tmpdir, f"pfam_scan_{uuid}.tbl"),
            bitscore=25,
        )

        supfam_db_path = os.path.join(tmpdir, f"supfam_models_{uuid}")
        hmmer.hmm_concat(supfam_models_dir, supfam_db_path)
        hmmer.hmmpress(supfam_db_path)
        supfam_hits_df = hmmer.hmmscan(
            query_fasta=input_fasta_path,
            hmm_path=supfam_db_path,
            output=os.path.join(tmpdir, f"supfam_scan_{uuid}.tbl"),
            bitscore=25.0,
        )
    swiss_with_pfam_supfam_hits = set(
        pfam_hits_df["query_name"].unique().tolist()
    ) | set(supfam_hits_df["query_name"].unique().tolist())
    return nontps_swiss[~nontps_swiss.Entry.isin(swiss_with_pfam_supfam_hits)]


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


def add_tps_substrates_to_rhea_data_inplace(
    rhea_reaction_smiles: pd.DataFrame, rhea_directions: pd.DataFrame, martsDB: pd.DataFrame
):
    rhea_reaction_smiles["substrate_canonical_smiles_no_stereo"] = rhea_reaction_smiles[
        "reaction_smiles"
    ].map(get_canonical_substrates)
    tps_substrates = set(martsDB.SMILES_substrate_canonical_no_stereo.tolist())
    rhea_reaction_smiles["accepted_tps_substrates"] = (
        rhea_reaction_smiles.substrate_canonical_smiles_no_stereo.map(
            lambda smiles_set: smiles_set.intersection(tps_substrates)
        )
    )
    rhea_ids_to_accepted_tps_substrates = (
        rhea_reaction_smiles[
            rhea_reaction_smiles["accepted_tps_substrates"].map(len) > 0
        ]
        .set_index("rhea_id")["accepted_tps_substrates"]
        .to_dict()
    )
    rhea_directions["accepted_tps_substrates"] = rhea_directions.apply(
        lambda row: rhea_ids_to_accepted_tps_substrates.get(row["RHEA_ID_MASTER"], set()).union(
            rhea_ids_to_accepted_tps_substrates.get(row["RHEA_ID_LR"], set())
        ).union(
            rhea_ids_to_accepted_tps_substrates.get(row["RHEA_ID_RL"], set())
        ).union(
            rhea_ids_to_accepted_tps_substrates.get(row["RHEA_ID_BI"], set())
        ),
        axis=1
    )


def get_substrate_based_hard_negatives(
    nontps_swissprot: pd.DataFrame,
    nontps_swissprot_clusters_df: pd.DataFrame,
    rhea_directions: pd.DataFrame,
) -> tuple[set[str], dict[str, set[str]]]:    
    rhea_master_ids_to_accepting_tps_substrates = rhea_directions[
        rhea_directions["accepted_tps_substrates"].map(len) > 0
    ].set_index("RHEA_ID_MASTER")["accepted_tps_substrates"].to_dict()

    nontps_swissprot = nontps_swissprot[nontps_swissprot["Rhea ID"].notna()]

    negatives_to_accepted_tps_substrates = nontps_swissprot[["Entry", "Rhea ID"]].set_index("Entry")["Rhea ID"].map(
            lambda x: (
                set()
                if x is None
                else set(
                    substrate
                    for rhea_id in str(x).split(" ")
                    if int(rhea_id.split("RHEA:")[-1]) in rhea_master_ids_to_accepting_tps_substrates
                    for substrate in rhea_master_ids_to_accepting_tps_substrates[int(rhea_id.split("RHEA:")[-1])]
                )
            )
        ).to_dict()
    
    negatives_to_accepted_tps_substrates = {neg: substrates for neg, substrates in negatives_to_accepted_tps_substrates.items() if len(substrates) > 0}

    hard_negative_cluster_reps = set(
        nontps_swissprot_clusters_df[
            nontps_swissprot_clusters_df["Member"].isin(
                negatives_to_accepted_tps_substrates
            )
        ]["Representative"].tolist())
    return hard_negative_cluster_reps, negatives_to_accepted_tps_substrates


def _get_sequence_based_hard_negative_clusters(
    negatives_cluster_representatives: pd.DataFrame,
    martsDB: pd.DataFrame,
    mmseqs: MMSeqs2Wrapper,
    e_value: float = 0.1,
) -> set[str]:

    with tempfile.TemporaryDirectory() as tmpdir:
        uuid = str(uuid4())
        input_fasta_path = os.path.join(tmpdir, f"negatives_rep_seq_{uuid}.fasta")
        with open(input_fasta_path, "w") as fasta_file:
            for _, row in negatives_cluster_representatives.iterrows():
                fasta_file.write(f">{row['Representative']}\n{row['Sequence']}\n")
        positives_fasta_path = os.path.join(tmpdir, f"positives_{uuid}.fasta")
        with open(positives_fasta_path, "w") as fasta_file:
            for _, row in martsDB.iterrows():
                fasta_file.write(
                    f">{row['Enzyme_marts_ID']}\n{row['Aminoacid_sequence']}\n"
                )
        output_prefix = os.path.join(tmpdir, f"neg_rep_vs_pos_{uuid}.m8")
        tmpdir = os.path.join(tmpdir, f"tmp_{uuid}")
        search_results_df = mmseqs.easy_search(
            query_fasta=input_fasta_path,
            target_fasta=positives_fasta_path,
            output=output_prefix,
            tmp=tmpdir,
            e_value=e_value,
        )
    return set(search_results_df["query"].unique().tolist())


def _get_sequence_based_hard_negative_clusters_sensitive(
    negatives_cluster_representatives: pd.DataFrame,
    martsDB: pd.DataFrame,
    mmseqs: MMSeqs2Wrapper,
    num_iterations: int = 2,
    e_value: float = 0.1,
    seq_id: float = 0.2,
) -> set[str]:

    with tempfile.TemporaryDirectory() as tmpdir:
        uuid = str(uuid4())
        target_fasta_path = os.path.join(tmpdir, f"negatives_rep_seq_{uuid}.fasta")
        with open(target_fasta_path, "w") as fasta_file:
            for _, row in negatives_cluster_representatives.iterrows():
                fasta_file.write(f">{row['Representative']}\n{row['Sequence']}\n")
        positives_fasta_path = os.path.join(tmpdir, f"positives_{uuid}.fasta")
        with open(positives_fasta_path, "w") as fasta_file:
            for _, row in martsDB.iterrows():
                fasta_file.write(
                    f">{row['Enzyme_marts_ID']}\n{row['Aminoacid_sequence']}\n"
                )
        output_prefix = os.path.join(tmpdir, f"neg_rep_vs_pos_{uuid}.m8")
        tmpdir = os.path.join(tmpdir, f"tmp_{uuid}")
        search_results_df = mmseqs.easy_search(
            query_fasta=positives_fasta_path,
            target_fasta=target_fasta_path,
            output=output_prefix,
            tmp=tmpdir,
            e_value=e_value,
            num_iterations=num_iterations,
            seq_id=seq_id,
        )
    return set(search_results_df["target"].unique().tolist())


def get_sequence_based_hard_negative_cluster_ids(
    negatives_cluster_representatives: pd.DataFrame,
    martsDB: pd.DataFrame,
    mmseqs: MMSeqs2Wrapper,
) -> set[str]:
    simple_hard_negatives = _get_sequence_based_hard_negative_clusters(
        negatives_cluster_representatives,
        martsDB.drop_duplicates(subset=["Enzyme_marts_ID"]),
        mmseqs,
    )

    typed_hard_negatives = set()
    for t in martsDB.OriginalType.unique():
        t_martsDB = martsDB[martsDB.OriginalType == t].drop_duplicates(
            subset=["Enzyme_marts_ID"]
        )
        typed_hard_negatives.update(
            _get_sequence_based_hard_negative_clusters_sensitive(
                negatives_cluster_representatives, t_martsDB, mmseqs
            )
        )

    kingdom_hard_negatives = set()
    for kingdom in martsDB.Kingdom.unique():
        kingdom_martsDB = martsDB[martsDB.Kingdom == kingdom].drop_duplicates(
            subset=["Enzyme_marts_ID"]
        )
        kingdom_hard_negatives.update(
            _get_sequence_based_hard_negative_clusters_sensitive(
                negatives_cluster_representatives, kingdom_martsDB, mmseqs
            )
        )

    return simple_hard_negatives.union(typed_hard_negatives).union(
        kingdom_hard_negatives
    )


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
