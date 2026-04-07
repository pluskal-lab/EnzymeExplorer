import logging
from uuid import uuid4
import pandas as pd
from enzymeexplorer.src.data_preparation.hmmer_wrapper import HMMerWrapper
from enzymeexplorer.src.data_preparation.constants import (
    PUTATIVE_TPS_IDS,
    PUTATIVE_TPS_IDS,
    TPS_ECS_TO_SUBSTRATES_BASE,
    TPS_GO_BLACKLIST,
)
from enzymeexplorer.src.data_preparation.mmseqs2_wrapper import MMSeqs2Wrapper
from enzymeexplorer.src.data_preparation.common_utils import (
    redundancy_reduce,
    cluster_dataset,
    get_stratified_group_kfold_splits,
    get_rhea_id_to_master_id_mappings,
    add_tps_substrates_to_rhea_data,
)
from tqdm.auto import tqdm
import os
import tempfile
from collections import defaultdict
from goatools.obo_parser import GODag
import numpy as np

logger = logging.getLogger(__file__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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


def filter_by_ec(tps_swiss, nontps_swiss, tps_ecs_to_substrates_base):
    tps_ecs = set().union(tps_ecs_to_substrates_base.keys())
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

        nontps_swissprot = filter_by_ec(
            tps_swissprot_ec, nontps_swissprot, TPS_ECS_TO_SUBSTRATES_BASE
        )

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


def get_substrate_based_hard_negatives(
    nontps_swissprot: pd.DataFrame,
    rhea_to_swissprot: pd.DataFrame,
    nontps_swissprot_clusters_df: pd.DataFrame,
    rhea_reaction_smiles: pd.DataFrame,
    rhea_id_to_master_id: dict[int, int],
) -> tuple[set[str], dict[str, set[str]]]:
    rhea_ids_to_tps_substrates = (
        rhea_reaction_smiles[
            rhea_reaction_smiles["accepted_tps_substrates"].map(len) > 0
        ]
        .set_index("rhea_id")["accepted_tps_substrates"]
        .to_dict()
    )

    rhea_master_ids_with_accepted_tps_substrates = set(
        rhea_id_to_master_id[rhea_id] for rhea_id in rhea_ids_to_tps_substrates.keys()
    )

    nontps_swissprot_ids = nontps_swissprot[
        nontps_swissprot.Entry.isin(set(rhea_to_swissprot.ID.to_list()))
    ]["Entry"].unique()

    negatives_to_accepted_tps_substrates = defaultdict(set)
    rhea_to_swissprot_grouped = (
        rhea_to_swissprot[
            rhea_to_swissprot["ID"].isin(nontps_swissprot_ids)
            & rhea_to_swissprot["MASTER_ID"].isin(
                rhea_master_ids_with_accepted_tps_substrates
            )
        ]
        .groupby(["ID", "MASTER_ID"])[["RHEA_ID", "DIRECTION"]]
        .agg(set)
        .reset_index()
    )
    for _, row in tqdm(
        rhea_to_swissprot_grouped.iterrows(),
        total=len(rhea_to_swissprot_grouped),
        desc="Mapping non-TPS SwissProt IDs to accepted TPS substrates based on Rhea associations",
    ):
        uniprot_id = row["ID"]
        directions = row["DIRECTION"]
        if any(direction in ["LR", "RL"] for direction in directions):
            negatives_to_accepted_tps_substrates[uniprot_id].update(
                set.union(
                    *[
                        rhea_ids_to_tps_substrates.get(rhea_id, set())
                        for rhea_id in row["RHEA_ID"]
                    ]
                )
            )
        else:
            negatives_to_accepted_tps_substrates[uniprot_id].update(
                set.union(
                    *[
                        rhea_ids_to_tps_substrates.get(rhea_id + 1, set())
                        for rhea_id in row["RHEA_ID"]
                    ]
                )
            )
            negatives_to_accepted_tps_substrates[uniprot_id].update(
                set.union(
                    *[
                        rhea_ids_to_tps_substrates.get(rhea_id + 2, set())
                        for rhea_id in row["RHEA_ID"]
                    ]
                )
            )

    negatives_to_accepted_tps_substrates = {
        neg: substrates
        for neg, substrates in negatives_to_accepted_tps_substrates.items()
        if len(substrates) > 0
    }

    hard_negative_cluster_reps = set(
        nontps_swissprot_clusters_df[
            nontps_swissprot_clusters_df["Member"].isin(
                negatives_to_accepted_tps_substrates
            )
        ]["Representative"].tolist()
    )
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


def randomised_negative_sampling(
    nontps_swissprot: pd.DataFrame, number_of_negatives: int, n_folds: int
) -> tuple[set[str], list[set[str]], dict[str, set[str]]]:
    if number_of_negatives == -1:
        negative_ids = nontps_swissprot.Entry.unique()
    else:
        negative_ids = np.random.choice(
            nontps_swissprot.Entry.unique().tolist(),
            size=number_of_negatives,
            replace=False,
        )
    negatives_to_accepted_tps_substrates = {}
    negatives_folds = [set(fold) for fold in np.array_split(negative_ids, n_folds)]
    negative_ids = set(negative_ids)
    return negative_ids, negatives_folds, negatives_to_accepted_tps_substrates


def mmseqs_based_negative_sampling(
    nontps_swissprot: pd.DataFrame,
    martsDB: pd.DataFrame,
    mmseqs: MMSeqs2Wrapper,
    n_folds: int,
    number_of_negatives: int,
    min_seq_id: float,
    coverage: float,
    rhea_to_swissprot: pd.DataFrame,
    rhea_reaction_smiles: pd.DataFrame,
    rhea_directions: pd.DataFrame,
) -> tuple[set[str], list[set[str]], dict[str, set[str]]]:
    nontps_swissprot_clusters_df, nontps_swissprot_representatives_df = cluster_dataset(
        nontps_swissprot,
        id_column="Entry",
        seq_column="Sequence",
        mmseqs=mmseqs,
        min_seq_id=min_seq_id,
        coverage=coverage,
    )

    logger.info(
        f"Clustered non-TPS SwissProt sequences into {nontps_swissprot_clusters_df['Representative'].nunique()} clusters"
    )

    rhea_reaction_smiles = add_tps_substrates_to_rhea_data(
        rhea_reaction_smiles, martsDB
    )

    rhea_id_to_master_id = get_rhea_id_to_master_id_mappings(rhea_directions)

    substrate_hard_negatives, negatives_to_accepted_tps_substrates = (
        get_substrate_based_hard_negatives(
            nontps_swissprot,
            rhea_to_swissprot,
            nontps_swissprot_clusters_df,
            rhea_reaction_smiles,
            rhea_id_to_master_id=rhea_id_to_master_id,
        )
    )

    logger.info(
        f"Identified {len(negatives_to_accepted_tps_substrates)} substrate-based hard negatives with {len(substrate_hard_negatives)} unique cluster representatives."
    )

    hard_negative_cluster_ids = get_sequence_based_hard_negative_cluster_ids(
        nontps_swissprot_representatives_df, martsDB, mmseqs
    )

    logger.info(
        f"Identified {len(hard_negative_cluster_ids)} sequence-based hard negative cluster representatives."
    )

    hard_negative_cluster_ids.update(substrate_hard_negatives)

    hard_negative_clusters = nontps_swissprot_clusters_df[
        nontps_swissprot_clusters_df["Representative"].isin(hard_negative_cluster_ids)
    ].copy()
    hard_negative_clusters["Type"] = "Hard"

    logger.info(f"Identified {len(hard_negative_clusters)} hard negative samples.")

    easy_negative_cluster_ids = nontps_swissprot_clusters_df[
        ~nontps_swissprot_clusters_df["Representative"].isin(hard_negative_cluster_ids)
    ]["Representative"].unique()
    
    if number_of_negatives != -1:
        easy_negative_cluster_ids = np.random.choice(
            easy_negative_cluster_ids,
            size=number_of_negatives - len(hard_negative_clusters),
            replace=False,
        )

        logger.info(f"Chosen {len(easy_negative_cluster_ids)} easy negative clusters.")

        easy_negative_clusters = nontps_swissprot_clusters_df[
            nontps_swissprot_clusters_df["Representative"].isin(easy_negative_cluster_ids)
        ].copy()
        easy_negative_clusters["Type"] = "Easy"
        easy_negative_clusters.drop_duplicates("Representative", inplace=True)
    else:
        logger.info(f"Selected all {len(easy_negative_cluster_ids)} easy negative clusters.")
        easy_negative_clusters = nontps_swissprot_clusters_df[
            nontps_swissprot_clusters_df["Representative"].isin(easy_negative_cluster_ids)].copy()
        easy_negative_clusters["Type"] = "Easy"

    logger.info(f"Selected {len(easy_negative_clusters)} easy negative samples.")

    negative_clusters = pd.concat([hard_negative_clusters, easy_negative_clusters])
    negatives_folds = get_stratified_group_kfold_splits(
        negative_clusters,
        id_column="Member",
        cluster_id_column="Representative",
        optimize_distribution=False,
        target_col="Type",
        classes=["Easy", "Hard"],
        n_folds=n_folds,
    )

    logger.info("Generated stratified group K-Fold splits for negatives.")

    negative_ids = set(negative_clusters.Member.unique())
    return negative_ids, negatives_folds, negatives_to_accepted_tps_substrates


def prepare_negatives_set(
    nontps_swissprot: pd.DataFrame,
    negative_ids: set,
    negatives_to_accepted_tps_substrates: dict,
    negatives_folds: list[set[str]],
) -> pd.DataFrame:
    negatives_data = nontps_swissprot[nontps_swissprot.Entry.isin(negative_ids)][
        ["Entry", "Sequence"]
    ].rename(columns={"Entry": "ID", "Sequence": "Aminoacid_sequence"})

    negatives_data["SMILES_substrate_canonical_no_stereo"] = "Unknown"
    negatives_data["SMILES_product_canonical_no_stereo"] = "Unknown"
    negatives_data["Kingdom"] = "Unknown"
    negatives_data["Class"] = "Unknown"
    negatives_data["Type"] = "Unknown"
    negatives_data["OriginalType"] = "Unknown"
    negatives_data["Fold"] = None
    negatives_data["Fold_ignore_in_eval"] = None

    negatives_data = negatives_data.reindex(
        negatives_data.index.repeat(
            negatives_data.ID.map(
                lambda x: (
                    len(negatives_to_accepted_tps_substrates[x])
                    if x in negatives_to_accepted_tps_substrates
                    else 1
                )
            )
        )
    ).reset_index(drop=True)

    for negative_id in negatives_to_accepted_tps_substrates:
        accepted_substrates = negatives_to_accepted_tps_substrates[negative_id]
        negatives_data.loc[
            negatives_data["ID"] == negative_id, "SMILES_substrate_canonical_no_stereo"
        ] = list(accepted_substrates)

    logger.info(
        "Assigned TPS substrate information to negative samples based on Rhea reaction directionality and MartsDB annotations."
    )

    for fold_idx, val_ids in enumerate(negatives_folds):
        negatives_data.loc[negatives_data["ID"].isin(val_ids), "Fold"] = fold_idx

    logger.info("Assigned folds to negative samples.")
    return negatives_data
