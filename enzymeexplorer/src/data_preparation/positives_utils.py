import logging
from pathlib import Path
import pandas as pd
from enzymeexplorer.src.data_preparation.constants import (
    MAJOR_CLASSES,
)
from enzymeexplorer.src.utils.data import get_canonical_smiles
from enzymeexplorer.src.data_preparation.mmseqs2_wrapper import MMSeqs2Wrapper
from collections import defaultdict
import pickle
from enzymeexplorer.src.data_preparation.common_utils import (
    cluster_dataset,
    get_is_splittable,
    get_stratified_group_kfold_splits,
)

logger = logging.getLogger(__file__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)



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


def prepare_positives_set(
    martsDB: pd.DataFrame,
    mmseqs: MMSeqs2Wrapper,
    id_to_kingdom_output_path: str,
    substrate_to_tps_type_output_path: str,
    min_seq_id: float,
    coverage: float,
    n_folds: int,
    final_positives_csv_path: str,
) -> pd.DataFrame:
    # computing categories per kingdom
    id_2_kingdom_dataset = (
        martsDB[["Enzyme_marts_ID", "Kingdom"]]
        .set_index("Enzyme_marts_ID")
        .to_dict()["Kingdom"]
    )
    with open(id_to_kingdom_output_path, "wb") as file:
        pickle.dump(id_2_kingdom_dataset, file)

    logger.info(f"Saved ID to Kingdom mapping to {id_to_kingdom_output_path}")

    substrate_2_tps_type = (
        martsDB[["SMILES_substrate_canonical_no_stereo", "Type"]]
        .drop_duplicates()
        .groupby("SMILES_substrate_canonical_no_stereo")["Type"]
        .apply(set)
        .reset_index()
        .set_index("SMILES_substrate_canonical_no_stereo")
        .to_dict()["Type"]
    )
    with open(substrate_to_tps_type_output_path, "wb") as file:
        pickle.dump(substrate_2_tps_type, file)

    logger.info(
        f"Saved substrate to TPS type mapping to {substrate_to_tps_type_output_path}"
    )

    martsDB_clusters_df, _ = cluster_dataset(
        martsDB,
        id_column="Enzyme_marts_ID",
        seq_column="Aminoacid_sequence",
        mmseqs=mmseqs,
        min_seq_id=min_seq_id,
        coverage=coverage,
    )

    logger.info(
        f"Clustered MartsDB sequences into {martsDB_clusters_df['Representative'].nunique()} clusters"
    )

    martsDB_with_clusters = martsDB.merge(
        martsDB_clusters_df,
        left_on="Enzyme_marts_ID",
        right_on="Member",
        how="left",
    )
    assert (
        martsDB_with_clusters[martsDB_with_clusters["Representative"].isna()].shape[0]
        == 0
    ), "Some sequences are not assigned to any cluster."

    is_martsdb_splittable, unsplittable_target_values = get_is_splittable(
        martsDB_with_clusters,
        id_column="Enzyme_marts_ID",
        cluster_id_column="Representative",
        target_col="SMILES_substrate_canonical_no_stereo",
        classes=MAJOR_CLASSES,
        n_folds=n_folds,
    )
    if not is_martsdb_splittable:
        raise ValueError(
            "The dataset cannot be split without leakage based on the provided target classes."
        )
    logger.info(
        f"MartsDB dataset is splittable without leakage. The following minor substrate classes will be ignored for evaluation: {', '.join(unsplittable_target_values)}"
    )

    positives_folds = get_stratified_group_kfold_splits(
        martsDB_with_clusters,
        id_column="Enzyme_marts_ID",
        cluster_id_column="Representative",
        optimize_distribution=True,
        target_col="SMILES_substrate_canonical_no_stereo",
        classes=MAJOR_CLASSES,
        n_folds=n_folds,
    )

    logger.info("Generated stratified group K-Fold splits for positives.")

    positives_data = martsDB[
        [
            "Enzyme_marts_ID",
            "Aminoacid_sequence",
            "SMILES_substrate_canonical_no_stereo",
            "SMILES_product_canonical_no_stereo",
            "Type",
            "OriginalType",
            "Kingdom",
            "Class",
        ]
    ].rename({"Enzyme_marts_ID": "ID"}, axis="columns")
    positives_data["Fold"] = None
    for fold_idx, val_ids in enumerate(positives_folds):
        positives_data.loc[positives_data["ID"].isin(val_ids), "Fold"] = fold_idx

    logger.info("Assigned folds to positive samples.")

    positives_data["Fold_ignore_in_eval"] = None

    positives_data.loc[
        positives_data["SMILES_substrate_canonical_no_stereo"].isin(
            unsplittable_target_values
        ),
        "Fold_ignore_in_eval",
    ] = 1

    positives_data.to_csv(
        Path(final_positives_csv_path),
        index=False,
    )
    logger.info(
        f"Final positives subset of size {len(positives_data)} saved to {Path(final_positives_csv_path)}"
    )
    return positives_data
