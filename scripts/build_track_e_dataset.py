#!/usr/bin/env python3
"""Build the Track E dataset: new-dataset TPS + old-dataset negatives.

Track E isolates the effect of the negative set by keeping TPS from the new
dataset while swapping in old negatives. Comparing D→E measures the value of
more TPS; comparing E→B measures the value of better negatives.

Outputs
-------
data/track_e_new_tps_old_neg.csv
    Combined training CSV with new-dataset column names.
data/gathered_embs_esm-1v-finetuned-subseq_track_e_embs_avg.h5
    Merged embeddings covering both new TPS and old negatives.
"""

import pandas as pd
import numpy as np


def main():
    new = pd.read_csv("data/EnzymeExplorer_Dataset.csv")
    old = pd.read_csv("data/TPS-Nov19_2023_with_synced_folds.csv")

    new_tps = new[new["Type"] != "Unknown"].copy()
    old_neg = old[old["Type (mono, sesq, di, …)"] == "Unknown"].copy()

    print(f"New TPS rows: {len(new_tps)} ({new_tps['ID'].nunique()} enzymes)")
    print(f"Old negatives: {len(old_neg)} ({old_neg['Uniprot ID'].nunique()} unique)")

    # Map old negatives to new-dataset column format
    old_neg_mapped = pd.DataFrame(
        {
            "ID": old_neg["Uniprot ID"].values,
            "Aminoacid_sequence": old_neg["Amino acid sequence"].values,
            "SMILES_substrate_canonical_no_stereo": old_neg[
                "SMILES_substrate_canonical_no_stereo"
            ].values,
            "SMILES_product_canonical_no_stereo": old_neg[
                "SMILES_product_canonical_no_stereo"
            ].values,
            "Type": "Unknown",
            "OriginalType": "Unknown",
            "Kingdom": old_neg["Kingdom (plant, fungi, bacteria)"].values,
            "Class": old_neg["Class (I or II)"].values,
        }
    )

    # Round-robin fold assignment for old negatives (5 folds: 0-4)
    neg_ids = old_neg_mapped["ID"].unique()
    id_to_fold = {uid: i % 5 for i, uid in enumerate(np.random.RandomState(42).permutation(neg_ids))}
    old_neg_mapped["Fold"] = old_neg_mapped["ID"].map(id_to_fold)
    old_neg_mapped["Fold_ignore_in_eval"] = False

    combined = pd.concat([new_tps, old_neg_mapped], ignore_index=True)
    print(f"Combined: {len(combined)} rows")

    out_csv = "data/track_e_new_tps_old_neg.csv"
    combined.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")

    # Build merged embeddings
    old_embs = pd.read_hdf(
        "data/gathered_embs_esm-1v-finetuned-subseq_old_dataset_embs_avg.h5"
    )
    new_embs = pd.read_hdf(
        "data/gathered_embs_esm-1v-finetuned-subseq_new_dataset_embs_avg.h5"
    )

    # Rename old embs column to match new
    old_embs = old_embs.rename(columns={"Uniprot ID": "ID"})

    # Keep only old negatives
    old_neg_embs = old_embs[old_embs["ID"].isin(set(old_neg["Uniprot ID"].unique()))]
    print(f"Old negative embeddings: {len(old_neg_embs)}")

    # Keep only new TPS
    new_tps_ids = set(new_tps["ID"].unique())
    new_all_embs = new_embs[new_embs["ID"].isin(new_tps_ids)]
    print(f"New TPS embeddings: {len(new_all_embs)}")

    # Actually for cross-dataset eval, we need ALL new dataset embeddings for eval
    # So merge: all new dataset embeddings + old negative embeddings
    merged = pd.concat([new_embs, old_neg_embs], ignore_index=True)
    # Remove duplicates (keep new dataset version if overlap)
    merged = merged.drop_duplicates(subset=["ID"], keep="first")
    print(f"Merged embeddings: {len(merged)}")

    out_h5 = "data/gathered_embs_esm-1v-finetuned-subseq_track_e_embs_avg.h5"
    merged.to_hdf(out_h5, key="data", mode="w")
    print(f"Saved {out_h5}")


if __name__ == "__main__":
    main()
