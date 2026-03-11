python -m enzymeexplorer.src.data_preparation.prep_data --swissprot-tsv-path "./data/swissprot_with_af_structures.tsv" > outputs/logs/data_prep.log 2>&1

python -m enzymeexplorer.src.structure_processing.domain_detections \
    --needed-proteins-csv-path "data/martsDB_reactions_2026_02_22_preprocessed.csv" \
    --input-directory-with-structures "data/martsDB_pdbs/" \
    --is-bfactor-confidence \
    --csv-id-column "Enzyme_marts_ID" \
    --recompute-existing-secondary-structure-residues \
    --detected-regions-root-path "data/martsDB_detected_domains/" \
    --n-jobs 16 --detections-output-path "data/martsDB_detected_domains.pkl" \
    --store-domains --domains-output-path "data/martsDB_detected_domains" > outputs/logs/martsDB_structures_segmentation.log 2>&1

python -m enzymeexplorer.src.structure_processing.domain_detections \
    --needed-proteins-csv-path "data/EnzymeExplorer_Dataset.csv" \
    --csv-id-column "ID" \
    --input-directory-with-structures "data/enzyme_explorer_pdbs/" \
    --is-bfactor-confidence \
    --recompute-existing-secondary-structure-residues \
    --detected-regions-root-path "data/enzyme_explorer_detected_domains/" \
    --n-jobs 16 --detections-output-path "data/enzyme_explorer_detected_domains.pkl" \
    --store-domains --domains-output-path "data/enzyme_explorer_detected_domains" > outputs/logs/enzyme_explorer_structures_segmentation.log 2>&1

python -m enzymeexplorer.src.structure_processing.get_structural_features \
    -refdoms ./data/martsDB_detected_domains.pkl -refdomsstructs ./data/martsDB_detected_domains \
    -querydoms ./data/martsDB_detected_domains.pkl -querydomsstructs ./data/martsDB_detected_domains \
    -storeintermediates -outputdir ./data/martsDB_vs_martsDB > outputs/logs/martsDB_vs_martsDB.log 2>&1

python -m enzymeexplorer.src.structure_processing.get_structural_features \
    -refdoms ./data/martsDB_detected_domains.pkl -refdomsstructs ./data/martsDB_detected_domains \
    -querydoms ./data/enzyme_explorer_detected_domains.pkl -querydomsstructs ./data/enzyme_explorer_detected_domains \
    -storeintermediates -outputdir ./data/enzexp_vs_martsDB > outputs/logs/enzexp_vs_martsDB.log 2>&1