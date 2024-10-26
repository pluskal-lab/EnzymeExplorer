from uuid import uuid4

from fastapi import FastAPI, File, UploadFile, BackgroundTasks, Form
from fastapi.responses import FileResponse
from pathlib import Path
import os
import pickle
from shutil import copyfile, rmtree
import logging
import subprocess
import re

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()  # Output to console
    ]
)

# Create FastAPI app instance
app = FastAPI()

@app.post("/detect_domains/")
async def upload_file(file: UploadFile = File(...),
                      is_bfactor_confidence: bool = Form(...)):
    # Read the contents of the uploaded file
    file_contents = await file.read()

    # Define the path where the .pdb file will be saved
    pdb_directory_temp = Path("_temp")
    if not pdb_directory_temp.exists():
        pdb_directory_temp.mkdir()
        af_source_path = Path("/home/samusevich/TerpeneMiner/data/alphafold_structs")
        for pdb_standard_id in ["1ps1",  "5eat", "3p5r", "P48449"]:
            pdb_standard_file_path = af_source_path / f"{pdb_standard_id}.pdb"
            copyfile(pdb_standard_file_path, pdb_directory_temp / f"{pdb_standard_id}.pdb")

    # Getting the ID
    pdb_id = file.filename.split(".")[0]
    pdb_id = re.sub(r'\(.*?\)', '', pdb_id)
    pdb_id = "".join(pdb_id.replace("-", "").split())

    # Define the path where the .pdb file will be saved
    pdb_file_path = pdb_directory_temp / f"{pdb_id}.pdb"

    # Saving the ID into a csv file
    id_filepath = f'{pdb_directory_temp / "dummy_id.csv"}'
    with open(id_filepath, "a") as file:
        file.writelines(f"ID\n{pdb_id}\n")

    # Save the content as a .pdb file
    with open(pdb_file_path, "wb") as pdb_file:
        pdb_file.write(file_contents)
    temp_filepath_name = Path("data/alphafold_structs") / f"{pdb_id}.pdb"
    if not temp_filepath_name.exists():
        copyfile(pdb_file_path, temp_filepath_name)
    temp_filepath_name_to_delete = not temp_filepath_name.exists()

    domain_detections_path = f"_temp/filename_2_detected_domains_completed_confident_{pdb_id}.pkl"
    detected_domain_structures_root = Path("_temp/detected_domains")
    if not detected_domain_structures_root.exists():
        detected_domain_structures_root.mkdir()
    os.system(
        "python -m terpeneminer.src.structure_processing.domain_detections "
        f'--needed-proteins-csv-path "{id_filepath}" '
        "--csv-id-column ID "
        "--n-jobs 16 "
        "--input-directory-with-structures _temp "
        f"{'--is-bfactor-confidence ' if is_bfactor_confidence else ''}"
        f'--detections-output-path "{domain_detections_path}" '
        f'--detected-regions-root-path _temp '
        f'--domains-output-path "{detected_domain_structures_root}" '
        "--store-domains "
        "--recompute-existing-secondary-structure-residues "
        "--do-not-store-intermediate-files"
    )

    with open(domain_detections_path, "rb") as file:
        detected_domains = pickle.load(file)

    all_secondary_structure_residues_path = "_temp/file_2_all_residues.pkl"
    with open(all_secondary_structure_residues_path, "rb") as file:
        file_2_all_residues = pickle.load(file)
    if pdb_id in file_2_all_residues:
        secondary_structure_res = file_2_all_residues[pdb_id]
    else:
        secondary_structure_res = None

    logger.info("Detected %d domains. Starting comparison to the known domains..", len(detected_domains))
    if detected_domains:
        current_computation_id = uuid4()
        comparison_results_path = f"_temp/filename_2_regions_vs_known_reg_dists_{current_computation_id}.pkl"
        os.system("python -m terpeneminer.src.structure_processing.comparing_to_known_domains_foldseek "
                  f'--known-domain-structures-root data/detected_domains/all '
                  f'--detected-domain-structures-root "{detected_domain_structures_root}" '
                  '--path-to-known-domains-subset data/domains_subset.pkl '
                  f'--output-path "{comparison_results_path}" '
                  f'--pdb-id "{pdb_id}"')

        logger.info("Compared detected domains to the known ones!")

        with open(comparison_results_path, "rb") as file:
            comparison_results = pickle.load(file)

        domain_id_2_aligned_pdb = {}

        with open('data/reaction_types_and_kingdoms.pkl', 'rb') as file:
            id_2_reaction_types, id_2_kingdom = pickle.load(file)
        with open('data/domain_module_id_2_domain_type.pkl', 'rb') as file:
            domain_module_id_2_domain_type = pickle.load(file)
        with open('data/id_2_domain_config.pkl', 'rb') as file:
            id_2_domain_config = pickle.load(file)

        for detected_domain_id in comparison_results[pdb_id]:
            detected_domain_file_path = detected_domain_structures_root / f"{detected_domain_id}.pdb"
            pdb_id_current = detected_domain_id.split('_')[0]
            closest_known_domain_id, foldseek_tm_score = max([(known_domain_id, tmscore)
                                                              for known_domain_id, tmscore in comparison_results[pdb_id][detected_domain_id]
                                                              if known_domain_id.split('_')[0] != pdb_id_current],
                                                             key=lambda x: x[1])
            closest_known_domain_id_pdb_id = closest_known_domain_id.split('_')[0]
            closest_known_domain_file_path = Path("data/detected_domains/all") / f"{closest_known_domain_id}.pdb"

            aligned_pdb_path = Path("_temp") / f"aligned_{detected_domain_id}_to_{closest_known_domain_id}"
            # Run TM-align and capture the output
            try:
                result = subprocess.run(
                    ["TMalign", closest_known_domain_file_path, detected_domain_file_path, "-o", aligned_pdb_path],
                    check=True,
                    capture_output=True,  # Capture stdout and stderr
                    text=True  # Ensure output is in text form, not bytes
                )
            except subprocess.CalledProcessError as e:
                raise ValueError(f"TM-align failed, details {e}")

            # Extract TM-score from the output
            output = result.stdout
            tm_score = None
            for line in output.splitlines():
                if "TM-score" in line and "Chain_1" in line:  # TM-score line (ignores local TM-scores)
                    tm_score = float(line.split()[1])
                    break
            domain_id_2_aligned_pdb[detected_domain_id] = {"closest_known_domain_pdb_id": closest_known_domain_id_pdb_id,
                                                           "whole_structure_domain_config": id_2_domain_config[closest_known_domain_id_pdb_id],
                                                           "closest_domain_type": domain_module_id_2_domain_type[closest_known_domain_id],
                                                           "closest_id_reaction_types": [tps_type.replace('Class', 'class')
                                                                                         for tps_type in id_2_reaction_types[closest_known_domain_id_pdb_id]],
                                                           "closest_id_kingdom": id_2_kingdom[closest_known_domain_id_pdb_id],
                                                           "tm_score": tm_score,
                                                           "aligned_pdb_name": f"{aligned_pdb_path.name}_all_atm"}
            os.remove(aligned_pdb_path)
            os.remove(f"{aligned_pdb_path}_all")
            os.remove(f"{aligned_pdb_path}_atm")
            os.remove(f"{aligned_pdb_path}_all_atm_lig")

        logger.info("Predicting domain types..")
        domain_predictions_path = f"_temp/domain_id_2_predictions_{uuid4()}.pkl"
        os.system(
            "python -m terpeneminer.src.structure_processing.predict_domain_types "
            "--tps-classifiers-path data/classifier_domain_and_plm_checkpoints.pkl "
            "--domain-classifiers-path data/domain_type_predictors_foldseek.pkl "
            f"--path-to-domain-comparisons {comparison_results_path} "
            f'--id "{pdb_id}" '
            f'--output-path "{domain_predictions_path}" ')

        with open(domain_predictions_path, "rb") as file:
            domain_id_2_predictions = pickle.load(file)
        os.remove(comparison_results_path)
        os.remove(domain_predictions_path)

    os.remove(pdb_file_path)
    if temp_filepath_name_to_delete:
        os.remove(temp_filepath_name)
    os.remove(id_filepath)
    os.remove(domain_detections_path)
    rmtree(detected_domain_structures_root)

    return {"domains": detected_domains, "secondary_structure_residues": secondary_structure_res,
            "comparison_to_known_domains": comparison_results[pdb_id] if detected_domains else None,
            "domain_type_predictions": domain_id_2_predictions if detected_domains else None,
            "aligned_pdb_filepaths": domain_id_2_aligned_pdb if detected_domains else None}

def delete_file(file_path: str):
    os.remove(file_path)

# endpoint to download the aligned PDB file
@app.get("/download_pdb/{aligned_pdb_name}")
async def download_aligned_pdb(aligned_pdb_name: str, background_tasks: BackgroundTasks):
    aligned_pdb_path = Path("_temp") / aligned_pdb_name
    if not os.path.exists(aligned_pdb_path):
        return {"error": "File not found"}
    # schedule file deletion after the response is sent
    background_tasks.add_task(delete_file, aligned_pdb_path)
    return FileResponse(aligned_pdb_path, media_type='application/octet-stream', filename=Path(aligned_pdb_path).name)


@app.post("/predict_tps/")
async def upload_file(file: UploadFile = File(...),
                      is_bfactor_confidence: bool = Form(...)):
    # Read the contents of the uploaded file
    file_contents = await file.read()

    # Define the path where the .pdb file will be saved
    pdb_directory_temp = Path("_temp")
    if not pdb_directory_temp.exists():
        pdb_directory_temp.mkdir()
        af_source_path = Path("/home/samusevich/TerpeneMiner/data/alphafold_structs")
        for pdb_standard_id in ["1ps1",  "5eat", "3p5r", "P48449"]:
            pdb_standard_file_path = af_source_path / f"{pdb_standard_id}.pdb"
            copyfile(pdb_standard_file_path, pdb_directory_temp / f"{pdb_standard_id}.pdb")

    # Getting the ID
    pdb_id = file.filename.split(".")[0]
    pdb_id = re.sub(r'\(.*?\)', '', pdb_id)
    pdb_id = "".join(pdb_id.replace("-", "").split())

    # Define the path where the .pdb file will be saved
    pdb_file_path = pdb_directory_temp / f"{pdb_id}.pdb"

    # Saving the ID into a csv file
    id_filepath = f'{pdb_directory_temp / "dummy_id.csv"}'
    with open(id_filepath, "a") as file:
        file.writelines(f"ID\n{pdb_id}\n")

    # Save the content as a .pdb file
    with open(pdb_file_path, "wb") as pdb_file:
        pdb_file.write(file_contents)
    temp_filepath_name = Path("data/alphafold_structs") / f"{pdb_id}.pdb"
    if not temp_filepath_name.exists():
        copyfile(pdb_file_path, temp_filepath_name)
    temp_filepath_name_to_delete = not temp_filepath_name.exists()

    domain_detections_path = f"_temp/filename_2_detected_domains_completed_confident_{pdb_id}.pkl"
    detected_domain_structures_root = Path("_temp/detected_domains")
    if not detected_domain_structures_root.exists():
        detected_domain_structures_root.mkdir()
    os.system(
        "python -m terpeneminer.src.structure_processing.domain_detections "
        f'--needed-proteins-csv-path "{id_filepath}" '
        "--csv-id-column ID "
        "--n-jobs 16 "
        "--input-directory-with-structures _temp "
        f"{'--is-bfactor-confidence ' if is_bfactor_confidence else ''}"
        f'--detections-output-path "{domain_detections_path}" '
        f'--detected-regions-root-path _temp '
        f'--domains-output-path "{detected_domain_structures_root}" '
        "--store-domains "
        "--recompute-existing-secondary-structure-residues "
        "--do-not-store-intermediate-files"
    )

    with open(domain_detections_path, "rb") as file:
        detected_domains = pickle.load(file)

    logger.info("Detected %d domains. Starting comparison to the known domains..", len(detected_domains))
    if detected_domains:
        current_computation_id = uuid4()
        comparison_results_path = f"_temp/filename_2_regions_vs_known_reg_dists_{current_computation_id}.pkl"
        os.system(
            "python -m terpeneminer.src.structure_processing.comparing_to_known_domains "
            "--input-directory-with-structures data/alphafold_structs/ "
            "--n-jobs 16 "
            f'--domain-detections-path "{domain_detections_path}" '
            "--domain-detections-residues-path _temp/file_2_all_residues.pkl "
            "--path-to-all-known-domains data/alphafold_structs/regions_completed_very_confident_all_ALL.pkl "
            "--path-to-known-domains-subset data/domains_subset.pkl "
            f'--pdb-filepath "{temp_filepath_name}" '
            f'--output-path "{comparison_results_path}"')

        logger.info("Compared detected domains to the known ones!")

        with open(comparison_results_path, "rb") as file:
            comparison_results = pickle.load(file)

        print(comparison_results)


