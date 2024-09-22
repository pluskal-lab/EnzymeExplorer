import os
import pickle
from shutil import copyfile
from pymol import cmd
from pathlib import Path
import argparse
from terpeneminer.src.structure_processing.structural_algorithms import (
    get_all_residues_per_file,
)


def parse_args() -> argparse.Namespace:
    """
    This function parses arguments
    :return: current argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="A script to compute secondary-structure residues of all proteins in a directory"
    )
    parser.add_argument(
        "--input-directory",
        help="A directory containing PDB structures",
        type=str,
        default="data/alphafold_structs/",
    )
    parser.add_argument(
        "--output-path",
        help="A file to save the computed secondary-structure residues to",
        type=str,
        default="data/alphafold_structs/file_2_all_residues.pkl",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cwd = os.getcwd()
    os.chdir(args.input_directory)
    pdb_files_raw = [filepath for filepath in Path(".").glob("*.pdb")]
    # substituting ids which Pymol cannot handle
    pdb_files = []
    filepath_2_corrected_filepath = {}
    for filepath in pdb_files_raw:
        filepath_str = str(filepath)
        if "(" in filepath_str or ")" in filepath_str or len(filepath_str.split()) > 1:
            filepath_2_corrected_filepath[filepath] = Path(
                "".join(filepath_str.replace("(", "").replace(")", "").split())
            )
            copyfile(filepath, filepath_2_corrected_filepath[filepath])
        pdb_files.append(filepath_2_corrected_filepath.get(filepath, filepath))
    file_2_all_residues = get_all_residues_per_file(pdb_files, cmd)
    os.chdir(cwd)
    with open(args.output_path, "wb") as f:
        pickle.dump(file_2_all_residues, f)
