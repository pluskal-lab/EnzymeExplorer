"""This script detects TPS domains in protein structures"""

import os
import tempfile
import yaml
import configargparse
from pathlib import Path
from collections import defaultdict
import pickle
import logging
from pymol import cmd  # type: ignore
from enzymeexplorer.src.structure_processing._pool_service import pool_session
from enzymeexplorer.src.structure_processing.structural_algorithms import (
    MappedRegion,
    get_remaining_residues,
    get_mapped_regions_with_surroundings_parallel,
)
from enzymeexplorer.src.structure_processing.utils import (
    save_file_to_all_residues,
    get_pdb_files,
    filter_pdb_files_by_foldseek_alignments,
    filter_domains_by_foldseek_alignments,
    store_domains,
    store_domain_separately,
    detect_domains_roughly,
    plot_aligned_domains,
    get_confident_residue_mappings,
    pick_disjoint_domains,
)

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# Canonical default domain templates — kept in sync with
# enzymeexplorer/configs/enzyme_explorer_domain_detection_config.yaml so that
# both the CLI (no --config) and the Python wrapper produce identical detections.
DEFAULT_DOMAIN_TEMPLATES = [
    {
        "name": "alpha",
        "path": "data/domain_templates/1ps1.pdb",
        "residues": "resi 39-308 & chain A & ss H+S",
        "thresholds": {"tmscore": 0.35, "min_align_len": 90},
    },
    {
        "name": "beta",
        "path": "data/domain_templates/5eat.pdb",
        "residues": (
            "resi 37-57+64-97+104-117+123-129+138-156+162-195+203-213+223-239"
            " & chain A & ss H+S"
        ),
        "thresholds": {"tmscore": 0.45, "min_align_len": 60},
    },
    {
        "name": "gamma",
        "path": "data/domain_templates/3p5r.pdb",
        "residues": (
            "resi 138-151+157-171+185-222+233-248+258-275+281-304+313-339"
            " & chain A & ss H+S"
        ),
        "thresholds": {"tmscore": 0.45, "min_align_len": 70},
    },
    {
        "name": "ids",
        "path": "data/domain_templates/1ubw.pdb",
        "residues": (
            "resi 73-85+93-121+139-161+167-191+204-231+236-263+324-346+352-361"
            " & chain A & ss H+S"
        ),
        "thresholds": {"tmscore": 0.50, "min_align_len": 85},
    },
    {
        "name": "delta",
        "path": "data/domain_templates/1w6j.pdb",
        "residues": (
            "resi 73-87+385-399+401-403+405-421+454-470+480-493+531-547+553-570"
            "+585-599+610-622+633-638+649-662+667-680+707-722+727-729"
            " & chain A & ss H+S"
        ),
        "thresholds": {"tmscore": 0.54, "min_align_len": 100},
    },
    {
        "name": "epsilon",
        "path": "data/domain_templates/1w6j.pdb",
        "residues": (
            "resi 103-115+123-134+151-164+171-183+191-200+213-217+226-228+231-246"
            "+254-263+268-270+273-277+291-306+309-330+337-351+356-371+376-378+510-515"
            " & chain A & ss H+S"
        ),
        "thresholds": {"tmscore": 0.65, "min_align_len": 95},
    },
    {
        "name": "zeta",
        "path": "data/domain_templates/P37295.pdb",
        "residues": "resi 3-248 & chain A & ss H+S",
        "thresholds": {"tmscore": 0.65, "min_align_len": 85},
    },
]


def parse_args() -> configargparse.Namespace:
    """
    This function parses arguments
    :return: current configargparse.Namespace
    """
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser
    )
    parser.add_argument(
        "-c",
        "--config",
        is_config_file=True,
        help="config file path",
        default="configs/enzyme_explorer_domain_detection_config.yaml",
    )

    parser.add_argument(
        "--needed-proteins-csv-path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--csv-id-column",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--input-directory-with-structures",
        help="A directory containing PDB structures",
        type=str,
        default="data/structs/",
    )
    parser.add_argument("--n-jobs", type=int, default=20)
    parser.add_argument("--n-iters", type=int, default=3)
    parser.add_argument(
        "--detect-multiple-domains-in-each-iteration",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--detections-output-path",
        help="A path to save a dictionary with the detected domains to",
        type=str,
    )
    parser.add_argument(
        "--store-domains",
        help="A flag to store detected domains",
        action="store_true",
        default=True,
    )
    parser.add_argument("--detected-regions-root-path", type=str)
    parser.add_argument(
        "--domains-output-path",
        help="A root path for saving the detected domains to",
        type=str,
    )
    parser.add_argument("--is-bfactor-confidence", action="store_true", default=True)
    parser.add_argument(
        "--do-not-store-intermediate-files", action="store_true", default=True
    )
    parser.add_argument(
        "--secondary-structure-residues-path",
        type=str,
        default="data/secondary_structure_residues.pkl",
    )
    parser.add_argument(
        "--recompute-existing-secondary-structure-residues",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--prefilter-pdbs-by-foldseek",
        action="store_true",
    )
    parser.add_argument(
        "--prefilter-e-value",
        type=float,
        default=10,
    )
    parser.add_argument(
        "--postfilter-domains-by-foldseek",
        action="store_true",
    )
    parser.add_argument(
        "--postfilter-e-value",
        type=float,
        default=5,
    )
    parser.add_argument(
        "--domain-templates",
        nargs="+",
        default=DEFAULT_DOMAIN_TEMPLATES,
    )
    return parser.parse_args()


def detect_domains(args) -> dict:
    """Run domain detection from a parsed args object.

    Returns the ``filename_2_known_regions_completed_confident`` mapping
    (``{pdb_filename_stem: list[MappedRegion]}``) in addition to writing it to
    ``args.detections_output_path``. Use ``run_domain_detection(...)`` for a
    keyword-friendly entry point.
    """
    input_directory = Path(args.input_directory_with_structures).absolute()

    domain_templates = args.domain_templates
    domain_templates = [yaml.safe_load(template) for template in domain_templates]

    for domain_template in domain_templates:
        domain_template["path"] = Path(domain_template["path"]).absolute()
    supported_domains = [template["name"] for template in domain_templates]
    domain_2_threshold = {
        template["name"]: template["thresholds"] for template in domain_templates
    }

    # reading the needed proteins
    pdb_files = get_pdb_files(
        args.needed_proteins_csv_path,
        args.csv_id_column,
        input_directory,
    )
    domain_type_to_pdb_files = {
        domain_name: pdb_files for domain_name in supported_domains
    }
    if args.prefilter_pdbs_by_foldseek:
        logger.info(f"Filtering PDB files by foldseek alignment to domain templates")
        domain_type_to_pdb_files = filter_pdb_files_by_foldseek_alignments(
            pdb_files,
            domain_templates,
            e_value=args.prefilter_e_value,
            cov_mode=1,
            coverage=0.5,
        )
        pdb_files_filtered = [
            pdb_file
            for pdb_file in set(
                [
                    pdb_file
                    for pdb_files in domain_type_to_pdb_files.values()
                    for pdb_file in pdb_files
                ]
            )
        ]
        logger.info(
            f"{len(pdb_files_filtered)} out of {len(pdb_files)} PDB files passed the foldseek prefiltering with e-value threshold {args.prefilter_e_value}"
        )
        if (
            len(pdb_files) - len(pdb_files_filtered) > 0
            and len(pdb_files) - len(pdb_files_filtered) < 11
        ):  # only logging the filtered out files if there are less than 100 of them, otherwise it would be too much to log
            logger.info(
                f"The following PDB files were filtered out by foldseek prefiltering:"
            )
            for pdb_file in set(pdb_files) - set(pdb_files_filtered):
                logger.info(f"\t{pdb_file}")
        pdb_files = pdb_files_filtered

    cwd = os.getcwd()
    n_jobs = int(getattr(args, "n_jobs", 20))
    # One pool service for the whole detection: serves SSR (parallel
    # path), get_alignments, get_mapped_regions_with_surroundings_parallel,
    # and store_domains. Spawn-based — PyMOL is not fork-safe once the
    # parent has loaded any PDB.
    with pool_session(n_jobs=n_jobs, working_dir=str(input_directory)):
        secondary_structure_residues_path = Path(args.secondary_structure_residues_path)
        if (
            not secondary_structure_residues_path.exists()
            or args.recompute_existing_secondary_structure_residues
        ):
            save_file_to_all_residues(
                secondary_structure_residues_path=secondary_structure_residues_path,
                pdb_files=pdb_files,
                domain_templates=domain_templates,
            )
    
        with open(secondary_structure_residues_path, "rb") as file:
            file_2_all_residues = pickle.load(file)
    
        # Capture the original SS-residues map so the per-call alignment-batch
        # workers can skip the redundant `cmd.iterate (X & ss H+S)` on every
        # freshly-loaded query (saves ~5 ms × N_calls).
        import enzymeexplorer.src.structure_processing.structural_algorithms as _sa
        _sa._SS_FULL_MAP_CACHE.clear()
        _sa._SS_FULL_MAP_CACHE.update(file_2_all_residues)
    
        # Parent-process cwd is also pinned to input_directory so that
        # in-process callers using relative `f"{stem}.pdb"` paths continue
        # to work alongside spawn-pool workers (which chdir via the
        # service initializer).
        os.chdir(input_directory)

        filename_2_known_regions: dict[str, list[MappedRegion]] = defaultdict(list)
        filename_2_remaining_residues: dict[str, set[str]] = file_2_all_residues.copy()
        domain_type_to_files_with_no_detections = defaultdict(set)
        for detection_iter in range(args.n_iters):
            logger.info(f"Starting detection iteration {detection_iter + 1}")
            
            # Detecting TPS domains in protein structures
            filename_2_potential_regions = detect_domains_roughly(
                {
                    domain_type: [
                        pdb_file
                        for pdb_file in pdb_files
                        if (len(filename_2_remaining_residues.get(pdb_file.stem, [])) >= 20)
                        and (pdb_file.stem not in domain_type_to_files_with_no_detections[domain_type])
                    ]
                    for domain_type, pdb_files in domain_type_to_pdb_files.items()
                },  # only considering files for which there are remaining residues
                filename_2_remaining_residues,
                domain_templates=domain_templates,
                args=args,
                iteration=detection_iter + 1,
            )
            for domain_type, pdb_files in domain_type_to_pdb_files.items():
                domain_type_to_files_with_no_detections[domain_type].update(
                [
                    pdb_file.stem
                    for pdb_file in pdb_files
                    if filename_2_potential_regions.get(pdb_file.stem, []) == []
                ]
            )
    
            if args.detect_multiple_domains_in_each_iteration:
                filename_2_detected_region = {
                    filename: (
                        pick_disjoint_domains(
                            sorted(
                                filename_2_potential_regions[filename],
                                key=lambda r: r.tmscore,
                                reverse=True,
                            )
                        )
                    )
                    for filename in filename_2_potential_regions
                    if len(filename_2_potential_regions[filename]) > 0
                }
            else:
                filename_2_detected_region = {
                    filename: (
                        pick_disjoint_domains(
                            sorted(
                                filename_2_potential_regions[filename],
                                key=lambda r: r.tmscore,
                                reverse=True,
                            )
                        )[:1]
                    )
                    for filename in filename_2_potential_regions
                    if len(filename_2_potential_regions[filename]) > 0
                }
    
            filename_2_detected_region_with_potential_expansion = (
                get_mapped_regions_with_surroundings_parallel(
                    list(filename_2_detected_region.keys()),
                    file_2_all_residues,
                    filename_2_detected_region,
                    n_jobs=args.n_jobs,
                    helix_sheet_neighbor_dist_threshold=15,
                    helix_sheet_domain_dist_threshold=15,
                )
            )
    
            # Get unsegmented parts
            filename_2_remaining_residues = get_remaining_residues(
                filename_2_detected_region_with_potential_expansion,
                filename_2_remaining_residues,
            )
            for filename in filename_2_detected_region:
                if len(filename_2_detected_region[filename]) == 0:
                    continue
                for region in filename_2_detected_region[filename]:
                    num_regions_of_same_domain_type = len(
                        [
                            reg
                            for reg in filename_2_known_regions[filename]
                            if reg.domain == region.domain
                        ]
                    )
                    region.module_id = (
                        f"{filename}_{region.domain}_{num_regions_of_same_domain_type}"
                    )
                    filename_2_known_regions[filename].append(region)
    
        if not args.do_not_store_intermediate_files:
            (Path(cwd) / args.detected_regions_root_path).mkdir(parents=True, exist_ok=True)
            store_domain_separately(
                filename_2_known_regions,
                supported_domains,
                Path(cwd) / args.detected_regions_root_path,
            )
    
        filename_2_known_regions_completed_iter1 = (
            get_mapped_regions_with_surroundings_parallel(
                list(filename_2_known_regions.keys()),
                file_2_all_residues,
                filename_2_known_regions,
                n_jobs=args.n_jobs,
                helix_sheet_neighbor_dist_threshold=20,
                helix_sheet_domain_dist_threshold=30,
            )
        )
    
        filename_2_known_regions_completed_iter2 = (
            get_mapped_regions_with_surroundings_parallel(
                list(filename_2_known_regions_completed_iter1.keys()),
                file_2_all_residues,
                filename_2_known_regions_completed_iter1,
                n_jobs=args.n_jobs,
                helix_sheet_neighbor_dist_threshold=17,
                helix_sheet_domain_dist_threshold=25,
            )
        )
    
        # Getting confident residues
        if args.is_bfactor_confidence:
            filename_2_known_regions_completed_confident = get_confident_residue_mappings(
                filename_2_known_regions_completed_iter2,
                file_2_all_residues,
                domain_2_threshold,
            )
        else:
            filename_2_known_regions_completed_confident = (
                filename_2_known_regions_completed_iter2
            )
    
        are_domains_stored = False
        if args.store_domains:
            store_domains(
                filename_2_known_regions_completed_confident,
                supported_domains,
                Path(cwd) / args.domains_output_path,
                n_jobs=args.n_jobs,
            )
            are_domains_stored = True
        if args.postfilter_domains_by_foldseek:
            logger.info(
                f"Filtering detected domains by foldseek alignment to domain templates"
            )
            if are_domains_stored:
                domain_pdbs_root = Path(cwd) / args.domains_output_path
                filename_2_known_regions_completed_confident = (
                    filter_domains_by_foldseek_alignments(
                        filename_2_known_regions_completed_confident,
                        supported_domains,
                        domain_templates,
                        domain_pdbs_root,
                        e_value=args.postfilter_e_value,
                    )
                )
            else:
                with tempfile.TemporaryDirectory() as tmpdir:
                    domain_pdbs_root = Path(tmpdir) / "domains"
                    store_domains(
                        filename_2_known_regions_completed_confident,
                        supported_domains,
                        domain_pdbs_root,
                        n_jobs=args.n_jobs,
                    )
                    filename_2_known_regions_completed_confident = (
                        filter_domains_by_foldseek_alignments(
                            filename_2_known_regions_completed_confident,
                            supported_domains,
                            domain_templates,
                            domain_pdbs_root,
                            e_value=args.postfilter_e_value,
                        )
                    )
    
        if not args.do_not_store_intermediate_files:
            (Path(cwd) / args.detected_regions_root_path).mkdir(parents=True, exist_ok=True)
            plot_aligned_domains(
                filename_2_known_regions_completed_confident,
                supported_domains,
                Path(cwd) / args.detected_regions_root_path,
            )

    # Pool service has shut down with the `with pool_session(...)` exit;
    # restore the parent's cwd and clear the SS cache.
    os.chdir(cwd)
    import enzymeexplorer.src.structure_processing.structural_algorithms as _sa
    _sa._SS_FULL_MAP_CACHE.clear()

    # save the confident regions
    Path(args.detections_output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.detections_output_path, "wb") as f:
        pickle.dump(filename_2_known_regions_completed_confident, f)

    logger.info(
        f"Finished domain detection. Detected domains saved to {args.detections_output_path}"
    )
    return filename_2_known_regions_completed_confident


def run_domain_detection(
    *,
    input_directory_with_structures,
    needed_proteins_csv_path,
    csv_id_column,
    detections_output_path,
    detected_regions_root_path=None,
    domains_output_path=None,
    n_jobs=10,
    n_iters=3,
    is_bfactor_confidence=True,
    do_not_store_intermediate_files=True,
    store_domains=True,
    detect_multiple_domains_in_each_iteration=True,
    secondary_structure_residues_path="data/secondary_structure_residues.pkl",
    recompute_existing_secondary_structure_residues=True,
    prefilter_pdbs_by_foldseek=False,
    prefilter_e_value: float = 10.0,
    postfilter_domains_by_foldseek=False,
    postfilter_e_value: float = 5.0,
    domain_templates=None,
) -> dict:
    """Python-callable wrapper around :func:`detect_domains`.

    Defaults mirror enzymeexplorer/configs/enzyme_explorer_domain_detection_config.yaml.
    Constructs an args namespace from kwargs and runs the same orchestration
    used by the CLI. Returns the detected-domains mapping.
    """
    import argparse as _argparse

    if domain_templates is None:
        domain_templates = DEFAULT_DOMAIN_TEMPLATES
    # Re-serialise as YAML strings so detect_domains() can yaml.safe_load them
    # exactly like the CLI path does.
    serialised_templates = [yaml.safe_dump(t) for t in domain_templates]

    args = _argparse.Namespace(
        input_directory_with_structures=str(input_directory_with_structures),
        needed_proteins_csv_path=str(needed_proteins_csv_path),
        csv_id_column=csv_id_column,
        detections_output_path=str(detections_output_path),
        detected_regions_root_path=(
            str(detected_regions_root_path) if detected_regions_root_path else None
        ),
        domains_output_path=(
            str(domains_output_path) if domains_output_path else None
        ),
        n_jobs=n_jobs,
        n_iters=n_iters,
        is_bfactor_confidence=is_bfactor_confidence,
        do_not_store_intermediate_files=do_not_store_intermediate_files,
        store_domains=store_domains,
        detect_multiple_domains_in_each_iteration=detect_multiple_domains_in_each_iteration,
        secondary_structure_residues_path=str(secondary_structure_residues_path),
        recompute_existing_secondary_structure_residues=recompute_existing_secondary_structure_residues,
        prefilter_pdbs_by_foldseek=prefilter_pdbs_by_foldseek,
        prefilter_e_value=prefilter_e_value,
        postfilter_domains_by_foldseek=postfilter_domains_by_foldseek,
        postfilter_e_value=postfilter_e_value,
        domain_templates=serialised_templates,
    )
    return detect_domains(args)


def main():
    args = parse_args()
    detect_domains(args)


if __name__ == "__main__":
    main()
