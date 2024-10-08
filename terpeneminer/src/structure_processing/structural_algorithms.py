# pylint: disable=C0302
"""This module contains our structural algorithms for segmentation of a protein structure into TPS-specific domains
and comparison of domains between each other"""

import os
import pickle
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Optional
from uuid import uuid4

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
from psico.fitting import tmalign  # type: ignore
from pymol import CmdException, cmd, stored  # type: ignore
from scipy.spatial import KDTree  # type: ignore
from tqdm.auto import tqdm  # type: ignore

SUPPORTED_DOMAINS = {"alpha", "beta", "gamma", "delta", "epsilon"}
DOMAIN_2_THRESHOLD = {
    "beta": (0.6, 50),
    "delta": (0.6, 50),
    "epsilon": (0.6, 50),
    "gamma": (0.55, 50),
    "alpha": (0.35, 130),
}


@dataclass(eq=True)
class MappedRegion:
    """A dataclass to store information about a particular structural module"""

    module_id: str
    domain: str
    tmscore: float
    residues_mapping: dict[int, int]


# https://pymolwiki.org/index.php/Selection_Exists
def exists_in_pymol(pymol_cmd, sele):
    """
    A function to check presence of an object in pymol session
    """
    sess = pymol_cmd.get_session()
    for i in sess["names"]:
        if isinstance(i, list) and sele == i[0]:
            return True
    return False


def prepare_domain(pymol_cmd, domain_name: str) -> tuple:
    """
    Creates a domain object in a PyMOL session based on the provided domain name.

    :param pymol_cmd: The PyMOL command object used to interact with the PyMOL session
    :type pymol_cmd: pymol.Cmd

    :param domain_name: The name of the domain to be created in the PyMOL session
    :type domain_name: str

    :return: A tuple containing the modified PyMOL command object and the new domain name string
    :rtype: tuple
    """
    domain_2_standard = {dom_name_: f"{dom_name_}" for dom_name_ in SUPPORTED_DOMAINS}
    domain_2_standard.update(
        {
            "alpha": "1ps1",
            "beta": "5eat",
            "gamma": "3p5r",
            "delta": "P48449",
            "epsilon": "P48449",
        }
    )
    assert domain_name in domain_2_standard, f"Domain {domain_name} is not supported"
    required_file = domain_2_standard[domain_name]
    if not exists_in_pymol(pymol_cmd, required_file):
        if not os.path.exists(f"{required_file}.pdb"):
            raise FileNotFoundError
        pymol_cmd.load(f"{required_file}.pdb")

    if "_" in domain_name:
        selection_condition = " & chain A & ss H+S"
    else:
        selection_condition = {
            "alpha": " & chain A & ss H+S",
            "beta": " & resi 37-57+64-97+104-117+123-129+138-156+162-195+203-213+223-239 & chain A & ss H+S",
            "gamma": " & resi 138-151+157-171+185-222+233-248+258-275+281-304+313-339 & chain A & ss H+S",
            "delta": " & resi 73-87+385-399+401-403+405-421+454-470+480-493+531-547+553-570+585-599+610-622+633-638+649-662+667-680+707-722+727-729 & chain A & ss H+S",
            "epsilon": " & resi 103-115+123-134+151-164+171-183+191-200+213-217+226-228+231-246+254-263+268-270+273-277+291-306+309-330+337-351+356-371+376-378+510-515 & chain A & ss H+S",
        }[domain_name]
    domain_name_new = f"{domain_name}_domain_{uuid4()}"
    pymol_cmd.select(domain_name_new, f"{required_file} {selection_condition}")
    return pymol_cmd, domain_name_new


def compress_selection_list(selected_residues: list[int]) -> str:
    """
    Compresses a list of selected residues into a concise string representation.

    :param selected_residues: A list of residue numbers to be compressed

    :return: A string representing the compressed form of the residue list, with consecutive residues represented
             as ranges (e.g., "1-3+5+7-10")
    """
    sorted_residues = sorted(map(int, selected_residues))
    start_res = None
    intervals = []
    for res in sorted_residues:
        if start_res is None:
            start_res = res
        else:
            if prev_res + 1 != res:
                if start_res == prev_res:
                    intervals.append(f"{start_res}")
                else:
                    intervals.append(f"{start_res}-{prev_res}")
                start_res = res
        prev_res = res

    if start_res is not None:
        if start_res == prev_res:
            intervals.append(f"{start_res}")
        else:
            intervals.append(f"{start_res}-{prev_res}")
    return "+".join(intervals)


def get_secondary_structure_residues_set(str_name: str, pymol_cmd) -> set[str]:
    """
    Retrieves a set of secondary-structure residues from an object
    :param str_name: object ID in the pymol session
    :param pymol_cmd:
    """
    stored.residues_set = set()
    pymol_cmd.iterate(f"({str_name} & ss H+S)", "stored.residues_set.add(resi)")
    result = stored.residues_set.copy()
    stored.residues_set = None
    return result


def compute_full_mapping(
    domain_obj: str,
    larger_obj: str,
    residues_mapping: dict[int, int],
    file_2_all_residues: dict[str, set[str]],
) -> dict:
    """
    Computes a full mapping of residues from a larger object to a domain object based on the closest aligned residue pairs.

    :param domain_obj: The identifier of the domain object
    :param larger_obj: The identifier of the larger object
    :param residues_mapping: A dictionary mapping residues from the larger object to residues in the domain object
    :param file_2_all_residues: A dictionary mapping object identifiers to sets of residues in those objects

    :return: A dictionary representing the full mapping of residues from the larger object to the domain object
    """
    if len(residues_mapping) == 0:
        return {}
    obj_res_2_mapped_shift = {
        domain_res: int(res) - int(domain_res)
        for res, domain_res in residues_mapping.items()
    }
    kdtree = KDTree(
        np.array(list(map(int, obj_res_2_mapped_shift.keys()))).reshape(-1, 1)
    )
    shift_values = list(obj_res_2_mapped_shift.values())

    mapped_residues = set()
    _temp1, _temp2 = [], []

    start_mapped_res, prev_mapped_res = None, None
    start_dom_res, prev_dom_res = None, None
    domain_intervals = []
    mapped_intervals = []
    sorted_domain_residues = sorted(map(int, file_2_all_residues[domain_obj]))
    obj_residues_set = set(map(int, file_2_all_residues[larger_obj]))
    residues_mapping_full = {}
    for domain_res in sorted_domain_residues:
        #         print('domain_res', domain_res)
        if domain_res in obj_res_2_mapped_shift:
            shift = obj_res_2_mapped_shift[domain_res]
        else:
            _, closest_indices = kdtree.query(domain_res, k=2)
            shift = int(
                round(
                    np.mean(
                        [shift_values[closest_idx] for closest_idx in closest_indices]
                    )
                )
            )
        mapped_res = int(domain_res) + shift
        #         print('mapped_res', mapped_res)
        if mapped_res not in mapped_residues and mapped_res in obj_residues_set:
            #             print('added!')
            residues_mapping_full[mapped_res] = domain_res
            _temp1.append(domain_res)
            _temp2.append(mapped_res)
            mapped_residues.add(mapped_res)

            if start_mapped_res is None:
                start_mapped_res = mapped_res
                start_dom_res = domain_res
            else:
                if prev_mapped_res + 1 != mapped_res:
                    if start_mapped_res == prev_mapped_res:
                        mapped_intervals.append(f"{start_mapped_res}")
                    else:
                        mapped_intervals.append(f"{start_mapped_res}-{prev_mapped_res}")
                    start_mapped_res = mapped_res
                if prev_dom_res + 1 != domain_res:
                    if start_dom_res == prev_dom_res:
                        domain_intervals.append(f"{start_dom_res}")
                    else:
                        domain_intervals.append(f"{start_dom_res}-{prev_dom_res}")
                    start_dom_res = domain_res
            prev_mapped_res = mapped_res
            prev_dom_res = domain_res
    if start_mapped_res is not None:
        if start_mapped_res == prev_mapped_res:
            mapped_intervals.append(f"{start_mapped_res}")
        else:
            mapped_intervals.append(f"{start_mapped_res}-{prev_mapped_res}")
    if start_dom_res is not None:
        if start_dom_res == prev_dom_res:
            domain_intervals.append(f"{start_dom_res}")
        else:
            domain_intervals.append(f"{start_dom_res}-{prev_dom_res}")

    if len(domain_intervals) == 0 or len(mapped_intervals) == 0:
        return {}
    return residues_mapping_full


def get_super_res_alignment(
    larger_obj: str,
    domain_obj: str,
    file_2_all_residues: dict[str, set[str]],
    min_domain_fraction: float = 0.1,
    pymol_cmd=cmd,
) -> tuple[float, dict]:
    """
    Performs sequence-independent alignment and assigns all residues of the domain object to the larger object.

    :param larger_obj: The identifier of the larger object to align
    :param domain_obj: The identifier of the domain object to align
    :param file_2_all_residues: A dictionary mapping object identifiers to sets of residues in those objects
    :param min_domain_fraction: The minimum fraction of domain residues that must be aligned for a valid mapping, defaults to 0.1
    :param pymol_cmd: The PyMOL command object used to interact with the PyMOL session, defaults to cmd

    :return: A tuple containing the TM-score of the alignment and a dictionary representing the full residues mapping
             between the domain object and the larger object
    """
    if larger_obj in file_2_all_residues and len(file_2_all_residues[larger_obj]) == 0:
        if exists_in_pymol(pymol_cmd, larger_obj):
            pymol_cmd.delete(larger_obj)
        if larger_obj in file_2_all_residues:
            del file_2_all_residues[larger_obj]
        return -float("inf"), {}

    if not exists_in_pymol(pymol_cmd, larger_obj):
        if not os.path.exists(f"{larger_obj}.pdb"):
            raise FileNotFoundError
        pymol_cmd.load(f"{larger_obj}.pdb")
        file_2_all_residues[larger_obj] = get_secondary_structure_residues_set(
            larger_obj, pymol_cmd
        ).intersection(file_2_all_residues[larger_obj])

    allowed_residues = file_2_all_residues[larger_obj]

    # limiting the residues in accordance with file_2_all_residues
    current_residues = get_secondary_structure_residues_set(larger_obj, pymol_cmd)
    orig_obj_to_remove = None
    if current_residues != allowed_residues:
        orig_obj_to_remove = larger_obj
        # from now on larger_obj is a selection id
        larger_obj = str(uuid4())
        residues_permitted = current_residues.intersection(allowed_residues)
        pymol_cmd.select(
            larger_obj,
            f"{orig_obj_to_remove} & resi {compress_selection_list(list(map(int, residues_permitted)))} & chain A",
        )
        file_2_all_residues[larger_obj] = residues_permitted

    loaded_new_domain_obj = False
    if not exists_in_pymol(pymol_cmd, domain_obj):
        if not os.path.exists(f"{domain_obj}.pdb"):
            if domain_obj.replace("_domain", "") in SUPPORTED_DOMAINS:
                pymol_cmd, domain_obj = prepare_domain(
                    pymol_cmd, domain_obj.replace("_domain", "")
                )
                file_2_all_residues[domain_obj] = get_secondary_structure_residues_set(
                    domain_obj, pymol_cmd
                )
                loaded_new_domain_obj = True
            else:
                raise NotImplementedError(
                    f"Domain {domain_obj} not supported, {domain_obj}.pdb did not exist"
                )
        else:
            pymol_cmd.load(f"{domain_obj}.pdb")
            loaded_new_domain_obj = True
    aln_name, domain_obj_secondary_structure, larger_obj_secondary_structure = [
        str(uuid4()) for _ in range(3)
    ]
    needs_clone = (
        pymol_cmd.get_object_list(domain_obj)[0]
        == pymol_cmd.get_object_list(larger_obj)[0]
    )
    if needs_clone:
        object_name = pymol_cmd.get_object_list(domain_obj)[0]
        new_object_name = f"{object_name}_new"
        pymol_cmd.copy(new_object_name, object_name)
        pymol_cmd.select(
            domain_obj,
            f"{new_object_name} & resi {compress_selection_list(list(map(int, file_2_all_residues[domain_obj])))} & chain A",
        )

    pymol_cmd.select(domain_obj_secondary_structure, f"{domain_obj} & ss H+S")
    pymol_cmd.select(larger_obj_secondary_structure, f"{larger_obj} & ss H+S")
    try:
        tmscore: float = tmalign(
            domain_obj_secondary_structure,
            larger_obj_secondary_structure,
            object=aln_name,
            quiet=1,
            args=f"-L {len(file_2_all_residues[domain_obj])}",
        )
    except CmdException:
        pymol_cmd.delete(domain_obj_secondary_structure)
        pymol_cmd.delete(larger_obj_secondary_structure)
        pymol_cmd.delete(larger_obj)
        del file_2_all_residues[larger_obj]
        if orig_obj_to_remove is not None:
            pymol_cmd.delete(orig_obj_to_remove)
            del file_2_all_residues[orig_obj_to_remove]
        if needs_clone:
            pymol_cmd.select(
                domain_obj,
                f"{object_name} & resi {compress_selection_list(list(map(int, file_2_all_residues[domain_obj])))} & chain A",
            )
            pymol_cmd.delete(new_object_name)
        if loaded_new_domain_obj:
            pymol_cmd.delete(domain_obj)
            if domain_obj in file_2_all_residues:
                del file_2_all_residues[domain_obj]
        return -float("inf"), {}
    raw_aln = pymol_cmd.get_raw_alignment(aln_name)
    idx2resi: dict = {}
    pymol_cmd.iterate(
        aln_name, "idx2resi[model, index] = resi", space={"idx2resi": idx2resi}
    )
    residues_mapping = {}
    for (protein_id_1, protein_idx_1), (protein_id_2, protein_idx_2) in raw_aln:
        res_1 = idx2resi[(protein_id_1, protein_idx_1)]
        res_2 = idx2resi[(protein_id_2, protein_idx_2)]
        residues_mapping[res_1] = res_2
    if len(residues_mapping) < min_domain_fraction * len(
        file_2_all_residues[domain_obj]
    ):
        tmscore, residues_mapping_full = -float("inf"), {}
    else:
        residues_mapping_full = compute_full_mapping(
            domain_obj,
            larger_obj,
            residues_mapping,
            file_2_all_residues=file_2_all_residues,
        )

    pymol_cmd.delete(aln_name)
    pymol_cmd.delete(domain_obj_secondary_structure)
    pymol_cmd.delete(larger_obj_secondary_structure)
    pymol_cmd.delete(larger_obj)
    del file_2_all_residues[larger_obj]
    if orig_obj_to_remove is not None:
        pymol_cmd.delete(orig_obj_to_remove)
        del file_2_all_residues[orig_obj_to_remove]
    if needs_clone:
        pymol_cmd.select(
            domain_obj,
            f"{object_name} & resi {compress_selection_list(list(map(int, file_2_all_residues[domain_obj])))} & chain A",
        )
        pymol_cmd.delete(new_object_name)
    if loaded_new_domain_obj:
        pymol_cmd.delete(domain_obj)
        if domain_obj in file_2_all_residues:
            del file_2_all_residues[domain_obj]
    return tmscore, residues_mapping_full


def find_longest_continuous_segments(
    residues_subset: set, all_residues: set, max_allowed_gap: int = 5
) -> list[int]:
    """
    Identifies and returns the longest continuous segments of residues from a given subset.

    :param residues_subset: A set of residue numbers to evaluate for continuous segments
    :param all_residues: A set of all residue numbers available in the sequence
    :param max_allowed_gap: The maximum gap allowed between consecutive residues to still consider them part of a continuous segment, defaults to 5

    :return: A list of residue numbers representing the longest continuous segment found in the subset
    """
    res_continuous_candidates: list[list[int]] = [[]]
    prev_res = None
    allowed_prev_residues = None

    for res in sorted(map(int, residues_subset)):
        if prev_res is not None:
            allowed_prev_residues = {prev_res + 1}.union(
                {prev_res + 1 + i for i in range(max_allowed_gap)}
            )
        if (
            prev_res is not None
            and allowed_prev_residues is not None
            and res not in allowed_prev_residues
            and len(set(map(str, allowed_prev_residues)).intersection(all_residues))
        ):
            res_continuous_candidates.append([])
        res_continuous_candidates[-1].append(res)
        prev_res = res
    residues_candidates = []
    residues_len = -float("inf")
    for cand_residues in res_continuous_candidates:
        if len(cand_residues) > residues_len:
            residues_len = len(cand_residues)
            residues_candidates = cand_residues
    # filling in allowed gaps
    residues_final = []
    prev_res = None
    for res in residues_candidates:
        if prev_res is not None and prev_res + max_allowed_gap >= res:
            for filled_res in range(prev_res + 1, res):
                residues_final.append(filled_res)
        residues_final.append(res)
        prev_res = res
    return residues_final


def fill_short_gaps(residues_subset: set[int], max_allowed_gap: int = 5) -> list[int]:
    """
    A function including missing residues if they are inside short gaps of the residues_subset
    """
    residues_final = []
    prev_res = None
    for res in sorted(map(int, residues_subset)):
        if prev_res is not None and prev_res + max_allowed_gap >= res:
            for filled_res in range(prev_res + 1, res):
                residues_final.append(filled_res)
        residues_final.append(res)
        prev_res = res
    return residues_final


def get_regions_of_interest(
    domains: set[str], file_2_mapped_regions: dict[str, list[MappedRegion]]
) -> list[tuple[str, MappedRegion]]:
    """
    Filters the dictionary of mapped regions to preserve only those regions belonging to specified domain types.

    :param domains: A set of domain types to filter the mapped regions by
    :param file_2_mapped_regions: A dictionary mapping filenames to lists of MappedRegion objects

    :return: A list of tuples, each containing a filename and a MappedRegion object corresponding to the specified domain types
    """
    all_mapped_regions_of_interest = []
    for filename, mapped_regions in file_2_mapped_regions.items():
        for region in mapped_regions:
            if region.domain in domains:
                all_mapped_regions_of_interest.append((filename, region))
    return all_mapped_regions_of_interest


def get_pairwise_tmscore(
    pymol_cmd,
    module_1: tuple[str, MappedRegion],
    module_2: tuple[str, MappedRegion],
    file_2_all_residues: dict[str, set],
) -> float:
    """
    Computes the pairwise TM-score between two domains.

    :param pymol_cmd: The PyMOL command object used to interact with the PyMOL session
    :param module_1: A tuple containing the filename and MappedRegion object for the first domain
    :param module_2: A tuple containing the filename and MappedRegion object for the second domain
    :param file_2_all_residues: A dictionary mapping filenames to sets of all residues in those files

    :return: The TM-score representing the structural similarity between the two domains
    """
    filename_1, region_1 = module_1
    filename_2, region_2 = module_2
    selection_1_id, selection_2_id = str(uuid4()), str(uuid4())
    for filename in [filename_1, filename_2]:
        if not exists_in_pymol(pymol_cmd, filename):
            if not os.path.exists(f"{filename}.pdb"):
                raise FileNotFoundError
            pymol_cmd.load(f"{filename}.pdb")

    region_residues_1 = set(fill_short_gaps(set(region_1.residues_mapping.keys())))
    try:
        pymol_cmd.select(
            f"{filename_1}_{selection_1_id}",
            f"{filename_1} & resi {compress_selection_list(list(region_residues_1))} & chain A & ss H+S",
        )
        file_2_all_residues[f"{filename_1}_{selection_1_id}"] = set(
            map(str, region_residues_1)
        )

        region_residues_2 = set(fill_short_gaps(set(region_2.residues_mapping.keys())))
        pymol_cmd.select(
            f"{filename_2}_{selection_2_id}",
            f"{filename_2} & resi {compress_selection_list(list(region_residues_2))} & chain A & ss H+S",
        )
    except CmdException:
        pymol_cmd.delete(filename_2)
        pymol_cmd.delete(filename_1)
        if exists_in_pymol(pymol_cmd, f"{filename_1}_{selection_1_id}"):
            pymol_cmd.delete(f"{filename_1}_{selection_1_id}")
        if f"{filename_1}_{selection_1_id}" in file_2_all_residues:
            del file_2_all_residues[f"{filename_1}_{selection_1_id}"]
        return -float("inf")
    file_2_all_residues[f"{filename_2}_{selection_2_id}"] = set(
        map(str, region_residues_2)
    )
    is_first_shorter = len(region_residues_1) < len(region_residues_2)
    if is_first_shorter:
        tmscore, _ = get_super_res_alignment(
            f"{filename_2}_{selection_2_id}",
            f"{filename_1}_{selection_1_id}",
            file_2_all_residues=file_2_all_residues,
            pymol_cmd=pymol_cmd,
        )
    else:
        tmscore, _ = get_super_res_alignment(
            f"{filename_1}_{selection_1_id}",
            f"{filename_2}_{selection_2_id}",
            file_2_all_residues=file_2_all_residues,
            pymol_cmd=pymol_cmd,
        )
    pymol_cmd.delete(filename_2)
    pymol_cmd.delete(filename_1)
    if is_first_shorter:
        pymol_cmd.delete(f"{filename_1}_{selection_1_id}")
        del file_2_all_residues[f"{filename_1}_{selection_1_id}"]
    else:
        pymol_cmd.delete(f"{filename_2}_{selection_2_id}")
        del file_2_all_residues[f"{filename_2}_{selection_2_id}"]
    return tmscore


def compute_region_distances(
    i: int,
    regions: list[tuple[str, MappedRegion]],
    file_2_all_residues: dict[str, set],
    save_output: bool = True,
    output_name: str = "all",
    precomputed_scores: dict[tuple[str, str], float] = None,
) -> list[tuple[int, int, float]]:
    """
    Computes pairwise alignment TM-scores between the i-th structural domain in the list `regions` and all subsequent domains.
    This function effectively computes a single row of an upper-triangle distance matrix.

    :param i: The index of the structural domain in the list `regions` to be compared
    :param regions: A list of tuples, where each tuple contains a filename and a MappedRegion object
    :param file_2_all_residues: A dictionary mapping filenames to sets of all residues in those files
    :param save_output: Whether to save the computed results to a file, defaults to True
    :param output_name: The base name for the output file, defaults to "all"

    :return: A list of tuples, each containing the index of the first region, the index of the second region,
             and the computed TM-score between the two regions
    """
    results = []
    region_1 = regions[i]
    results_path = f"{output_name}_tm_region_very_conf_{i}_{region_1[1].module_id}.pkl"
    if os.path.exists(results_path):
        return []
    j = i + 1
    for region_2 in regions[j:]:
        if precomputed_scores is not None:
            if (region_1[1].module_id, region_2[1].module_id) in precomputed_scores:
                dist = precomputed_scores[
                    (region_1[1].module_id, region_2[1].module_id)
                ]
            elif (region_2[1].module_id, region_1[1].module_id) in precomputed_scores:
                dist = precomputed_scores[
                    (region_2[1].module_id, region_1[1].module_id)
                ]
            else:
                dist = get_pairwise_tmscore(
                    cmd, region_1, region_2, file_2_all_residues
                )
        else:
            dist = get_pairwise_tmscore(cmd, region_1, region_2, file_2_all_residues)
        results.append((i, j, dist))
        j += 1
    if save_output:
        with open(results_path, "wb") as file:
            pickle.dump(results, file)

    return results


def get_all_residues_per_file(pdb_files: list[Path], pymol_cmd) -> dict[str, set[str]]:
    """
    Computes a set of all residues in each PDB file provided.

    :param pdb_files: A list of Path objects representing the PDB files to be processed
    :param pymol_cmd: The PyMOL command object used to interact with the PyMOL session

    :return: A dictionary mapping each PDB filename (without extension) to a set of all residues found in that file
    """
    file_2_all_residues = {}
    for filepath in tqdm(pdb_files, desc="All residues"):
        str_name = filepath.stem
        pymol_cmd.load(str(filepath))
        all_residues = get_secondary_structure_residues_set(str_name, pymol_cmd)
        if len(all_residues):
            file_2_all_residues[str_name] = all_residues
        pymol_cmd.delete(str_name)
    return file_2_all_residues


def get_alignments(
    pdb_filepaths: list[Path],
    domain_name: str,
    file_2_current_residues: dict[str, set[str]],
    tmscore_threshold: Optional[float] = None,
    mapping_size_threshold: Optional[int] = None,
    n_jobs: int = 8,
) -> dict[str, list[tuple[float, dict]]]:
    """
    Computes alignments of a specified domain object to all structures in the provided list of PDB file paths.

    :param pdb_filepaths: A list of Path objects representing the PDB files to be aligned
    :param domain_name: The name of the domain to align against the structures
    :param file_2_current_residues: A dictionary mapping filenames to sets of current residues available in those files
    :param tmscore_threshold: An optional TM-score threshold; alignments with scores below this threshold will be discarded
    :param mapping_size_threshold: An optional minimum size for the residue mapping; alignments with fewer residues will be discarded
    :param n_jobs: The number of parallel jobs to use for the alignment computation, defaults to 8

    :return: A dictionary mapping each PDB filename (without extension) to a list of tuples,
             where each tuple contains a TM-score and a dictionary representing the residue mapping for that alignment
    """
    align_partial = partial(
        get_super_res_alignment,
        domain_obj=f"{domain_name}_domain",
        file_2_all_residues=file_2_current_residues,
    )
    pdb_filenames = [filepath.stem for filepath in pdb_filepaths]
    with Pool(n_jobs) as pool:
        list_of_alignment_results = pool.map(align_partial, pdb_filenames)
    file_2_tmscore_residues = defaultdict(list)

    for pdb_path, (tmscore, residues_mapping) in zip(
        pdb_filepaths, list_of_alignment_results
    ):
        if (
            residues_mapping is not None
            and (tmscore_threshold is None or tmscore >= tmscore_threshold)
            and (
                mapping_size_threshold is None
                or len(residues_mapping) >= mapping_size_threshold
            )
        ):
            file_2_tmscore_residues[pdb_path.stem].append((tmscore, residues_mapping))
    return file_2_tmscore_residues


def get_mapped_regions_per_file(
    domain_2_file_2_tmscore_residues: dict[str, dict[str, list]],
    domain_2_thresholds: dict[str, tuple[float, int]],
) -> dict[str, list[MappedRegion]]:
    """
    Detects reliable alignments of domains per file based on TM-score thresholds provided in `domain_2_thresholds`.

    :param domain_2_file_2_tmscore_residues: A dictionary mapping domain names to another dictionary that maps filenames
                                             to lists of alignment results, where each result is a tuple containing a TM-score
                                             and a residue mapping
    :param domain_2_thresholds: A dictionary mapping domain names to a tuple, where each tuple contains a TM-score detection threshold
                                and a minimum mapping size threshold

    :return: A dictionary mapping filenames to lists of MappedRegion objects representing the detected reliable domain alignments
    """
    file_2_mapped_regions = defaultdict(list)
    for domain, (
        tmscore_threshold,
        mapping_size_threshold,
    ) in domain_2_thresholds.items():
        if domain in domain_2_file_2_tmscore_residues:
            for filename, mappings in domain_2_file_2_tmscore_residues[domain].items():
                for (tmscore, residues_mapping) in mappings:
                    if (
                        tmscore > tmscore_threshold
                        and len(residues_mapping) >= mapping_size_threshold
                    ):
                        file_2_mapped_regions[filename].append(
                            MappedRegion(
                                module_id=f"{filename}_{domain}_{int(tmscore * 1000)}",
                                domain=domain,
                                tmscore=tmscore,
                                residues_mapping=residues_mapping,
                            )
                        )
    return file_2_mapped_regions


def get_remaining_residues_per_file(
    all_residues: set[str],
    mapped_regions: list[MappedRegion],
) -> set[str]:
    """
    Function retrieving currently unassigned residues
    """
    mapped_residues: set[int] = set()
    for mapping in mapped_regions:
        mapped_residues = mapped_residues.union(set(mapping.residues_mapping.keys()))
    return all_residues.difference({str(val) for val in mapped_residues})


def get_remaining_residues(
    file_2_mapped_regions: dict[str, list[MappedRegion]],
    file_2_previously_remaining_residues: dict[str, set[str]],
) -> dict[str, set[str]]:
    """
    Function retrieving currently unassigned residues for each file from the `file_2_previously_remaining_residues` keys
    """
    file_2_remaining_residues = {}
    for filename, all_residues in file_2_previously_remaining_residues.items():
        mapped_regions = file_2_mapped_regions.get(filename, [])
        file_2_remaining_residues[filename] = get_remaining_residues_per_file(
            all_residues, mapped_regions
        )
    return file_2_remaining_residues


def plot_aligned_domains(
    file_2_tmscore_residues: dict[str, list[tuple[float, dict[str, str]]]],
    title: str = "",
    save_path: Optional[str | Path] = None,
):
    """
    Helper function plotting TM-scores of detected domains on x-axis and
    the number of residues assigned to the domain object on y-axis
    """
    plt.figure(figsize=(17, 9))
    all_tmscores_and_mappings: list = sum(file_2_tmscore_residues.values(), [])
    results_of_mapping = [
        (tmscore, len(mapping))
        for tmscore, mapping in all_tmscores_and_mappings
        if mapping is not None
    ]
    mapping_lenghts = list(map(lambda x: x[1], results_of_mapping))
    plt.scatter(list(map(lambda x: x[0], results_of_mapping)), mapping_lenghts)
    plt.xticks(np.arange(0, 1, 0.05), rotation=90)
    plt.yticks(np.arange(min(mapping_lenghts) - 10, max(mapping_lenghts) + 10, 5))
    plt.xlabel("TM-score", fontsize=11)
    plt.ylabel("Number of residues assigned to the domain", fontsize=11)
    plt.title(title, fontsize=14)
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def detect_second_encounter_of_domain(
    domain_name: str,
    domain_2_threshold: dict[str, tuple[float, int]],
    domain_2_file_2_tmscore_residues: dict[str, dict[str, list]],
    pdb_files: list[Path],
    file_2_remaining_residues: dict[str, set[str]],
) -> tuple[dict[str, list[MappedRegion]], dict[str, set[str]]]:
    """
    Function to detect second instance of the already present domain-type, e.g., in alpha-alpha architectures
    """
    (tmscore_threshold, mapping_size_threshold) = domain_2_threshold[domain_name]
    file_2_tmscore_residues = get_alignments(
        pdb_files,
        domain_name,
        file_2_remaining_residues,
        tmscore_threshold,
        mapping_size_threshold,
    )
    for filename, mappings in tqdm(file_2_tmscore_residues.items()):
        for (tmscore, residues_mapping) in mappings:
            if (
                tmscore >= tmscore_threshold
                and len(residues_mapping) >= mapping_size_threshold
            ):
                domain_2_file_2_tmscore_residues[domain_name][filename].append(
                    (tmscore, residues_mapping)
                )
    file_2_mapped_regions = get_mapped_regions_per_file(
        domain_2_file_2_tmscore_residues, domain_2_threshold
    )
    file_2_remaining_residues = get_remaining_residues(
        file_2_mapped_regions, file_2_remaining_residues
    )
    return file_2_mapped_regions, file_2_remaining_residues


def get_currently_longest_unmapped_regions(
    file_2_remaining_residues: dict[str, set[str]],
    file_2_all_residues: dict[str, set[str]],
) -> dict[str, list[int]]:
    """
    Function computing the longest sequence of unmapped residues per file
    """
    file_2_longest_unmapped_region = {}
    for filename, unmapped_residues in file_2_remaining_residues.items():
        file_2_longest_unmapped_region[filename] = find_longest_continuous_segments(
            unmapped_residues, file_2_all_residues[filename]
        )
    return file_2_longest_unmapped_region


def return_short_enough_segments(segment: list[int], max_allowed_length: int):
    """
    A recursive function splitting `segment` of residues into short intervals of max length `max_allowed_length`
    """
    if max(segment) - min(segment) <= max_allowed_length:
        return [segment]
    mid_index = (max(segment) + min(segment)) / 2
    return return_short_enough_segments(
        [res for res in segment if res <= mid_index], max_allowed_length
    ) + return_short_enough_segments(
        [res for res in segment if res > mid_index], max_allowed_length
    )


def find_continuous_segments_longer_than(
    residues_subset: set[str],
    min_secondary_struct_len: int = 40,
    max_allowed_gap: int = 5,
) -> list[list[int]]:
    """
    Function computing continuous intervals of secondary-structure residues,
     such that each interval has length at least `min_secondary_struct_len`
     (these continuous segments might have some residues missing,
     but not more that `max_allowed_gap` residues consequently)
    """
    res_continuous_segments: list[list[int]] = [[]]
    prev_res = None
    for res in sorted(map(int, residues_subset)):
        if prev_res is not None:
            allowed_prev_residues = {prev_res + 1}.union(
                {prev_res + 1 + i for i in range(max_allowed_gap)}
            )
            if res not in allowed_prev_residues:
                if (
                    max(res_continuous_segments[-1]) - min(res_continuous_segments[-1])
                    >= min_secondary_struct_len
                ):
                    res_continuous_segments.append([])
                else:
                    res_continuous_segments[-1] = []
        res_continuous_segments[-1].append(res)
        prev_res = res
    if (
        len(res_continuous_segments[-1]) == 0
        or max(res_continuous_segments[-1]) - min(res_continuous_segments[-1])
        < min_secondary_struct_len
    ):
        res_continuous_segments = res_continuous_segments[:-1]
    return res_continuous_segments


def get_mapped_regions_with_surroundings(
    filename: str,
    file_2_all_residues: dict[str, set[str]],
    filename_2_known_regions: dict[str, list[MappedRegion]],
    helix_sheet_dist_threshold: float = 17,
    max_allowed_segment_len: int = 7,
) -> list[MappedRegion]:
    """
    A function detecting unassigned parts of secondary structure which are close a particular domain in 3D space
    """
    already_mapped_residues: set[int] = set()
    for mapped_region in filename_2_known_regions[filename]:
        already_mapped_residues = already_mapped_residues.union(
            set(mapped_region.residues_mapping.keys())
        )
    remaining_residues = {int(i) for i in file_2_all_residues[filename]}.difference(
        already_mapped_residues
    )

    if not exists_in_pymol(cmd, filename):
        if not os.path.exists(f"{filename}.pdb"):
            raise FileNotFoundError(f"{filename}.pdb")
        cmd.load(f"{filename}.pdb")

    # for each mapped region, compute alpha helixes
    region_i_2_segments = defaultdict(list)
    for mapped_region_i, mapped_region in enumerate(filename_2_known_regions[filename]):
        region_continuous_segments = find_continuous_segments_longer_than(
            set(map(str, mapped_region.residues_mapping.keys())),
            min_secondary_struct_len=5,
            max_allowed_gap=2,
        )
        segment_i = 0
        for region_segment_master in region_continuous_segments:
            for region_segment in return_short_enough_segments(
                region_segment_master, max_allowed_length=max_allowed_segment_len
            ):
                segment_name = f"bigger_selection_{mapped_region_i}_{segment_i}"
                cmd.select(
                    segment_name,
                    f"{filename} & resi {compress_selection_list(region_segment)} & chain A",
                )
                segment_i += 1
                region_i_2_segments[mapped_region_i].append(segment_name)
    mapped_region_2_added_residues = defaultdict(list)

    # detect secondary structure residue segments in the unmapped parts
    remaining_residues_segments = find_continuous_segments_longer_than(
        set(map(str, remaining_residues)), min_secondary_struct_len=0, max_allowed_gap=1
    )
    for residue_segment_remaining_master in remaining_residues_segments:
        for residue_segment_remaining in return_short_enough_segments(
            residue_segment_remaining_master, max_allowed_length=max_allowed_segment_len
        ):
            cmd.select(
                "small_selection",
                f"{filename} & resi {compress_selection_list(residue_segment_remaining)} & chain A",
            )
            min_dist = float("inf")
            closest_region_i = None
            all_dists_with_regions = []
            for mapped_region_i, mapped_region in enumerate(
                filename_2_known_regions[filename]
            ):
                region_dist_min = float("inf")
                for segment_selection in region_i_2_segments[mapped_region_i]:
                    distance = cmd.distance(
                        "dist",
                        selection1="small_selection",
                        selection2=segment_selection,
                        mode=4,
                    )
                    region_dist_min = min(region_dist_min, distance)
                    cmd.delete("dist")
                all_dists_with_regions.append((region_dist_min, mapped_region_i))
                if region_dist_min < min_dist:
                    min_dist = region_dist_min
                    closest_region_i = mapped_region_i
            cmd.delete("small_selection")

            if min_dist < helix_sheet_dist_threshold:
                if len(all_dists_with_regions) >= 2:
                    # leave unassigned if it is similarly close to two different regions
                    second_closest_dist, second_closest_region_i = min(
                        [
                            (dist, region)
                            for (dist, region) in all_dists_with_regions
                            if dist > min_dist
                        ],
                        key=lambda x: x[0],
                    )
                    if (
                        second_closest_region_i == closest_region_i
                        or min_dist < 0.9 * second_closest_dist
                    ):
                        mapped_region_2_added_residues[closest_region_i].extend(
                            residue_segment_remaining
                        )
                else:
                    mapped_region_2_added_residues[closest_region_i].extend(
                        residue_segment_remaining
                    )

    # cleaning RAM to avoid pymol slow-down
    for added_segments in region_i_2_segments.values():
        for segment_name in added_segments:
            cmd.delete(segment_name)
    new_mapped_regions = []
    for mapped_region_i, mapped_region_init in enumerate(
        filename_2_known_regions[filename]
    ):
        new_residues_mapping = mapped_region_init.residues_mapping.copy()
        for newly_assigned_residue in mapped_region_2_added_residues[mapped_region_i]:
            new_residues_mapping[int(newly_assigned_residue)] = -1
        new_mapped_regions.append(
            MappedRegion(
                module_id=mapped_region_init.module_id,
                domain=mapped_region_init.domain,
                tmscore=mapped_region_init.tmscore,
                residues_mapping=new_residues_mapping,
            )
        )
    # if there's file, then we free RAM cause we can read the file back
    if os.path.exists(f"{filename}.pdb"):
        cmd.delete(filename)
    return new_mapped_regions


def get_mapped_regions_with_surroundings_parallel(
    pdb_filepaths: list[Path | str],
    file_2_all_residues: dict[str, set[str]],
    filename_2_known_regions: dict[str, list[MappedRegion]],
    n_jobs: int = 8,
) -> dict[str, list[MappedRegion]]:
    """
    A function for detecting unassigned parts of secondary structure which are close a particular domain in 3D space
    for all files in parallel
    """
    get_mapped_regions_with_surroundings_partial = partial(
        get_mapped_regions_with_surroundings,
        file_2_all_residues=file_2_all_residues,
        filename_2_known_regions=filename_2_known_regions,
    )
    pdb_filenames = [
        filepath.stem if isinstance(filepath, Path) else filepath.replace(".pdb", "")
        for filepath in pdb_filepaths
    ]
    with Pool(n_jobs) as pool:
        list_of_new_mapped_regions = pool.map(
            get_mapped_regions_with_surroundings_partial, pdb_filenames
        )
    filename_2_known_regions_completed = dict(
        zip(pdb_filenames, list_of_new_mapped_regions)
    )
    return filename_2_known_regions_completed
