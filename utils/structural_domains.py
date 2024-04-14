import os
import pickle
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from random import sample
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
from pymol import CmdException, cmd, stored
from scipy.spatial import KDTree
from tqdm.auto import tqdm, trange

try:
    from pymol import tmalign
except:
    pass
import argparse
import sys

import pandas as pd

SUPPORTED_DOMAINS = {"alpha", "beta", "gamma", "delta", "epsilon"}


@dataclass(eq=True)
class MappedRegion:
    module_id: str
    domain: str
    tmscore: float
    residues_mapping: dict


@dataclass
class MappedRegionWithNeighbourhood:
    mapped_region: MappedRegion
    close_residues: set


# https://pymolwiki.org/index.php/Selection_Exists
def exists_in_pymol(cmd, sele):
    sess = cmd.get_session()
    for i in sess["names"]:
        if isinstance(i, list) and sele == i[0]:
            return True
    return False


def prepare_domain(cmd, domain_name):
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
    if not exists_in_pymol(cmd, required_file):
        if not os.path.exists(f"{required_file}.pdb"):
            raise FileNotFoundError
        cmd.load(f"{required_file}.pdb")

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
    cmd.select(domain_name_new, f"{required_file} {selection_condition}")
    return cmd, domain_name_new


def compress_selection_list(selected_residues):
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


def get_secondary_structure_residues_set(str_name, cmd):
    stored.residues_set = set()
    cmd.iterate(f"({str_name} & ss H+S)", "stored.residues_set.add(resi)")
    result = stored.residues_set.copy()
    stored.residues_set = None
    return result


def compute_full_mapping(domain_obj, larger_obj, residues_mapping, file_2_all_residues):
    if len(residues_mapping) == 0:
        return dict()
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
    residues_mapping_full = dict()
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
        return dict()
    return residues_mapping_full


def get_super_res_alignment(
    larger_obj, domain_obj, file_2_all_residues, min_domain_fraction=0.1, cmd=cmd
):
    if larger_obj in file_2_all_residues and len(file_2_all_residues[larger_obj]) == 0:
        if exists_in_pymol(cmd, larger_obj):
            cmd.delete(larger_obj)
        if larger_obj in file_2_all_residues:
            del file_2_all_residues[larger_obj]
        return float("inf"), None

    if not exists_in_pymol(cmd, larger_obj):
        if not os.path.exists(f"{larger_obj}.pdb"):
            raise FileNotFoundError
        cmd.load(f"{larger_obj}.pdb")
        file_2_all_residues[larger_obj] = get_secondary_structure_residues_set(
            larger_obj, cmd
        ).intersection(file_2_all_residues[larger_obj])

    allowed_residues = file_2_all_residues[larger_obj]

    # limiting the residues in accordance with file_2_all_residues
    current_residues = get_secondary_structure_residues_set(larger_obj, cmd)
    orig_obj_to_remove = None
    if current_residues != allowed_residues:
        orig_obj_to_remove = larger_obj
        # from now on larger_obj is a selection id
        larger_obj = uuid4()
        residues_permitted = current_residues.intersection(allowed_residues)
        cmd.select(
            larger_obj,
            f"{orig_obj_to_remove} & resi {compress_selection_list(residues_permitted)} & chain A",
        )
        file_2_all_residues[larger_obj] = residues_permitted

    loaded_new_domain_obj = False
    if not exists_in_pymol(cmd, domain_obj):
        if not os.path.exists(f"{domain_obj}.pdb"):
            if domain_obj.replace("_domain", "") in SUPPORTED_DOMAINS:
                cmd, domain_obj = prepare_domain(cmd, domain_obj.replace("_domain", ""))
                file_2_all_residues[domain_obj] = get_secondary_structure_residues_set(
                    domain_obj, cmd
                )
                loaded_new_domain_obj = True
            else:
                raise NotImplementedError(
                    f"Domain {domain_obj} not supported, {domain_obj}.pdb did not exist"
                )
        else:
            cmd.load(f"{domain_obj}.pdb")
            loaded_new_domain_obj = True
    aln_name, domain_obj_secondary_structure, larger_obj_secondary_structure = [
        uuid4() for _ in range(3)
    ]
    needs_clone = (
        cmd.get_object_list(domain_obj)[0] == cmd.get_object_list(larger_obj)[0]
    )
    if needs_clone:
        object_name = cmd.get_object_list(domain_obj)[0]
        new_object_name = f"{object_name}_new"
        cmd.copy(new_object_name, object_name)
        cmd.select(
            domain_obj,
            f"{new_object_name} & resi {compress_selection_list(file_2_all_residues[domain_obj])} & chain A",
        )

    cmd.select(domain_obj_secondary_structure, f"{domain_obj} & ss H+S")
    cmd.select(larger_obj_secondary_structure, f"{larger_obj} & ss H+S")
    try:
        tmscore = tmalign.tmalign(
            domain_obj_secondary_structure,
            larger_obj_secondary_structure,
            object=aln_name,
            quiet=1,
            args=f"-L {len(file_2_all_residues[domain_obj])}",
        )
    except CmdException:
        cmd.delete(domain_obj_secondary_structure)
        cmd.delete(larger_obj_secondary_structure)
        cmd.delete(larger_obj)
        del file_2_all_residues[larger_obj]
        if orig_obj_to_remove is not None:
            cmd.delete(orig_obj_to_remove)
            del file_2_all_residues[orig_obj_to_remove]
        if needs_clone:
            cmd.select(
                domain_obj,
                f"{object_name} & resi {compress_selection_list(file_2_all_residues[domain_obj])} & chain A",
            )
            cmd.delete(new_object_name)
        if loaded_new_domain_obj:
            cmd.delete(domain_obj)
            if domain_obj in file_2_all_residues:
                del file_2_all_residues[domain_obj]
        return float("inf"), None
    raw_aln = cmd.get_raw_alignment(aln_name)
    idx2resi = {}
    cmd.iterate(aln_name, "idx2resi[model, index] = resi", space={"idx2resi": idx2resi})
    residues_mapping = dict()
    for (protein_id_1, protein_idx_1), (protein_id_2, protein_idx_2) in raw_aln:
        res_1 = idx2resi[(protein_id_1, protein_idx_1)]
        res_2 = idx2resi[(protein_id_2, protein_idx_2)]
        residues_mapping[res_1] = res_2
    if len(residues_mapping) < min_domain_fraction * len(
        file_2_all_residues[domain_obj]
    ):
        tmscore, residues_mapping_full = float("inf"), None
    else:
        residues_mapping_full = compute_full_mapping(
            domain_obj,
            larger_obj,
            residues_mapping,
            file_2_all_residues=file_2_all_residues,
        )

    cmd.delete(aln_name)
    cmd.delete(domain_obj_secondary_structure)
    cmd.delete(larger_obj_secondary_structure)
    cmd.delete(larger_obj)
    del file_2_all_residues[larger_obj]
    if orig_obj_to_remove is not None:
        cmd.delete(orig_obj_to_remove)
        del file_2_all_residues[orig_obj_to_remove]
    if needs_clone:
        cmd.select(
            domain_obj,
            f"{object_name} & resi {compress_selection_list(file_2_all_residues[domain_obj])} & chain A",
        )
        cmd.delete(new_object_name)
    if loaded_new_domain_obj:
        cmd.delete(domain_obj)
        if domain_obj in file_2_all_residues:
            del file_2_all_residues[domain_obj]
    return tmscore, residues_mapping_full


def find_longest_continuous_segment(residues_subset, all_residues, max_allowed_gap=5):
    res_continuous_candidates = [[]]
    prev_res = None
    for res in sorted(map(int, residues_subset)):
        if prev_res is not None:
            allowed_prev_residues = {prev_res + 1}.union(
                {prev_res + 1 + i for i in range(max_allowed_gap)}
            )
        #             print('allowed_prev_residues', allowed_prev_residues, 'res', res)
        if (
            prev_res is not None
            and res not in allowed_prev_residues
            and len(set(map(str, allowed_prev_residues)).intersection(all_residues))
        ):
            #             print('started new')
            res_continuous_candidates.append([])
        res_continuous_candidates[-1].append(res)
        prev_res = res
    residues_candidates = None
    residues_len = -float("inf")
    for cand_residues in res_continuous_candidates:
        if len(cand_residues) > residues_len:
            residues_len = len(cand_residues)
            residues_candidates = cand_residues
    # filling in allowed gaps
    residues_final = []
    prev_res = None
    #     print('residues_candidates', residues_candidates)
    for res in residues_candidates:
        if prev_res is not None and prev_res + max_allowed_gap >= res:
            for filled_res in range(prev_res + 1, res):
                residues_final.append(filled_res)
        residues_final.append(res)
        prev_res = res
    #     print('residues_final', residues_final)
    return residues_final


def fill_short_gaps(residues_subset, max_allowed_gap=5):
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
    domains: set, file_2_mapped_regions: dict
) -> list[tuple[str, MappedRegion]]:
    all_mapped_regions_of_interest = []
    for filename, mapped_regions in file_2_mapped_regions.items():
        for region in mapped_regions:
            if region.domain in domains:
                all_mapped_regions_of_interest.append((filename, region))
    return all_mapped_regions_of_interest


def get_pairwise_tmscore(
    cmd,
    module_1: tuple[str, MappedRegion],
    module_2: tuple[str, MappedRegion],
    file_2_all_residues,
) -> float:
    filename_1, region_1 = module_1
    filename_2, region_2 = module_2
    selection_1_id, selection_2_id = uuid4(), uuid4()
    for filename in [filename_1, filename_2]:
        if not exists_in_pymol(cmd, filename):
            if not os.path.exists(f"{filename}.pdb"):
                raise FileNotFoundError
            cmd.load(f"{filename}.pdb")

    region_residues_1 = set(fill_short_gaps(region_1.residues_mapping.keys()))
    try:
        cmd.select(
            f"{filename_1}_{selection_1_id}",
            f"{filename_1} & resi {compress_selection_list(region_residues_1)} & chain A & ss H+S",
        )
        file_2_all_residues[f"{filename_1}_{selection_1_id}"] = set(
            map(str, region_residues_1)
        )

        region_residues_2 = set(fill_short_gaps(region_2.residues_mapping.keys()))
        cmd.select(
            f"{filename_2}_{selection_2_id}",
            f"{filename_2} & resi {compress_selection_list(region_residues_2)} & chain A & ss H+S",
        )
    except CmdException:
        cmd.delete(filename_2)
        cmd.delete(filename_1)
        if exists_in_pymol(cmd, f"{filename_1}_{selection_1_id}"):
            cmd.delete(f"{filename_1}_{selection_1_id}")
        if f"{filename_1}_{selection_1_id}" in file_2_all_residues:
            del file_2_all_residues[f"{filename_1}_{selection_1_id}"]
        return float("inf")
    file_2_all_residues[f"{filename_2}_{selection_2_id}"] = set(
        map(str, region_residues_2)
    )
    is_first_shorter = len(region_residues_1) < len(region_residues_2)
    if is_first_shorter:
        tmscore, _ = get_super_res_alignment(
            f"{filename_2}_{selection_2_id}",
            f"{filename_1}_{selection_1_id}",
            file_2_all_residues=file_2_all_residues,
            cmd=cmd,
        )
    else:
        tmscore, _ = get_super_res_alignment(
            f"{filename_1}_{selection_1_id}",
            f"{filename_2}_{selection_2_id}",
            file_2_all_residues=file_2_all_residues,
            cmd=cmd,
        )
    cmd.delete(filename_2)
    cmd.delete(filename_1)
    if is_first_shorter:
        cmd.delete(f"{filename_1}_{selection_1_id}")
        del file_2_all_residues[f"{filename_1}_{selection_1_id}"]
    else:
        cmd.delete(f"{filename_2}_{selection_2_id}")
        del file_2_all_residues[f"{filename_2}_{selection_2_id}"]
    return tmscore


def compute_region_distances(
    i: int,
    regions: list[tuple[str, MappedRegion]],
    file_2_all_residues,
    save_output=True,
    output_name="all",
):
    results = []
    region_1 = regions[i]
    results_path = f"{output_name}_tm_region_very_conf_{i}_{region_1[1].module_id}.pkl"
    if os.path.exists(results_path):
        print("path existed")
        return
    j = i + 1
    for region_2 in regions[j:]:
        dist = get_pairwise_tmscore(cmd, region_1, region_2, file_2_all_residues)
        results.append((i, j, dist))
        j += 1
    print(f"Done! Storing {results_path}")
    if save_output:
        with open(results_path, "wb") as f:
            pickle.dump(results, f)

    return results


def get_all_residues_per_file(pdb_files, cmd):
    file_2_all_residues = dict()
    for filepath in tqdm(pdb_files, desc="All residues"):
        str_name = filepath.stem
        cmd.load(str(filepath))
        all_residues = get_secondary_structure_residues_set(str_name, cmd)
        if len(all_residues):
            file_2_all_residues[str_name] = all_residues
        cmd.delete(str_name)
    return file_2_all_residues


def get_alignments(
    pdb_filepaths,
    domain_name,
    file_2_current_residues,
    tmscore_threshold=None,
    mapping_size_threshold=None,
    n_jobs=8,
):
    align_partial = partial(
        get_super_res_alignment,
        domain_obj=f"{domain_name}_domain",
        file_2_all_residues=file_2_current_residues,
    )
    pdb_filenames = [filepath.stem for filepath in pdb_filepaths]
    with Pool(n_jobs) as p:
        list_of_alignment_results = p.map(align_partial, pdb_filenames)
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


def get_mapped_regions_per_file(domain_2_file_2_tmscore_residues, domain_2_thresholds):
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
    mapped_regions: list[MappedRegionWithNeighbourhood | MappedRegion],
) -> set[str]:
    mapped_residues = set()
    for mapping in mapped_regions:
        #             if isinstance(mapping, MappedRegionWithNeighbourhood):
        if "MappedRegionWithNeighbourhood" in str(type(mapping)):
            mapped_residues = mapped_residues.union(
                set(mapping.mapped_region.residues_mapping.keys())
            ).union(set(mapping.close_residues))
        else:
            mapped_residues = mapped_residues.union(
                set(mapping.residues_mapping.keys())
            )
    return all_residues.difference({str(val) for val in mapped_residues})


def get_remaining_residues(file_2_mapped_regions, file_2_previously_remaining_residues):
    file_2_remaining_residues = dict()
    for filename, all_residues in file_2_previously_remaining_residues.items():
        mapped_regions = file_2_mapped_regions.get(filename, [])
        file_2_remaining_residues[filename] = get_remaining_residues_per_file(
            all_residues, mapped_regions
        )
    return file_2_remaining_residues


def plot_aligned_domains(file_2_tmscore_residues):
    plt.figure(figsize=(17, 9))
    results_of_mapping = [
        (tmscore, len(mapping))
        for tmscore, mapping in sum(file_2_tmscore_residues.values(), [])
        if mapping is not None
    ]
    mapping_lenghts = list(map(lambda x: x[1], results_of_mapping))
    plt.scatter(list(map(lambda x: x[0], results_of_mapping)), mapping_lenghts)
    plt.xticks(np.arange(0, 1, 0.05), rotation=90)
    plt.yticks(np.arange(min(mapping_lenghts) - 10, max(mapping_lenghts) + 10, 5))
    plt.show()


def detect_second_encounter_of_domain(
    domain_name,
    domain_2_threshold,
    domain_2_file_2_tmscore_residues,
    pdb_files,
    file_2_remaining_residues,
):
    tmscore_threshold, mapping_size_threshold = domain_2_threshold[domain_name]
    file_2_tmscore_residues = get_alignments(
        pdb_files,
        domain_name,
        file_2_remaining_residues,
        tmscore_threshold,
        mapping_size_threshold,
    )
    print(
        f"For {domain_name} # of 2nd encounters in the same protein is: {len(file_2_tmscore_residues)}"
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
    file_2_remaining_residues, file_2_all_residues
):
    file_2_longest_unmapped_region = dict()
    for filename, unmapped_residues in file_2_remaining_residues.items():
        file_2_longest_unmapped_region[filename] = find_longest_continuous_segment(
            unmapped_residues, file_2_all_residues[filename]
        )
    return file_2_longest_unmapped_region


def return_short_enough_segments(segment, max_allowed_length):
    if max(segment) - min(segment) <= max_allowed_length:
        return [segment]
    mid_index = (max(segment) + min(segment)) / 2
    return return_short_enough_segments(
        [res for res in segment if res <= mid_index], max_allowed_length
    ) + return_short_enough_segments(
        [res for res in segment if res > mid_index], max_allowed_length
    )


def find_continuous_segments_longer_than(
    residues_subset, min_secondary_struct_len=40, max_allowed_gap=5
):
    res_continuous_segments = [[]]
    prev_res = None
    for res in sorted(map(int, residues_subset)):
        if prev_res is not None:
            allowed_prev_residues = {prev_res + 1}.union(
                {prev_res + 1 + i for i in range(max_allowed_gap)}
            )
        if prev_res is not None and res not in allowed_prev_residues:
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
    file_2_all_residues: dict,
    filename_2_known_regions: dict,
    helix_sheet_dist_threshold: float = 17,
    max_allowed_segment_len=7,
):
    already_mapped_residues = set()
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
            set(mapped_region.residues_mapping.keys()),
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
                    f"{filename} & resi {compress_selection_list(set(region_segment))} & chain A",
                )
                segment_i += 1
                region_i_2_segments[mapped_region_i].append(segment_name)
    mapped_region_2_added_residues = defaultdict(list)

    # detect secondary structure residue segments in the unmapped parts
    remaining_residues_segments = find_continuous_segments_longer_than(
        remaining_residues, min_secondary_struct_len=0, max_allowed_gap=1
    )
    for residue_segment_remaining_master in remaining_residues_segments:
        for residue_segment_remaining in return_short_enough_segments(
            residue_segment_remaining_master, max_allowed_length=max_allowed_segment_len
        ):
            cmd.select(
                "small_selection",
                f"{filename} & resi {compress_selection_list(set(residue_segment_remaining))} & chain A",
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
    pdb_filepaths,
    file_2_all_residues: dict,
    filename_2_known_regions: dict,
    single_thread=False,
    n_jobs=8,
):
    get_mapped_regions_with_surroundings_partial = partial(
        get_mapped_regions_with_surroundings,
        file_2_all_residues=file_2_all_residues,
        filename_2_known_regions=filename_2_known_regions,
    )
    pdb_filenames = [
        filepath.stem if isinstance(filepath, Path) else filepath.replace(".pdb", "")
        for filepath in pdb_filepaths
    ]
    if single_thread:
        list_of_new_mapped_regions = []
        for filename in tqdm(pdb_filenames):
            list_of_new_mapped_regions.extend(
                get_mapped_regions_with_surroundings_partial(filename)
            )
    else:
        with Pool(n_jobs) as p:
            list_of_new_mapped_regions = p.map(
                get_mapped_regions_with_surroundings_partial, pdb_filenames
            )
    filename_2_known_regions_completed = dict(
        zip(pdb_filenames, list_of_new_mapped_regions)
    )
    return filename_2_known_regions_completed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-start-i", type=int, default=0)
    parser.add_argument("--domain-end-i", type=int, default=2500)
    parser.add_argument("--pdb-start-i", type=int, default=0)
    parser.add_argument("--pdb-end-i", type=int, default=2500)
    args = parser.parse_args()

    domain_start_i, domain_end_boarder_i = args.domain_start_i, args.domain_end_i
    pdb_start_i, pdb_end_boarder_i = args.pdb_start_i, args.pdb_end_i

    blacklist_files = {"1ps1.pdb", "5eat.pdb", "3p5r.pdb", "1sqc.pdb"}
    pdb_files = [
        filepath
        for filepath in Path(".").glob("*.pdb")
        if str(filepath) not in blacklist_files
    ]
    # file_2_all_residues = get_all_residues_per_file(pdb_files, cmd)
    # with open('file_2_all_residues.pkl', 'wb') as f:
    #     pickle.dump(file_2_all_residues, f)
    #
    # sys.exit(0)
    with open("file_2_all_residues.pkl", "rb") as f:
        file_2_all_residues = pickle.load(f)

    terpene_synthases_df = pd.read_csv("../missing_jung_ids.csv")
    required_ids = set(terpene_synthases_df["Uniprot ID"].values)

    # pdb_files = [
    #     filepath
    #     for filepath in pdb_files
    #     if filepath.stem in file_2_all_residues and filepath.stem in required_ids
    # ]

    domains_sorted = sorted(SUPPORTED_DOMAINS)
    domains_to_process = domains_sorted[domain_start_i:domain_end_boarder_i]

    print(f"Domains to process: {len(domains_to_process)}")
    with open("../ids_to_score.pkl", "rb") as file:
        ids_to_score = pickle.load(file)
    pdb_files = sorted(
        [filepath for filepath in pdb_files if filepath.stem in ids_to_score]
    )

    import time

    start_time = time.time()

    type_2_file_2_tmscore_residues = {
        domain: get_alignments(
            pdb_files[pdb_start_i:pdb_end_boarder_i], domain, file_2_all_residues
        )
        for domain in domains_to_process
    }
    with open(
        f"type_2_file_2_tmscore_residues_doms_{domain_start_i}_{domain_end_boarder_i}_additional_alpha_pdbs_{pdb_start_i}_{pdb_end_boarder_i}.pkl",
        "wb",
    ) as f:
        pickle.dump(type_2_file_2_tmscore_residues, f)

    print(
        f"doms_{domain_start_i}_{domain_end_boarder_i}_pdbs_{pdb_start_i}_{pdb_end_boarder_i} structures took {time.time() - start_time} s"
    )
    sys.exit(0)

    # # picking the best-aligning alpha standard
    # file_2_rms_residues_alpha = dict()
    # file_2_best_alpha_type = dict()
    # for file in file_2_all_residues.keys():
    #     best_alpha_type = None
    #     best_tmscore = float("inf")
    #     best_mapping = None
    #     for (
    #         alpha_type,
    #         file_2_tmscore_residues_type,
    #     ) in alpha_type_2_file_2_tmscore_residues.items():
    #         if file in file_2_tmscore_residues_type:
    #             tmscore, residues_mapping = file_2_tmscore_residues_type[file][0]
    #             if tmscore > best_tmscore:
    #                 best_tmscore = tmscore
    #                 best_mapping = residues_mapping
    #                 best_alpha_type = alpha_type
    #     if best_mapping is not None:
    #         file_2_rms_residues_alpha[file] = [(best_tmscore, best_mapping)]
    #         file_2_best_alpha_type[file] = best_alpha_type
    #
    # domain_2_threshold = {f"alpha_{i}": (0.5, 170) for i in range(21)}
    # domain_2_file_2_rms_residues = {"alpha": file_2_rms_residues_alpha}
    # print("alpha done")
    #
    # file_2_mapped_regions = get_mapped_regions_per_file(
    #     domain_2_file_2_rms_residues, domain_2_threshold
    # )
    # file_2_remaining_residues = get_remaining_residues(
    #     file_2_mapped_regions, file_2_all_residues
    # )
    # second_alpha_type_2_file_2_tmscore_residues = {
    #     f"alpha_{alpha_i}": detect_second_encounter_of_domain(
    #         f"alpha_{alpha_i}",
    #         domain_2_threshold,
    #         domain_2_file_2_rms_residues,
    #         pdb_files,
    #         file_2_remaining_residues,
    #     )
    #     for alpha_i in range(21)
    # }
    # file_2_best_second_alpha_type = dict()
    # for file in file_2_all_residues.keys():
    #     best_alpha_type = None
    #     best_tmscore = -1
    #     best_mapping = None
    #     for (
    #         alpha_type,
    #         file_2_tmscore_residues_type,
    #     ) in second_alpha_type_2_file_2_tmscore_residues.items():
    #         if file in file_2_tmscore_residues_type:
    #             tmscore, residues_mapping = file_2_tmscore_residues_type[file][0]
    #             if tmscore > best_tmscore:
    #                 best_tmscore = tmscore
    #                 best_mapping = residues_mapping
    #                 best_alpha_type = alpha_type
    #     if best_mapping is not None:
    #         file_2_rms_residues_alpha[file].append((best_tmscore, best_mapping))
    #         file_2_best_second_alpha_type[file] = best_alpha_type

    # # Beta

    # domain_2_threshold.update({"beta": (0.7, 115)})
    # domain_2_file_2_rms_residues.update({"beta": file_2_rms_residues_beta})
    # domain_2_threshold = {"beta": (0.7, 115)}
    # domain_2_file_2_rms_residues = {"beta": file_2_rms_residues_beta}
    #
    # print("beta done")
    # file_2_mapped_regions = get_mapped_regions_per_file(
    #     domain_2_file_2_rms_residues, domain_2_threshold
    # )
    # file_2_remaining_residues = get_remaining_residues(
    #     file_2_mapped_regions, file_2_all_residues
    # )
    # file_2_remaining_residues = get_remaining_residues(
    #     file_2_mapped_regions, file_2_remaining_residues
    # )
    # # ### second
    # (
    #     file_2_mapped_regions,
    #     file_2_remaining_residues,
    # ) = detect_second_encounter_of_domain(
    #     "beta",
    #     domain_2_threshold,
    #     domain_2_file_2_rms_residues,
    #     pdb_files,
    #     file_2_remaining_residues,
    # )
    #
    # # # Gamma
    # file_2_rms_residues_gamma = get_alignments(pdb_files, "gamma", file_2_all_residues)
    # domain_2_threshold.update({"gamma": (0.64, 105)})
    # domain_2_file_2_rms_residues.update({"gamma": file_2_rms_residues_gamma})
    # file_2_mapped_regions = get_mapped_regions_per_file(
    #     domain_2_file_2_rms_residues, domain_2_threshold
    # )
    # file_2_remaining_residues = get_remaining_residues(
    #     file_2_mapped_regions, file_2_remaining_residues
    # )
    # (
    #     file_2_mapped_regions,
    #     file_2_remaining_residues,
    # ) = detect_second_encounter_of_domain(
    #     "gamma",
    #     domain_2_threshold,
    #     domain_2_file_2_rms_residues,
    #     pdb_files,
    #     file_2_remaining_residues,
    # )
    #
    # print("gamma done")
    # # # Delta
    # # file_2_rms_residues_delta = get_alignments(pdb_files, "delta", file_2_all_residues)
    # # domain_2_threshold.update({"delta": (0.64, 140)})
    # # domain_2_file_2_rms_residues.update({"delta": file_2_rms_residues_delta})
    # #
    # # # file_2_mapped_regions = get_mapped_regions_per_file(domain_2_file_2_rms_residues, domain_2_threshold)
    # # # file_2_remaining_residues = get_remaining_residues(file_2_mapped_regions, file_2_remaining_residues)
    # # # file_2_mapped_regions, file_2_remaining_residues = detect_second_encounter_of_domain('delta', domain_2_threshold, domain_2_file_2_rms_residues, pdb_files, file_2_remaining_residues)
    # #
    # # print("delta done")
    # # # # Epsilon
    # # file_2_rms_residues_epsilon = get_alignments(
    # #     pdb_files, "epsilon", file_2_all_residues
    # # )
    # # domain_2_threshold.update({"epsilon": (0.64, 150)})
    # # domain_2_file_2_rms_residues.update({"epsilon": file_2_rms_residues_epsilon})
    # #
    # # print("epsilon done")
    #
    # with open("file_2_mapped_regions_beta_gamma.pkl", "wb") as f:
    #     pickle.dump(file_2_mapped_regions, f)
    # sys.exit(0)
    #
    # # with open("domain_2_file_2_rms_residues_relaxed_beta_gamma.pkl", "wb") as f:
    # #     pickle.dump(domain_2_file_2_rms_residues, f)
    #
    # # with open("file_2_best_alpha_type.pkl", "wb") as f:
    # #     pickle.dump(file_2_best_alpha_type, f)
    # #
    # # with open("file_2_best_second_alpha_type.pkl", "wb") as f:
    # #     pickle.dump(file_2_best_second_alpha_type, f)
    #
    # file_2_mapped_regions = get_mapped_regions_per_file(
    #     domain_2_file_2_rms_residues, domain_2_threshold
    # )
    # # file_2_remaining_residues = get_remaining_residues(file_2_mapped_regions, file_2_remaining_residues)
    # # file_2_mapped_regions, file_2_remaining_residues = detect_second_encounter_of_domain('epsilon', domain_2_threshold, domain_2_file_2_rms_residues, pdb_files, file_2_remaining_residues)
    #
    # with open("file_2_mapped_regions_part_2.pkl", "wb") as f:
    #     pickle.dump(file_2_mapped_regions, f)
    #
    # file_2_mapped_regions_with_neighbourhood = defaultdict(list)
    # for file, mapped_modules in file_2_mapped_regions.items():
    #     for mapped_region in mapped_modules:
    #         mapped_region_with_surrounding = get_mapped_region_with_domain_surroundings(
    #             cmd, file, file_2_all_residues[file], mapped_region
    #         )
    #         file_2_mapped_regions_with_neighbourhood[file].append(
    #             mapped_region_with_surrounding
    #         )
    #
    # with open("file_2_mapped_regions_with_neighbourhood_part2.pkl", "wb") as f:
    #     pickle.dump(file_2_mapped_regions_with_neighbourhood, f)
    # # # # Unmapped regions
    # #
    # # for filename, unmapped_residues in file_2_remaining_residues.items():
    # #     for residues in find_continuous_segments_longer_than(unmapped_residues, min_len=50, max_allowed_gap=3):
    # #         file_2_mapped_regions[filename].append(MappedRegion(module_id=f'{filename}_unmapped_9999',
    # #                                                                             domain='unmapped',
    # #                                                                                 rmsd=99,
    # #                                                                                 residues_mapping={res: res for res in residues}))
    #
    # sys.exit(0)
    # # # All module counts
    #
    # regions_all = []
    # regions_sample = []
    #
    # for domain in [
    #     "alpha",
    #     "alpha_prime",
    #     "beta",
    #     "gamma",
    #     "delta",
    #     "epsilon",
    #     "unmapped",
    # ]:
    #     _regions = get_regions_of_interest({domain}, file_2_mapped_regions)
    #     print(domain, len(_regions))
    #     regions_sample.extend(sample(_regions, int(len(_regions) / 3)))
    #     regions_all.extend(_regions)
    #
    # with open("regions_sampled.pkl", "wb") as f:
    #     pickle.dump(regions_sample, f)
    # with open("regions_all.pkl", "wb") as f:
    #     pickle.dump(regions_all, f)
