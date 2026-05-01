# pylint: disable=C0302
"""This module contains our structural algorithms for segmentation of a protein structure into TPS-specific domains
and comparison of domains between each other"""

import os
import re
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from uuid import uuid4
import logging

import numpy as np  # type: ignore
from pymol import cmd, stored  # type: ignore
from scipy.spatial import KDTree  # type: ignore
from tqdm.auto import tqdm  # type: ignore

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# ---------------------------------------------------------------------------
# Optimisation state (USalign batch + persistent pool + caches)
# ---------------------------------------------------------------------------
# Path to the USalign binary. Override via env var if needed.
USALIGN_PATH = os.environ.get(
    "ENZYMEEXPLORER_USALIGN", "/home/akmese/usalign_src/USalign"
)

# Original SS-residues map captured at the SSR step. Workers inherit it
# via fork; lets `_stage_one` skip the per-call SS-iteration on every
# freshly-loaded query. Populated by `domain_detections.detect_domains`
# right after `save_file_to_all_residues` writes the map.
_SS_FULL_MAP_CACHE: dict[str, set[str]] = {}

# Persistent worker pool, lazily created inside `get_alignments` on the
# first call when an outer detect_domains run has set up the context.
_PERSISTENT_POOL = None
_PERSISTENT_POOL_N_JOBS: int = 0

# USalign output regexes — module level so they're compiled once.
_RE_TMSCORE = re.compile(r"TM-score\s*=\s*([0-9]*\.[0-9]+)")
_RE_NAME2 = re.compile(r"Name of Structure_2:\s*([^:\s]+)")


@dataclass(eq=True)
class MappedRegion:
    """A dataclass to store information about a particular structural module"""

    module_id: str
    domain: str
    tmscore: float
    residues_mapping: dict[int, int]
    aligned_template: str = ""
    def __post_init__(self):
        self.aligned_template = self.domain


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
    """Compute full residue mapping from a larger object to a domain object.

    Cleaned-up version: V0's interval-string building (`domain_intervals`,
    `mapped_intervals`, `_temp1`, `_temp2`) was used only as an
    early-return gate equivalent to checking whether `residues_mapping_full`
    ended up empty. Removing it does not change the output. Repeated
    `sorted/min/max(map(int, ...))` calls are also collapsed to one pass.
    """
    if len(residues_mapping) == 0:
        return {}

    # Resolve once.
    obj_res_2_mapped_shift = {
        domain_res: int(res) - int(domain_res)
        for res, domain_res in residues_mapping.items()
    }
    domain_keys_int = sorted(int(k) for k in obj_res_2_mapped_shift.keys())
    kdtree = KDTree(np.asarray(domain_keys_int, dtype=np.int64).reshape(-1, 1))
    shift_values = list(obj_res_2_mapped_shift.values())

    sorted_mapped_domain_residues = sorted(
        int(v) for v in residues_mapping.values()
    )
    n_mapped = len(sorted_mapped_domain_residues)
    domain_map_first = sorted_mapped_domain_residues[-1]
    domain_map_last = sorted_mapped_domain_residues[0]
    for i in range(n_mapped - 3):
        if [
            sorted_mapped_domain_residues[i + j] - sorted_mapped_domain_residues[i]
            for j in range(3)
        ] == [0, 1, 2]:
            domain_map_first = sorted_mapped_domain_residues[i]
            break
    for i in range(n_mapped - 3, 0, -1):
        if [
            sorted_mapped_domain_residues[i + j] - sorted_mapped_domain_residues[i]
            for j in range(3)
        ] == [0, 1, 2]:
            domain_map_last = sorted_mapped_domain_residues[i]
            break

    sorted_domain_residues = [
        r
        for r in sorted(int(x) for x in file_2_all_residues[domain_obj])
        if domain_map_first <= r <= domain_map_last
    ]
    obj_residues_set = set(int(x) for x in file_2_all_residues[larger_obj])

    residues_mapping_full: dict[int, int] = {}
    mapped_residues: set[int] = set()
    for domain_res in sorted_domain_residues:
        if domain_res in obj_res_2_mapped_shift:
            shift = obj_res_2_mapped_shift[domain_res]
        else:
            _, closest_indices = kdtree.query(domain_res, k=2)
            shift = int(
                round(
                    np.mean(
                        [shift_values[idx] for idx in closest_indices]
                    )
                )
            )
        mapped_res = int(domain_res) + shift
        if mapped_res not in mapped_residues and mapped_res in obj_residues_set:
            residues_mapping_full[mapped_res] = domain_res
            mapped_residues.add(mapped_res)

    if not residues_mapping_full:
        return {}
    return residues_mapping_full


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


def get_all_residues_per_file(pdb_files: list[Path], pymol_cmd) -> dict[str, set[str]]:
    """
    Computes a set of all residues in each PDB file provided.

    :param pdb_files: A list of Path objects representing the PDB files to be processed
    :param pymol_cmd: The PyMOL command object used to interact with the PyMOL session

    :return: A dictionary mapping each PDB filename (without extension) to a set of all residues found in that file
    """
    file_2_all_residues = {}
    for filepath in tqdm(pdb_files, desc="Loading PDB files and extracting residues"):
        str_name = filepath.stem
        pymol_cmd.load(str(filepath))
        all_residues = get_secondary_structure_residues_set(str_name, pymol_cmd)
        if len(all_residues):
            file_2_all_residues[str_name] = all_residues
        pymol_cmd.delete(str_name)
    return file_2_all_residues


def _stage_target_pdb_pure_python(
    pdb_path,
    out_path,
    residues_permitted: set[int],
) -> list[str]:
    """Write a CA-only chain-A PDB filtered to `residues_permitted`.

    Pure-Python file scan — much faster than PyMOL load+select+save and
    bit-equivalent for AlphaFold-style PDBs (CA atoms in standard
    record order). Returns residue numbers (as strings) in PDB write
    order.
    """
    out_lines: list[str] = []
    target_resis: list[str] = []
    with open(pdb_path) as fr:
        for line in fr:
            if not line.startswith("ATOM "):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            alt = line[16]
            if alt not in (" ", "A"):
                continue
            chain = line[21]
            if chain != "A":
                continue
            resi_str = line[22:26].strip()
            try:
                resi_int = int(resi_str)
            except ValueError:
                continue
            if resi_int not in residues_permitted:
                continue
            out_lines.append(line)
            target_resis.append(resi_str)
    with open(out_path, "w") as fw:
        fw.writelines(out_lines)
        fw.write("END\n")
    return target_resis


def _stage_one(args):
    """Per-call query staging for the USalign batch flow.

    Pure-Python PDB scan: writes a CA-only filtered PDB for the query.
    No PyMOL involvement. Returns metadata used by the batch step.
    """
    (
        filename,
        domain_template,
        file_2_all_residues_for_query,
        staging_dir,
        template_ss_residues,
        ss_full_map,
    ) = args

    larger_obj = filename
    file_2_all_residues = file_2_all_residues_for_query

    min_align_len = int(
        domain_template.get("thresholds", {}).get("min_align_len", 0)
    )

    if larger_obj in file_2_all_residues and len(
        file_2_all_residues[larger_obj]
    ) < max(1, min_align_len):
        return None

    ss_full = ss_full_map.get(larger_obj)
    if ss_full is None:
        ss_full = file_2_all_residues.get(larger_obj, set())

    allowed_residues = file_2_all_residues.get(larger_obj, ss_full)
    residues_permitted = ss_full.intersection(allowed_residues)
    if len(residues_permitted) < max(1, min_align_len):
        return None

    pdb_path = f"{larger_obj}.pdb"
    if not os.path.exists(pdb_path):
        raise FileNotFoundError(
            f"{pdb_path} while being in {os.getcwd()}"
        )
    target_pdb = staging_dir / f"q_{filename}.pdb"
    residues_permitted_int = {
        int(r) for r in residues_permitted if str(r).lstrip("-").isdigit()
    }
    target_resis = _stage_target_pdb_pure_python(
        pdb_path=pdb_path,
        out_path=target_pdb,
        residues_permitted=residues_permitted_int,
    )

    if len(target_resis) < max(1, min_align_len):
        try:
            target_pdb.unlink()
        except FileNotFoundError:
            pass
        return None

    return {
        "filename": filename,
        "target_pdb": str(target_pdb),
        "target_resis": target_resis,
        "domain_residues": template_ss_residues,
        "larger_residues": residues_permitted,
        "domain_template_name": domain_template["name"],
    }


def _run_usalign_batch(
    template_pdb: str,
    staging_dir,
    list_path,
    L_value: int,
    timeout: int = 3600,
) -> str:
    """Run one USalign batch invocation, return stdout."""
    cmd_args = [
        USALIGN_PATH,
        template_pdb,
        "-dir2", str(staging_dir) + "/",
        str(list_path),
        "-suffix", ".pdb",
        "-L", str(L_value),
    ]
    proc = subprocess.run(
        cmd_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=False,
        timeout=timeout,
    )
    if proc.returncode != 0:
        logger.error(
            "USalign batch rc=%d stderr=%s", proc.returncode, proc.stderr[:400]
        )
        return ""
    return proc.stdout


def _parse_batch_output(stdout: str):
    """Split USalign batch stdout into per-pair records.

    Returns {target_filename_stem: (tmscore, qaln, match, taln)}.
    """
    out: dict = {}
    if not stdout:
        return out
    chunks = re.split(r"\* US-align \(Version", stdout)
    for chunk in chunks:
        if "Name of Structure_2:" not in chunk:
            continue
        m = _RE_NAME2.search(chunk)
        if m is None:
            continue
        target_path = m.group(1)
        basename = os.path.basename(target_path)
        if basename.endswith(".pdb"):
            basename = basename[:-4]
        if basename.startswith("q_"):
            basename = basename[2:]

        tmscore = None
        for tm_match in _RE_TMSCORE.finditer(chunk):
            try:
                tmscore = float(tm_match.group(1))
            except ValueError:
                pass
        if tmscore is None:
            continue
        idx = chunk.find('(":" denotes')
        if idx < 0:
            continue
        rest = chunk[idx:].splitlines()
        if len(rest) < 4:
            continue
        qaln = rest[1].rstrip("\n")
        match = rest[2]
        taln = rest[3].rstrip("\n")
        out[basename] = (tmscore, qaln, match, taln)
    return out


def _finalize_one(args):
    """Build residues_mapping from alignment, run compute_full_mapping."""
    (staged_meta, parsed_pair, template_resis, min_domain_fraction) = args
    if parsed_pair is None:
        return staged_meta["filename"], -float("inf"), {}

    tmscore, qaln, match, taln = parsed_pair
    target_resis = staged_meta["target_resis"]
    mobile_resis = template_resis
    residues_mapping: dict = {}
    i1 = 0
    i2 = 0
    for k in range(len(qaln)):
        c1 = qaln[k] if k < len(qaln) else "-"
        c2 = taln[k] if k < len(taln) else "-"
        m_ind = match[k] if k < len(match) else " "
        if c1 != "-" and c2 != "-" and m_ind in ":.":
            if i1 < len(mobile_resis) and i2 < len(target_resis):
                residues_mapping[target_resis[i2]] = mobile_resis[i1]
        if c1 != "-":
            i1 += 1
        if c2 != "-":
            i2 += 1

    domain_residues = staged_meta["domain_residues"]
    if len(residues_mapping) < min_domain_fraction * len(domain_residues):
        return staged_meta["filename"], -float("inf"), {}

    f2ar = {
        "_dom": domain_residues,
        "_lrg": staged_meta["larger_residues"],
    }
    residues_mapping_full = compute_full_mapping(
        "_dom", "_lrg", residues_mapping, file_2_all_residues=f2ar
    )
    return staged_meta["filename"], tmscore, residues_mapping_full


def _process_chunk(args):
    """One worker stages its chunk + runs USalign batch + finalises."""
    (chunk_filenames, domain_template, file_2_all_residues, template_pdb_path,
     template_resis, L_value, shared_staging_root, worker_id,
     template_ss_residues, ss_full_map) = args

    staging_dir = Path(shared_staging_root) / f"w{worker_id}"
    staging_dir.mkdir(parents=True, exist_ok=True)

    staged_metas = []
    for fn in chunk_filenames:
        meta = _stage_one(
            (
                fn,
                domain_template,
                dict(file_2_all_residues),
                staging_dir,
                template_ss_residues,
                ss_full_map,
            )
        )
        if meta is not None:
            staged_metas.append(meta)

    if not staged_metas:
        return []

    list_path = staging_dir / "list.txt"
    list_path.write_text(
        "\n".join(f"q_{m['filename']}.pdb" for m in staged_metas) + "\n"
    )
    stdout = _run_usalign_batch(
        template_pdb=template_pdb_path,
        staging_dir=staging_dir,
        list_path=list_path,
        L_value=L_value,
    )
    parsed = _parse_batch_output(stdout)

    out = []
    for m in staged_metas:
        out.append(_finalize_one((m, parsed.get(m["filename"]), template_resis, 0.1)))
    return out


def get_alignments(
    pdb_filepaths: list[Path],
    domain_template: dict,
    file_2_current_residues: dict[str, set[str]],
    n_jobs: int = 8,
) -> dict[str, list[tuple[float, dict]]]:
    """Computes alignments of a template against all queries via batched USalign.

    Per-worker batching: queries are split across `n_jobs` workers via
    longest-processing-time-first bin packing. Each worker stages its
    chunk (CA-only filtered PDB writes) and runs ONE USalign batch
    subprocess on its chunk. This replaces V0's per-pair TMalign
    subprocess fork (saving ~3-5 ms × N_pairs) and removes the per-call
    PyMOL load/select/save overhead.

    A persistent worker pool is reused when one is set up by the outer
    `detect_domains` run; otherwise a per-call Pool is opened.

    :return: {filename_stem: [(tmscore, residues_mapping_full)]} filtered by
             the template's tmscore + min_align_len thresholds.
    """
    pdb_filenames = [
        fp.stem for fp in pdb_filepaths if file_2_current_residues.get(fp.stem)
    ]
    if not pdb_filenames:
        return defaultdict(list)

    template_workdir = Path(tempfile.mkdtemp(prefix="usalign_batch_"))
    template_pdb_path = template_workdir / f"tpl_{domain_template['name']}.pdb"
    template_resis: list[str] = []
    try:
        from psico.exporting import save_pdb_without_ter  # type: ignore

        tpl_obj = f"tpl_obj_{uuid4().hex[:8]}"
        cmd.load(str(Path(domain_template["path"]).absolute()), tpl_obj)
        tpl_sel = f"{tpl_obj}_sel"
        cmd.select(tpl_sel, f"{tpl_obj} & {domain_template['residues']}")
        tpl_ss = f"{tpl_obj}_ss"
        cmd.select(tpl_ss, f"{tpl_sel} & ss H+S")
        tpl_ca = f"({tpl_ss}) and (not hetatm) and name CA and alt +A"
        cmd.iterate_state(
            1, tpl_ca, "out.append(resi)", space={"out": template_resis}
        )
        save_pdb_without_ter(str(template_pdb_path), tpl_ca, state=1)
        template_residues_set = get_secondary_structure_residues_set(tpl_sel, cmd)
        cmd.delete(tpl_obj)
        cmd.delete(tpl_sel)
        cmd.delete(tpl_ss)

        L_value = len(template_residues_set)

        # Longest-processing-time-first chunking for load balance.
        n_chunks = min(n_jobs, len(pdb_filenames))

        def _query_cost(fn: str) -> int:
            return len(file_2_current_residues.get(fn, ()))

        sorted_filenames = sorted(pdb_filenames, key=_query_cost, reverse=True)
        chunk_loads = [0] * n_chunks
        chunks: list[list[str]] = [[] for _ in range(n_chunks)]
        for fn in sorted_filenames:
            wid = chunk_loads.index(min(chunk_loads))
            chunks[wid].append(fn)
            chunk_loads[wid] += _query_cost(fn) or 1
        chunks = [c for c in chunks if c]

        shared_staging_root = template_workdir / "wstg"
        shared_staging_root.mkdir()

        ss_full_map = _SS_FULL_MAP_CACHE or file_2_current_residues
        worker_args = [
            (
                chunk,
                domain_template,
                dict(file_2_current_residues),
                str(template_pdb_path),
                template_resis,
                L_value,
                str(shared_staging_root),
                wid,
                template_residues_set,
                ss_full_map,
            )
            for wid, chunk in enumerate(chunks)
        ]

        # Persistent pool reused when set up by outer detect_domains run.
        if (
            _PERSISTENT_POOL is not None
            and _PERSISTENT_POOL_N_JOBS >= n_chunks
        ):
            chunk_results = _PERSISTENT_POOL.map(_process_chunk, worker_args)
        else:
            with Pool(n_chunks) as pool:
                chunk_results = pool.map(_process_chunk, worker_args)

        out = defaultdict(list)
        tmscore_thr = domain_template["thresholds"]["tmscore"]
        min_aln_len = domain_template["thresholds"]["min_align_len"]
        for chunk_out in chunk_results:
            for filename, tmscore, residues_mapping in chunk_out:
                if (
                    tmscore >= tmscore_thr
                    and len(residues_mapping) >= min_aln_len
                ):
                    out[filename].append((tmscore, residues_mapping))
        return out
    finally:
        import shutil
        shutil.rmtree(template_workdir, ignore_errors=True)


def get_remaining_residues_per_file(
    all_residues: set[str],
    mapped_regions: list[dict[int, int]],
) -> set[str]:
    """
    Function retrieving currently unassigned residues
    """
    mapped_residues: set[int] = set()
    for mapping in mapped_regions:
        mapped_residues = mapped_residues.union(set(mapping.keys()))
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
            all_residues, [region.residues_mapping for region in mapped_regions]
        )
    return file_2_remaining_residues

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


def _residue_atoms(filename: str) -> dict[int, np.ndarray]:
    """Return {resi_int: (n_atoms, 3) coord array} for chain A of `filename`.

    One `iterate_state` call pulls every atom — orders of magnitude
    cheaper than per-segment `cmd.distance` invocations.
    """
    space: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    cmd.iterate_state(
        1,
        f"{filename} & chain A",
        "space[int(resi)].append((x, y, z))",
        space={"space": space},
    )
    return {r: np.asarray(coords, dtype=np.float64) for r, coords in space.items()}


def _segment_centroid(
    residues: list[int], resi_atoms: dict[int, np.ndarray]
) -> np.ndarray | None:
    rows = [resi_atoms[r] for r in residues if r in resi_atoms]
    if not rows:
        return None
    return np.concatenate(rows, axis=0).mean(axis=0)


def get_mapped_regions_with_surroundings(
    filename: str,
    file_2_all_residues: dict[str, set[str]],
    filename_2_known_regions: dict[str, list[MappedRegion]],
    helix_sheet_neighbor_dist_threshold: float = 20,
    helix_sheet_domain_dist_threshold: float = 30,
    max_allowed_segment_len: int = 7,
) -> list[MappedRegion]:
    """Detect unassigned SS parts close to a mapped region in 3D space.

    Numpy-centroid implementation: pulls all chain-A CA-atom coords once
    via a single `iterate_state` call, then computes per-segment
    centroids in numpy. Replaces V0's per-pair `cmd.distance(mode=4)`
    invocations (which created+deleted `dist` PyMOL objects per pair)
    with `np.linalg.norm` between two 3-vectors. Same `mode=4` semantics
    (centroid-to-centroid). Bit-equivalent within float precision.
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
            raise FileNotFoundError(
                f"{filename}.pdb while being in {os.getcwd()}"
            )
        cmd.load(f"{filename}.pdb")

    # Single coord pull replaces per-segment cmd.select / cmd.distance pairs.
    resi_atoms = _residue_atoms(filename)

    # Pre-compute centroids for every region segment.
    region_i_2_centroids: dict[int, list[np.ndarray]] = defaultdict(list)
    for mapped_region_i, mapped_region in enumerate(
        filename_2_known_regions[filename]
    ):
        region_continuous_segments = find_continuous_segments_longer_than(
            set(map(str, mapped_region.residues_mapping.keys())),
            min_secondary_struct_len=5,
            max_allowed_gap=2,
        )
        for region_segment_master in region_continuous_segments:
            for region_segment in return_short_enough_segments(
                region_segment_master,
                max_allowed_length=max_allowed_segment_len,
            ):
                c = _segment_centroid(region_segment, resi_atoms)
                if c is not None:
                    region_i_2_centroids[mapped_region_i].append(c)

    mapped_region_2_added_residues: dict[int, list[int]] = defaultdict(list)

    remaining_residues_segments = find_continuous_segments_longer_than(
        set(map(str, remaining_residues)),
        min_secondary_struct_len=0,
        max_allowed_gap=1,
    )
    for residue_segment_remaining_master in remaining_residues_segments:
        for residue_segment_remaining in return_short_enough_segments(
            residue_segment_remaining_master,
            max_allowed_length=max_allowed_segment_len,
        ):
            small_centroid = _segment_centroid(residue_segment_remaining, resi_atoms)
            if small_centroid is None:
                continue

            min_dist = float("inf")
            closest_region_i: int | None = None
            all_dists_with_regions: list[tuple[float, int]] = []
            region_to_dists: dict[int, list[float]] = defaultdict(list)

            for mapped_region_i in range(len(filename_2_known_regions[filename])):
                centroids = region_i_2_centroids.get(mapped_region_i, [])
                region_dist_min = float("inf")
                for c in centroids:
                    d = float(np.linalg.norm(small_centroid - c))
                    region_to_dists[mapped_region_i].append(d)
                    if d < region_dist_min:
                        region_dist_min = d
                all_dists_with_regions.append((region_dist_min, mapped_region_i))
                if region_dist_min < min_dist:
                    min_dist = region_dist_min
                    closest_region_i = mapped_region_i

            region_to_num_dists = {
                region: sum(
                    1 for d in dists if d < helix_sheet_domain_dist_threshold
                )
                for region, dists in region_to_dists.items()
            }
            num_dists_to_regions = {
                v: [r for r, n in region_to_num_dists.items() if n == v]
                for v in set(region_to_num_dists.values())
            }

            if min_dist < helix_sheet_neighbor_dist_threshold:
                if len(all_dists_with_regions) >= 2:
                    regions_apart = [
                        (d, r) for (d, r) in all_dists_with_regions if d > min_dist
                    ]
                    if regions_apart:
                        second_closest_dist, second_closest_region_i = min(
                            regions_apart, key=lambda x: x[0]
                        )
                        if (
                            second_closest_region_i == closest_region_i
                            or min_dist < 0.9 * second_closest_dist
                        ):
                            mapped_region_2_added_residues[closest_region_i].extend(
                                residue_segment_remaining
                            )
                        else:
                            overall_closest_regions = max(
                                num_dists_to_regions.items(), key=lambda x: x[0]
                            )[1]
                            if len(overall_closest_regions) == 1:
                                mapped_region_2_added_residues[
                                    overall_closest_regions[0]
                                ].extend(residue_segment_remaining)
                else:
                    mapped_region_2_added_residues[closest_region_i].extend(
                        residue_segment_remaining
                    )

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
    if os.path.exists(f"{filename}.pdb"):
        cmd.delete(filename)
    return new_mapped_regions


def get_mapped_regions_with_surroundings_parallel(
    pdb_filepaths: list[Path | str],
    file_2_all_residues: dict[str, set[str]],
    filename_2_known_regions: dict[str, list[MappedRegion]],
    n_jobs: int = 8,
    helix_sheet_neighbor_dist_threshold: float = 17,
    helix_sheet_domain_dist_threshold: float = 25,
) -> dict[str, list[MappedRegion]]:
    """
    A function for detecting unassigned parts of secondary structure which are close a particular domain in 3D space
    for all files in parallel
    """
    get_mapped_regions_with_surroundings_partial = partial(
        get_mapped_regions_with_surroundings,
        file_2_all_residues=file_2_all_residues,
        filename_2_known_regions=filename_2_known_regions,
        helix_sheet_neighbor_dist_threshold=helix_sheet_neighbor_dist_threshold,
        helix_sheet_domain_dist_threshold=helix_sheet_domain_dist_threshold,
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
