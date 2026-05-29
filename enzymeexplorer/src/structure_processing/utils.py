import json
import logging
from enzymeexplorer.src.structure_processing.structural_algorithms import (
    MappedRegion,
    compress_selection_list,
    get_alignments,
    find_continuous_segments_longer_than,
)
from enzymeexplorer.src.structure_processing._pool_service import (
    require_active_service,
)
from pymol import cmd
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # type: ignore
import tempfile
from pathlib import Path
import os
import pickle
from collections import defaultdict
from enzymeexplorer.src.structure_processing.foldseek_wrapper import FoldseekWrapper
from tqdm.auto import tqdm
import re
import time
from datetime import datetime
import configargparse
from functools import partial



logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def __get_domain_names(
    seq_2_regions: dict[str, list[MappedRegion]],
) -> list[str]:
    return [
        region.module_id for regions in seq_2_regions.values() for region in regions
    ]


def __get_domain_2_seq_id_and_domain_type_maps(
    seq_2_regions: dict[str, list[MappedRegion]],
) -> tuple[dict[str, str], dict[str, str]]:
    domain_2_seq_id = {
        region.module_id: seq_id
        for seq_id, regions in seq_2_regions.items()
        for region in regions
    }
    domain_2_domain_type = {
        region.module_id: region.domain
        for regions in seq_2_regions.values()
        for region in regions
    }
    return domain_2_seq_id, domain_2_domain_type


_REF_HASH_SIDECAR_NAME = "_ref_set_hash.txt"


def _stable_ref_hash(
    reference_domains: list[str],
    reference_domains_dir: str,
    ref_preprocessed_name_map: dict[str, str] | None = None,
) -> str:
    """Content-based hash of the reference PDB set.

    Hashes ``(sorted_logical_name, on_disk_filename, pdb_bytes)`` for
    every domain so the cache key is portable across hosts: a prebuilt
    foldseek DB shipped in the production bundle keys identically on
    any machine that extracts the same PDBs.

    ``ref_preprocessed_name_map`` maps the *logical* domain name (the
    post-preprocessing id used inside the pipeline) to the *on-disk*
    PDB stem. If the caller doesn't pass it, the logical name is
    assumed to also be the on-disk name.

    The digest is memoized to a sidecar at
    ``<reference_domains_dir>/../<sidecar>`` so repeated calls only
    read the PDBs once. The sidecar can ALSO be precomputed and
    shipped inside the bundle; if present and well-formed it
    short-circuits the entire content hash — that's what makes the
    cache key match the prebuilt foldseek DB on the very first
    prediction.
    """
    import hashlib

    sidecar = Path(reference_domains_dir).parent / _REF_HASH_SIDECAR_NAME
    if sidecar.is_file():
        try:
            cached = sidecar.read_text().strip()
        except OSError:
            cached = ""
        # Accept any 32-hex-char digest; ignore stale/garbage files.
        if len(cached) == 32 and all(c in "0123456789abcdef" for c in cached):
            return cached

    name_map = ref_preprocessed_name_map or {}
    h = hashlib.blake2b(digest_size=16)
    for d in sorted(reference_domains):
        h.update(d.encode())
        h.update(b"\0")
        on_disk = name_map.get(d, d)
        h.update(on_disk.encode())
        h.update(b"\0")
        p = Path(reference_domains_dir) / f"{on_disk}.pdb"
        try:
            with p.open("rb") as fh:
                while True:
                    chunk = fh.read(1 << 20)  # 1 MiB
                    if not chunk:
                        break
                    h.update(chunk)
        except FileNotFoundError:
            h.update(b"<missing>")
        h.update(b"\n")
    digest = h.hexdigest()

    # Best-effort sidecar write so subsequent calls skip the rehash.
    try:
        sidecar.write_text(digest + "\n")
    except OSError:
        pass
    return digest


def _resolve_foldseek_db_root() -> Path:
    """Where to keep cached foldseek DBs. Configurable via env.

    Defaults to ``<repo>/data/foldseek_cache`` (the location the
    production bundle ships the prebuilt cache into) regardless of
    the caller's current working directory.
    """
    env = os.environ.get("ENZYMEEXPLORER_FOLDSEEK_REF_DB")
    if env:
        return Path(env)
    from enzymeexplorer.src.utils.project_info import get_data_root
    return get_data_root() / "foldseek_cache"


def _build_or_get_foldseek_ref_db(
    reference_domains: list[str],
    reference_domains_dir: str,
    ref_preprocessed_name_map: dict[str, str],
    threads: int = 8,
) -> Path:
    """Return cached foldseek reference-DB directory; build on first miss."""
    import json
    from datetime import datetime

    db_root = _resolve_foldseek_db_root()
    ref_hash = _stable_ref_hash(
        reference_domains, reference_domains_dir, ref_preprocessed_name_map,
    )
    db_dir = db_root / ref_hash
    db_path = db_dir / "db"
    marker = db_dir / "READY"

    if marker.exists():
        logger.info("foldseek-DB cache hit at %s", db_dir)
        return db_dir

    logger.info(
        "foldseek-DB cache miss; building DB at %s (%d entries)",
        db_dir,
        len(reference_domains),
    )
    db_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as staging:
        staging_p = Path(staging)
        for d in reference_domains:
            src = Path(reference_domains_dir) / (
                f"{ref_preprocessed_name_map.get(d, d)}.pdb"
            )
            dst = staging_p / f"{d}.pdb"
            try:
                os.symlink(src.absolute(), dst)
            except FileExistsError:
                pass

        FoldseekWrapper(threads=threads).createdb(str(staging_p), str(db_path))

    (db_dir / "manifest.json").write_text(
        json.dumps(
            {
                "ref_hash": ref_hash,
                "n_reference": len(reference_domains),
                "reference_domains_dir": str(reference_domains_dir),
                "created_at": datetime.now().isoformat(timespec="seconds"),
            },
            indent=2,
        )
    )
    marker.touch()
    return db_dir


def _foldseek_search_against_cached_db(
    query_dir: str,
    db_dir: Path,
    output_tsv: str,
    max_seqs: int,
    e_value: float = 100.0,
    threads: int = 8,
    coverage: float | None = None,
    cov_mode: int | None = None,
) -> pd.DataFrame:
    """Run foldseek `search` against a pre-built reference DB."""
    db_path = db_dir / "db"
    fs = FoldseekWrapper(threads=threads)

    with tempfile.TemporaryDirectory() as workdir:
        wp = Path(workdir)
        query_db = wp / "querydb"
        result_db = wp / "resultdb"
        tmp_dir = wp / "fs_tmp"
        tmp_dir.mkdir()

        fs.createdb(query_dir, str(query_db))
        fs.search(
            query_db=str(query_db),
            target_db=str(db_path),
            result_db=str(result_db),
            tmp_dir=str(tmp_dir),
            max_seqs=max_seqs,
            e_value=e_value,
            sensitivity=10,
            write_alignments=True,  # -a flag, needed for convertalis
            coverage=coverage,
            cov_mode=cov_mode,
        )
        return fs.convertalis(
            query_db=str(query_db),
            target_db=str(db_path),
            result_db=str(result_db),
            output_tsv=output_tsv,
        )


def get_foldseek_alignment_df(
    query_seq_2_regions: dict[str, list[MappedRegion]],
    query_domains_dir: str,
    query_preprocessed_name_map: dict[str, str],
    ref_seq_2_regions: dict[str, list[MappedRegion]],
    reference_domains_dir: str,
    ref_preprocessed_name_map: dict[str, str],
) -> pd.DataFrame:
    """Compute pairwise distance based features between domains.

    Uses a CACHED foldseek reference database keyed by content hash of
    the reference set: subsequent calls with the same reference reuse
    the on-disk DB (huge win for batch screening pipelines that rerun
    against martsDB). Output is bit-equivalent to V0's `easy_search`.
    """
    query_domains = __get_domain_names(query_seq_2_regions)
    reference_domains = __get_domain_names(ref_seq_2_regions)

    assert set(
        [file.stem for file in Path(reference_domains_dir).glob("*.pdb")]
    ) >= set(
        [ref_preprocessed_name_map.get(domain, domain) for domain in reference_domains]
    ), "Reference domains directory does not contain all required domain structures."

    assert set([file.stem for file in Path(query_domains_dir).glob("*.pdb")]) >= set(
        [query_preprocessed_name_map.get(domain, domain) for domain in query_domains]
    ), "Query domains directory does not contain all required domain structures."

    logger.info(
        f"Running Foldseek alignment for {len(query_domains)} query domains "
        f"against {len(reference_domains)} reference domains "
        f"(eg. {query_domains[0]} vs {reference_domains[0]})"
    )

    db_dir = _build_or_get_foldseek_ref_db(
        reference_domains, reference_domains_dir, ref_preprocessed_name_map
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        query_pdbs_dir = Path(tmpdir) / "query_pdbs"
        query_pdbs_dir.mkdir(parents=True, exist_ok=True)
        for domain in query_domains:
            src_path = Path(query_domains_dir) / (
                f"{query_preprocessed_name_map.get(domain, domain)}.pdb"
            )
            dst_path = query_pdbs_dir / f"{domain}.pdb"
            os.symlink(src_path.absolute(), dst_path)

        out_tsv = Path(tmpdir) / "foldseek_output.tsv"
        alignment_df = _foldseek_search_against_cached_db(
            query_dir=str(query_pdbs_dir),
            db_dir=db_dir,
            output_tsv=str(out_tsv),
            max_seqs=len(reference_domains) * 2,
            e_value=1000000,
            coverage=0.5,
            cov_mode=0
        )
        # Same query-name fixup easy_search did.
        query_set = set(query_domains)

        def _fix_query(x: str) -> str:
            if x in query_set:
                return x
            stripped = "_".join(x.split("_")[:-1])
            return stripped if stripped in query_set else x

        alignment_df["query"] = alignment_df["query"].map(_fix_query)

    query_domain_2_seq_id, query_domain_2_domain_type = (
        __get_domain_2_seq_id_and_domain_type_maps(query_seq_2_regions)
    )
    ref_domain_2_seq_id, ref_domain_2_domain_type = (
        __get_domain_2_seq_id_and_domain_type_maps(ref_seq_2_regions)
    )

    alignment_df = alignment_df.sort_values(by="alntmscore", ascending=False)
    alignment_df["query_domain_type"] = alignment_df["query"].map(query_domain_2_domain_type)
    alignment_df["query_seq_id"] = alignment_df["query"].map(query_domain_2_seq_id)
    alignment_df["target_domain_type"] = alignment_df["target"].map(ref_domain_2_domain_type)
    alignment_df["target_seq_id"] = alignment_df["target"].map(ref_domain_2_seq_id)

    return alignment_df


def get_reference_domain_type_2_module_ids(
    reference_seq_2_regions: dict[str, list[MappedRegion]],
) -> dict[str, list[str]]:
    """Get a mapping from reference domain types to module ids.

    Args:
        reference_seq_2_regions (dict[str, list[MappedRegion]]): A dictionary mapping reference sequence ids to lists of reference domains.
    Returns:
        defaultdict[str, list[str]]: A dictionary mapping reference domain types to sorted lists of module ids
    """
    ref_domain_type_2_module_id = defaultdict(list)
    for ref_seq in reference_seq_2_regions:
        for region in reference_seq_2_regions[ref_seq]:
            ref_domain_type_2_module_id[region.domain].append(region.module_id)
    ref_domain_type_2_module_id["alpha_1"] = ref_domain_type_2_module_id["alpha"]
    ref_domain_type_2_module_id["alpha_2"] = ref_domain_type_2_module_id["alpha"]
    del ref_domain_type_2_module_id["alpha"]

    for domain_type in ref_domain_type_2_module_id:
        ref_domain_type_2_module_id[domain_type] = sorted(
            ref_domain_type_2_module_id[domain_type]
        )
    return ref_domain_type_2_module_id

# Reference layout caps used by ``cap_regions_to_reference_layout``.
# These must match the slot counts baked into
# ``get_reference_domain_type_2_module_ids`` (2 alpha slots → alpha_1/alpha_2,
# 1 each for beta/gamma/delta/epsilon).
REFERENCE_LAYOUT_CAPS = {
    "alpha": 2, "beta": 1, "gamma": 1, "delta": 1, "epsilon": 1,
}


def save_seq_to_regions_json(
    seq_2_regions: "dict[str, list[MappedRegion]]",
    out_path: "str | Path",
) -> Path:
    """Dump a ``{seq_id: [MappedRegion, ...]}`` mapping as portable JSON.

    External programs that don't have EnzymeExplorer importable can read
    this file without hitting the pickle/import problem; every region is
    represented as a plain dict via ``MappedRegion.to_dict``. The path
    is normalised to absolute and returned so callers can log it.

    The on-disk shape mirrors the in-memory mapping::

        {
          "<seq_id>": [
            {"module_id": "...", "domain": "...", "tmscore": 0.85,
             "residues_mapping": {"1": 10, "2": 11, ...},
             "aligned_template": "..."}
          ],
          ...
        }
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        seq_id: [r.to_dict() for r in regions]
        for seq_id, regions in seq_2_regions.items()
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return out_path.resolve()


def load_seq_to_regions_json(
    in_path: "str | Path",
) -> "dict[str, list[MappedRegion]]":
    """Reverse of :func:`save_seq_to_regions_json` — restore the full
    ``{seq_id: [MappedRegion, ...]}`` mapping from a JSON sidecar."""
    payload = json.loads(Path(in_path).read_text())
    return {
        seq_id: [MappedRegion.from_dict(r) for r in regions]
        for seq_id, regions in payload.items()
    }


def cap_regions_to_reference_layout(
    seq_2_regions: dict[str, "list[MappedRegion]"],
    existing_name_map: dict[str, str] | None = None,
    caps: dict[str, int] = REFERENCE_LAYOUT_CAPS,
) -> "tuple[dict[str, list[MappedRegion]], dict[str, str]]":
    """Trim each sequence's regions to the reference layout (top-N by tmscore
    per domain type) and re-issue contiguous per-type module-id suffixes.

    The fixed reference layout has 2 alpha + 1 each of {beta, gamma, delta,
    epsilon}; a query with more detected domains overflows into slots that
    don't exist (e.g. ``alpha_3``), which raises ``KeyError`` in
    :func:`get_structural_features`. This pre-trim keeps only the
    highest-tmscore region per overflow type and renames the survivors so
    their numeric suffixes stay in ``{0..caps[dt]-1}``, which is what
    ``get_structural_features`` adds 1 to when resolving ``alpha`` slots.

    Args:
        seq_2_regions: post-preprocessing query regions (domain types are
            already canonicalised to ``{alpha, beta, gamma, delta, epsilon}``).
        existing_name_map: the ``{preprocessed_module_id: original_module_id}``
            mapping returned by :func:`preprocess_domains_by_renaming_domain_types`,
            or ``None`` when no preprocessing happened. The returned name
            map composes through this so PDB-file lookups continue to
            resolve to the on-disk stem.
        caps: per-domain-type retention cap. Defaults to the reference
            layout.

    Returns:
        ``(capped_seq_2_regions, new_name_map)`` where ``new_name_map``
        maps every relabelled ``module_id`` to the *original* on-disk
        stem (i.e. the file name the foldseek symlink step expects).
    """
    from dataclasses import replace

    existing_name_map = existing_name_map or {}
    capped: dict[str, list[MappedRegion]] = {}
    new_name_map: dict[str, str] = {}
    n_dropped = 0
    for seq_id, regions in seq_2_regions.items():
        by_type: dict[str, list[MappedRegion]] = defaultdict(list)
        for r in regions:
            by_type[r.domain].append(r)
        kept: list[MappedRegion] = []
        for dt, rs in by_type.items():
            cap = caps.get(dt)
            rs_sorted = sorted(rs, key=lambda r: r.tmscore, reverse=True)
            if cap is None:
                kept.extend(rs_sorted)
                continue
            kept.extend(rs_sorted[:cap])
            n_dropped += max(0, len(rs_sorted) - cap)
        ctrs: dict[str, int] = defaultdict(int)
        relabelled: list[MappedRegion] = []
        for r in kept:
            new_mid = f"{seq_id}_{r.domain}_{ctrs[r.domain]}"
            ctrs[r.domain] += 1
            original = existing_name_map.get(r.module_id, r.module_id)
            new_name_map[new_mid] = original
            relabelled.append(replace(r, module_id=new_mid))
        capped[seq_id] = relabelled
    if n_dropped:
        logger.info(
            "cap_regions_to_reference_layout: dropped %d region(s) exceeding "
            "the reference layout caps %s", n_dropped, dict(caps),
        )
    return capped, new_name_map


def preprocess_domains_by_renaming_domain_types(
    seq_2_regions: dict[str, list[MappedRegion]], domain_type_preprocessing_config: dict[str, str]
) -> tuple[dict[str, list[MappedRegion]], dict[str, str]]:
    """Preprocess domains by renaming their types according to the provided configuration.

    Args:
        seq_2_regions (dict[str, list[MappedRegion]]): A dictionary mapping sequence ids to lists of domains.
        domain_type_preprocessing_config (dict[str, list[str]]): A dictionary mapping new domain types to lists of old domain types.

    Returns:
        tuple[dict[str, list[MappedRegion]], dict[str, str]]: A tuple containing the preprocessed sequence-to-domains mapping and the preprocessed name map.
    """
    renamed_modules_name_map = {}
    preprocessed_seq_2_regions = {}
    for seq_id, regions in seq_2_regions.items():
        preprocessed_regions = []
        domain_type_ctrs = defaultdict(int)
        for region in regions:
            new_domain_type = domain_type_preprocessing_config[region.domain]
            new_region = MappedRegion(
                module_id=f"{seq_id}_{new_domain_type}_{domain_type_ctrs[new_domain_type]}",
                domain=new_domain_type,
                tmscore=region.tmscore,
                residues_mapping=region.residues_mapping,
                aligned_template=region.domain,
            )
            new_region.aligned_template = region.domain
            preprocessed_regions.append(new_region)
            renamed_modules_name_map[new_region.module_id] = region.module_id
            domain_type_ctrs[new_domain_type] += 1
        preprocessed_seq_2_regions[seq_id] = preprocessed_regions

    return preprocessed_seq_2_regions, renamed_modules_name_map


def get_col_idx_for_structural_features(
    ref_domain_type_2_module_ids: dict[str, list[str]],
    feature_domain_types: list[str]
) -> dict[str, dict[str, int]]:
    """Initialize a numpy array to store structural features.

    Args:
        ref_domain_type_2_module_ids (dict[str, list[str]]): A dictionary mapping reference domain types to lists of module ids.

    Returns:
        dict[str, dict[str, int]]: A dictionary mapping domain types to dictionaries mapping module ids to column indices in the structural features array.
    """
    domain_type_2_module_id_2_col_idx = {}
    idx = 0
    for feature_domain_type in feature_domain_types:
        domain_type_2_module_id_2_col_idx[feature_domain_type] = {}
        for domain_name in ref_domain_type_2_module_ids[feature_domain_type]:
            domain_type_2_module_id_2_col_idx[feature_domain_type][domain_name] = idx
            idx += 1
    return domain_type_2_module_id_2_col_idx


def get_reference_sequence_col_indices(
    ref_seq_2_regions: dict[str, list[MappedRegion]],
    domain_type_2_ref_module_id_2_col_idx: dict[str, dict[str, int]],
) -> dict[str, list[int]]:
    """Get a mapping from reference sequence ids to lists of column indices in the structural features array corresponding to the domains present in the reference sequence.

    Args:
        ref_seq_2_regions (dict[str, list[MappedRegion]]): A dictionary mapping reference sequence ids to lists of reference domains.
        domain_type_2_ref_module_id_2_col_idx (dict[str, dict[str, int]]): A dictionary mapping domain types to dictionaries mapping reference module ids to column indices in the structural features array.

    Returns:
        dict[str, list[int]]: A dictionary mapping reference sequence ids to lists of column indices in the structural features array corresponding to the domains present in the reference sequence.
    """
    ref_seq_2_col_idx = defaultdict(list)
    for ref_seq in ref_seq_2_regions:
        for region in ref_seq_2_regions[ref_seq]:
            domain_type = region.domain
            if domain_type == "alpha":
                ref_seq_2_col_idx[ref_seq].append(
                    domain_type_2_ref_module_id_2_col_idx[domain_type + "_1"][
                        region.module_id
                    ]
                )
                ref_seq_2_col_idx[ref_seq].append(
                    domain_type_2_ref_module_id_2_col_idx[domain_type + "_2"][
                        region.module_id
                    ]
                )
            else:
                ref_seq_2_col_idx[ref_seq].append(
                    domain_type_2_ref_module_id_2_col_idx[domain_type][region.module_id]
                )
    return ref_seq_2_col_idx


def get_reference_domains_col_indices(
    domain_type_2_ref_module_id_2_col_idx: dict[str, dict[str, int]],
) -> dict[str, list[int]]:
    """Get a mapping from reference domain ids to column indices in the structural features array corresponding to the domains.

    Args:
        domain_type_2_ref_module_id_2_col_idx (dict[str, dict[str, int]]): A dictionary mapping domain types to dictionaries mapping reference module ids to column indices in the structural features array.

    Returns:
        dict[str, list[int]]: A dictionary mapping reference domain ids to lists of column indices in the structural features array corresponding to the domains.
    """
    ref_domain_module_id_2_col_idx = defaultdict(list)
    for domain_type in domain_type_2_ref_module_id_2_col_idx:
        for module_id in domain_type_2_ref_module_id_2_col_idx[domain_type]:
            ref_domain_module_id_2_col_idx[module_id].append(
                domain_type_2_ref_module_id_2_col_idx[domain_type][module_id]
            )
    return ref_domain_module_id_2_col_idx


def get_domain_type_2_col_idx_range(
    domain_type_2_ref_module_id_2_col_idx: dict[str, dict[str, int]],
) -> dict[str, tuple[int, int]]:
    """Get a mapping from domain types to tuples of (min column index, max column index) in the structural features array corresponding to the domains of the given type.

    Args:
        domain_type_2_ref_module_id_2_col_idx (dict[str, dict[str, int]]): A dictionary mapping domain types to dictionaries mapping reference module ids to column indices in the structural features array.

    Returns:
        dict[str, tuple[int, int]]: A dictionary mapping domain types to tuples of (min column index, max column index) in the structural features array corresponding to the domains of the given type.
    """
    domain_type_2_col_idx_range = {}
    for domain_type in domain_type_2_ref_module_id_2_col_idx:
        col_idxs = list(domain_type_2_ref_module_id_2_col_idx[domain_type].values())
        domain_type_2_col_idx_range[domain_type] = (min(col_idxs), max(col_idxs) + 1)
    return domain_type_2_col_idx_range


def get_structural_features(
    alignment_df: pd.DataFrame,
    query_sequence_ids: list[str],
    domain_type_2_ref_module_id_2_col_idx: dict[str, dict[str, int]],
) -> np.ndarray:
    """Fill the structural features array based on foldseek alignment results.

    Vectorised implementation: replaces V0's per-query boolean mask +
    iterrows loop with a single numpy slot assignment. Semantics match
    V0 (last-write-wins on duplicate (row, col); alignment_df is sorted
    descending by alntmscore by the caller).
    """
    n_features = sum(
        len(m) for m in domain_type_2_ref_module_id_2_col_idx.values()
    )
    structural_features = np.ones(
        (len(query_sequence_ids), n_features), dtype=np.float64
    )
    if len(alignment_df) == 0 or len(query_sequence_ids) == 0:
        return structural_features

    qid_to_row = {q: i for i, q in enumerate(query_sequence_ids)}

    df = alignment_df[
        (alignment_df["query_domain_type"] == alignment_df["target_domain_type"])
        & alignment_df["query_seq_id"].isin(qid_to_row)
    ]
    if len(df) == 0:
        return structural_features

    # Resolve final domain_type: for alpha → "alpha_{int(query.split('_')[-1])+1}".
    final_dt = df["query_domain_type"].astype(object).copy()
    is_alpha = (df["query_domain_type"] == "alpha").to_numpy()
    if is_alpha.any():
        alpha_suffix = (
            df.loc[is_alpha, "query"].str.rsplit("_", n=1).str[-1].astype(int) + 1
        )
        final_dt.loc[is_alpha] = "alpha_" + alpha_suffix.astype(str)

    flat_lookup: dict[tuple[str, str], int] = {}
    for dt, mp in domain_type_2_ref_module_id_2_col_idx.items():
        for mid, ci in mp.items():
            flat_lookup[(dt, mid)] = ci

    final_dt_arr = final_dt.to_numpy()
    target_arr = df["target"].to_numpy()
    cols = np.empty(len(df), dtype=np.int64)
    for i in range(len(df)):
        key = (final_dt_arr[i], target_arr[i])
        try:
            cols[i] = flat_lookup[key]
        except KeyError:
            raise KeyError(
                f"Domain {key[1]} of type {key[0]} not found in reference layout."
            )

    rows = df["query_seq_id"].map(qid_to_row).to_numpy(dtype=np.int64)
    vals = (1.0 - df["alntmscore"].to_numpy(dtype=np.float64))

    # Last-write-wins: same iteration order as V0.
    flat = rows * n_features + cols
    dedup = pd.DataFrame({"flat": flat, "val": vals}).drop_duplicates(
        subset=["flat"], keep="last"
    )
    flat_final = dedup["flat"].to_numpy()
    structural_features[
        flat_final // n_features, flat_final % n_features
    ] = dedup["val"].to_numpy()
    return structural_features


SSR_PARALLEL_THRESHOLD = 500


def _ssr_one_pdb(pdb_path_str: str) -> tuple[str, set[str]]:
    """Compute the SS-residue set for ONE PDB. Run inside a spawn worker."""
    from pymol import cmd, stored  # type: ignore

    p = Path(pdb_path_str)
    obj_name = p.stem
    cmd.load(str(p.absolute()), obj_name)
    stored.residues_set = set()
    cmd.iterate(f"({obj_name} & ss H+S)", "stored.residues_set.add(resi)")
    result = stored.residues_set.copy()
    stored.residues_set = None
    cmd.delete(obj_name)
    return obj_name, result if result else set()


def save_file_to_all_residues(
    secondary_structure_residues_path: Path,
    pdb_files: list[Path],
    domain_templates: list[dict[str, Path]],
):
    """Compute and persist the SS-residues map for every PDB.

    Below `SSR_PARALLEL_THRESHOLD` PDBs: in-process serial path (PyMOL
    in the parent — fast for small inputs because spawn-pool startup
    dominates). Above the threshold: parallel SSR routed through the
    centralized pool service. Requires an active `pool_session` for the
    parallel path; the serial path runs unconditionally.
    """
    import pickle

    inputs: list[Path] = []
    for pdb_file in pdb_files:
        if Path(pdb_file).exists():
            inputs.append(Path(pdb_file))
    for template in domain_templates:
        tpath = Path(template["path"])
        if tpath.exists() and tpath not in inputs:
            inputs.append(tpath)

    Path(secondary_structure_residues_path).parent.mkdir(
        parents=True, exist_ok=True
    )

    if len(inputs) < SSR_PARALLEL_THRESHOLD:
        # Serial in-process path.
        from pymol import cmd  # type: ignore
        from enzymeexplorer.src.structure_processing.structural_algorithms import (
            get_all_residues_per_file,
        )

        logger.info(
            "SSR (serial in-proc, %d < threshold %d): %d files",
            len(inputs),
            SSR_PARALLEL_THRESHOLD,
            len(inputs),
        )
        file_2_all_residues = get_all_residues_per_file(inputs, cmd)
    else:
        svc = require_active_service()
        logger.info(
            "SSR (parallel via pool service, n_jobs=%d): %d files",
            svc.n_jobs,
            len(inputs),
        )
        chunksize = max(1, len(inputs) // (svc.n_jobs * 20))
        file_2_all_residues = {}
        for stem, residues in svc.imap_unordered(
            _ssr_one_pdb,
            [str(p) for p in inputs],
            chunksize=chunksize,
        ):
            if residues:
                file_2_all_residues[stem] = residues

    with open(secondary_structure_residues_path, "wb") as f:
        pickle.dump(file_2_all_residues, f)
    logger.info(
        "SSR: wrote %s (%d entries)",
        secondary_structure_residues_path,
        len(file_2_all_residues),
    )


def get_pdb_files(
    needed_proteins_csv_path: str, csv_id_column: str, input_directory: Path
) -> list[Path]:
    if needed_proteins_csv_path is not None:
        proteins_df = pd.read_csv(needed_proteins_csv_path)
        relevant_protein_ids = set(proteins_df[csv_id_column].unique())
    else:
        relevant_protein_ids = set(
            [filepath.stem for filepath in input_directory.glob("*.pdb")]
        )

    pdb_files = []
    logger.info(
        f"Filtering PDB files in {input_directory} to only include those specified in {needed_proteins_csv_path}"
    )
    for protein_id in relevant_protein_ids:
        pdb_path = input_directory / f"{protein_id}.pdb"
        if not pdb_path.exists():
            logger.warning(
                f"PDB file for {protein_id} not found at {pdb_path}, skipping this protein."
            )
            continue
        pdb_files.append(pdb_path.absolute())

    for filename in [pdb_file.stem for pdb_file in pdb_files]:
        filename_regex = "[a-zA-Z0-9_]+"
        if not re.fullmatch(filename_regex, filename):
            raise ValueError(
                f"Filename {filename} does not match the expected pattern {filename_regex}, which may cause issues with PyMOL selection syntax. Consider renaming this file."
            )
    return pdb_files


def filter_pdb_files_by_foldseek_alignments(
    pdb_files: list[Path],
    domain_templates: list[dict[str, Path | str]],
    batch_size: int = 1000,
    e_value: float = 10,
    cov_mode: int = 1,
    coverage: float = 0.5,
) -> dict[str, list[Path]]:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        target_dir = tmpdir_path / "target"
        target_dir.mkdir()
        store_templates(domain_templates, target_dir)

        filtered_pdb_file_names: defaultdict[str, set] = defaultdict(set)
        query_dir = tmpdir_path / "query"
        query_dir.mkdir()
        for idx in range(0, len(pdb_files), batch_size):
            batch_dir = query_dir / f"batch_{idx}"
            batch_dir.mkdir()
            batch_pdb_files = pdb_files[idx : min(idx + batch_size, len(pdb_files))]
            for pdb_file in batch_pdb_files:
                os.symlink(pdb_file, batch_dir / pdb_file.name)
            alignment_df = FoldseekWrapper().easy_search(
                query_dir=str(batch_dir),
                target_dir=str(target_dir),
                tmp_dir=str(tmpdir_path / "tmp_foldseek"),
                output=str(tmpdir_path / "foldseek_output.tsv"),
                max_seqs=batch_size * 2,
                e_value=e_value,
                cov_mode=cov_mode,
                coverage=coverage,
            )
            for domain_template in domain_templates:
                filtered_pdb_file_names[str(domain_template["name"])].update(
                    set(
                        alignment_df[alignment_df["target"] == domain_template["name"]][
                            "query"
                        ].unique()
                    )
                )

        return {
            domain_name: [
                pdb_file
                for pdb_file in pdb_files
                if pdb_file.stem in filtered_pdb_files
            ]
            for domain_name, filtered_pdb_files in filtered_pdb_file_names.items()
        }


def filter_domains_by_foldseek_alignments(
    filename_2_known_regions_completed_confident: dict[str, list[MappedRegion]],
    supported_domains: list[str],
    domain_templates: list[dict[str, Path | str]],
    domain_pdbs_root: Path,
    e_value: float = 10,
) -> dict[str, list[MappedRegion]]:
    all_regions = set(
        [
            region.module_id
            for regions in filename_2_known_regions_completed_confident.values()
            for region in regions
        ]
    )
    high_conf_regions = set(
        [
            region.module_id
            for regions in filename_2_known_regions_completed_confident.values()
            for region in regions
            if region.tmscore >= 0.4
        ]
    )
    filtered_domain_pdb_files = set()
    filtered_regions = set()
    filename_2_known_regions_completed_confident_filtered = defaultdict(list)
    for domain in supported_domains:
        domain_pdbs_dir = domain_pdbs_root / domain
        if domain_pdbs_dir.exists():
            domain_pdb_files = [
                path.absolute()
                for path in domain_pdbs_dir.glob(f"*.pdb")
                if path.stem in set([region for region in all_regions])
            ]
            if domain_pdb_files:
                filtered_domains = set(
                    [
                        pdb_file
                        for pdb_files in filter_pdb_files_by_foldseek_alignments(
                            domain_pdb_files,
                            [
                                domain_template
                                for domain_template in domain_templates
                                if domain_template["name"] == domain
                            ],
                            batch_size=3000,
                            e_value=e_value,
                            cov_mode=2,
                            coverage=0.6,
                        ).values()
                        for pdb_file in pdb_files
                    ]
                )
                filtered_domains.update(
                    [domain_pdbs_dir / f"{region}.pdb" for region in high_conf_regions]
                )

                filtered_domain_regions_ids = set(
                    [filtered_domain.stem for filtered_domain in filtered_domains]
                )
                filtered_domain_pdb_files.update(
                    [filtered_domain.name for filtered_domain in filtered_domains]
                )
                filtered_regions.update(filtered_domain_regions_ids)
                for domain_pdb_file in domain_pdb_files:
                    if domain_pdb_file not in filtered_domains:
                        os.remove(domain_pdb_file)

    root_pdb_files = [
        path.absolute()
        for path in domain_pdbs_root.glob(f"*.pdb")
        if path.stem in all_regions
    ]
    for domain_pdb_file in root_pdb_files:
        if domain_pdb_file.name not in filtered_domain_pdb_files:
            logger.info(
                f"Removing domain pdb file {domain_pdb_file} due to lack of foldseek alignment."
            )
            os.remove(domain_pdb_file)

    renamed = {}
    for filename in filename_2_known_regions_completed_confident:
        filtered_regions_for_filename = sorted(
            [
                region
                for region in filename_2_known_regions_completed_confident[filename]
                if region.module_id in filtered_regions
            ],
            key=lambda r: r.module_id,
        )
        if len(filtered_regions_for_filename) == 0:
            continue
        group_by_domain_type = defaultdict(list)
        for region in filtered_regions_for_filename:
            group_by_domain_type[region.domain].append(copy.deepcopy(region))
        regions_for_file = []
        for domain in group_by_domain_type:
            if (len(group_by_domain_type[domain]) - 1) != int(
                group_by_domain_type[domain][-1].module_id.split("_")[-1]
            ):
                for region in group_by_domain_type[domain]:
                    old_module_id = region.module_id
                    region.module_id = f"{filename}_{region.domain}_{group_by_domain_type[domain].index(region)}"
                    os.rename(
                        domain_pdbs_root / domain / f"{old_module_id}.pdb",
                        domain_pdbs_root / domain / f"{region.module_id}.pdb",
                    )
                    os.rename(
                        domain_pdbs_root / f"{old_module_id}.pdb",
                        domain_pdbs_root / f"{region.module_id}.pdb",
                    )
                    regions_for_file.append(region)
                    renamed[old_module_id] = region.module_id
            else:
                for region in group_by_domain_type[domain]:
                    regions_for_file.append(region)

        filename_2_known_regions_completed_confident_filtered[filename] = sorted(
            regions_for_file, key=lambda r: r.module_id
        )
    return filename_2_known_regions_completed_confident_filtered


def store_domain_separately(
    filename_2_known_regions_completed_confident: dict[str, list[MappedRegion]],
    supported_domains: list[str],
    detected_regions_root_path: Path,
):
    domain_2_regions_completed_confident = defaultdict(list)
    for (
        filename,
        protein_regions,
    ) in filename_2_known_regions_completed_confident.items():
        for region in protein_regions:
            domain_2_regions_completed_confident["all"].append((filename, region))
            domain_2_regions_completed_confident[region.domain].append(
                (filename, region)
            )
    with open(
        detected_regions_root_path / "regions_completed_very_confident_all_ALL.pkl",
        "wb",
    ) as f:
        pickle.dump(domain_2_regions_completed_confident["all"], f)
    for domain_name in supported_domains:
        with open(
            detected_regions_root_path
            / f"regions_completed_very_confident_{domain_name}_ALL.pkl",
            "wb",
        ) as f:
            pickle.dump(domain_2_regions_completed_confident[domain_name], f)


def store_templates(
    domain_templates: list[dict[str, Path | str]],
    output_path: Path,
):
    for template in domain_templates:
        try:
            output_pdb_path = output_path / f"{template['name']}.pdb"
            cmd.delete(f"{template['name']}")
            cmd.delete(f'{template["name"]}_domain')
            cmd.load(template["path"], str(f'{template["name"]}_domain'))
            cmd.select(
                f'{template["name"]}',
                f"{template['name']}_domain & {template['residues']}",
            )
            cmd.save(f"{output_pdb_path}", f'{template["name"]}')
            cmd.delete(f"{template['name']}")
            cmd.delete(f'{template["name"]}_domain')
        except Exception as e:
            logger.error(
                f"Error storing domain {template['name']} from file {template['path']}: {e}"
            )


def store_domain(
    filename_region: tuple[str, MappedRegion],
    domains_output_path: Path,
):
    try:
        filename, region = filename_region
        PATH = Path(domains_output_path / f"{region.domain}")
        mapped_residues = list(set(region.residues_mapping.keys()))
        cmd.delete(filename)
        cmd.load(f"{filename}.pdb")
        logger.info(
            f"{region.module_id} {filename} & resi {compress_selection_list(mapped_residues)}",
        )
        cmd.select(
            f"{region.module_id}",
            f"{filename} & resi {compress_selection_list(mapped_residues)}",
        )
        cmd.save(f"{PATH}/{region.module_id}.pdb", f"{region.module_id}")
        cmd.save(
            f"{domains_output_path}/{region.module_id}.pdb",
            f"{region.module_id}",
        )
        cmd.delete(filename)
        return True
    except Exception as e:
        logger.error(
            f"Error storing domain {region.module_id} from file {filename}: {e}"
        )
        return False


def store_domains(
    filename_2_known_regions_completed_confident: dict[str, list[MappedRegion]],
    supported_domains: list[str],
    domains_output_path: Path,
    n_jobs: int = 1,  # kept for backwards compatibility; ignored
):
    """Save detected-domain PDBs in parallel via the pool service.

    `store_domain` is PyMOL-heavy (`cmd.load`/`cmd.select`/`cmd.save`),
    so this MUST run on a spawn-based pool — fork-based workers
    deadlock once the parent has loaded any PDB. The `n_jobs` argument
    is retained only for backwards compatibility; the active pool
    service determines actual parallelism.
    """
    if not domains_output_path.exists():
        domains_output_path.mkdir(parents=True)
    for domain_name in supported_domains:
        PATH = domains_output_path / f"{domain_name}"
        if not PATH.exists():
            PATH.mkdir(parents=True)
    store_domain_partial = partial(
        store_domain, domains_output_path=domains_output_path
    )
    filename_domain_tuples = [
        (filename, region)
        for (
            filename,
            regions,
        ) in filename_2_known_regions_completed_confident.items()
        for region in regions
    ]
    if not filename_domain_tuples:
        return
    require_active_service().map(store_domain_partial, filename_domain_tuples)


def plot_aligned_domains(
    filename_2_known_regions_completed_confident: dict[str, list[MappedRegion]],
    supported_domains: list[str],
    save_dir: Path,
):
    """
    Helper function plotting TM-scores of detected domains on x-axis and
    the number of residues assigned to the domain object on y-axis
    """
    execution_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for domain_this in supported_domains:
        all_tmscores_and_mappings = [
            (region.tmscore, region.residues_mapping)
            for regions in filename_2_known_regions_completed_confident.values()
            for region in regions
            if region.domain == domain_this
        ]
        if len(all_tmscores_and_mappings) > 0:
            plt.figure(figsize=(17, 9))
            results_of_mapping = [
                (tmscore, len(mapping))
                for tmscore, mapping in all_tmscores_and_mappings
                if mapping is not None
            ]
            mapping_lenghts = list(map(lambda x: x[1], results_of_mapping))
            plt.scatter(list(map(lambda x: x[0], results_of_mapping)), mapping_lenghts)
            plt.xticks(np.arange(0, 1, 0.05), rotation=90)
            plt.yticks(
                np.arange(min(mapping_lenghts) - 10, max(mapping_lenghts) + 10, 5)
            )
            plt.xlabel("TM-score", fontsize=11)
            plt.ylabel("Number of residues assigned to the domain", fontsize=11)
            plt.title(f"{domain_this} domain detections", fontsize=14)
            plt.savefig(
                save_dir / f"{domain_this}_detections_{execution_timestamp}.png"
            )
            plt.show()


def detect_domains_roughly(
    domain_to_pdb_files: dict[str, list[Path]],
    file_2_all_residues_mapping: dict[str, set[str]],
    domain_templates: list[dict],
    args: configargparse.Namespace,
    iteration: int = 0,
) -> dict[str, list[MappedRegion]]:
    """
    Detects protein domains in multiple structures based on alignment scores and domain-specific thresholds.

    :param file_2_all_residues_mapping: A dictionary mapping file identifiers to sets of residue sequences present in those files
    :param domain_templates: A list of dictionaries containing domain template information
    :param output_root: The root directory where output images and serialized results will be saved
    :param args: arguments containing parameters number of iterations and flags for storing intermediate results

    :return: A dictionary mapping each filename to a list of known MappedRegion objects representing the detected
             reliable domains, while ensuring that no overlaying domains are included.
    """
    file_2_possible_regions: dict = defaultdict(list)
    for domain_template in domain_templates:
        domain_this = domain_template["name"]
        logger.info("Started detection of domain %s", domain_this)
        start_t = time.time()
        file_2_tmscore_residues_domain = get_alignments(
            domain_to_pdb_files[domain_this],
            domain_template=domain_template,
            file_2_current_residues=file_2_all_residues_mapping,
            n_jobs=args.n_jobs,
        )
        logger.info(
            "Detection of %s domain. Execution took %d seconds",
            domain_this,
            time.time() - start_t,
        )

        num_of_new_detections = 0
        for sequence_id, current_detections in file_2_tmscore_residues_domain.items():
            logger.info(sequence_id)
            for i, (tm_score, res_mapping) in enumerate(current_detections):
                logger.info(f"tm_score: {tm_score:.2f}")
                logger.info(f"len of res_mapping: {len(res_mapping)}")
                if (
                    len(res_mapping) >= domain_template["thresholds"]["min_align_len"]
                    and tm_score >= domain_template["thresholds"]["tmscore"]
                ):
                    num_of_new_detections += 1
                    file_2_possible_regions[sequence_id].append(
                        MappedRegion(
                            module_id=f"{sequence_id}_{domain_this}_{i}",
                            domain=domain_this,
                            tmscore=tm_score,
                            residues_mapping=res_mapping,
                        ),
                    )

        logger.info(
            "Detected %d potential %s domains in iteration %d",
            num_of_new_detections,
            domain_this,
            iteration,
        )

    return file_2_possible_regions


def is_similar_to_known_region(
    region_known: MappedRegion,
    region_new: MappedRegion,
    threshold_recall_threshold: float = 0.5,
) -> bool:
    """
    Checks whether two regions overlap sufficiently based on a recall threshold.

    :param region_known: The known region to compare against
    :param region_new: The new region to be compared
    :param threshold_recall_threshold: The minimum recall threshold for the regions to be considered similar, defaults to 0.5

    :return: True if the overlap between the two regions meets or exceeds the threshold, otherwise False
    """
    mapped_residues_known = set(region_known.residues_mapping.keys())
    mapped_residues_new = set(region_new.residues_mapping.keys())
    if len(mapped_residues_new) == 0 or len(mapped_residues_known) == 0:
        return False
    return (
        len(mapped_residues_new.intersection(mapped_residues_known))
        / len(mapped_residues_new)
        >= threshold_recall_threshold
    )


def _first_atom_bfactors(sequence_id: str) -> dict[int, float]:
    """Return {resi: b_factor_of_first_atom} via a single PyMOL `iterate`.

    Avoids BioPython's slow PDBParser. The first atom per residue in
    standard PDB ordering is N (matches V0's `for atom in residue: break`
    on AlphaFold structures). Filtering the iterate to `name N` visits
    only ~one atom per residue.
    """
    from pymol import cmd  # type: ignore
    from enzymeexplorer.src.structure_processing.structural_algorithms import (
        exists_in_pymol,
    )

    loaded = False
    if not exists_in_pymol(cmd, sequence_id):
        if not os.path.exists(f"{sequence_id}.pdb"):
            raise FileNotFoundError(
                f"{sequence_id}.pdb (cwd={os.getcwd()})"
            )
        cmd.load(f"{sequence_id}.pdb", sequence_id)
        loaded = True

    out: dict[int, float] = {}
    cmd.iterate(
        f"({sequence_id} & name N)",
        "out[int(resi)] = float(b)",
        space={"out": out},
    )
    if loaded:
        cmd.delete(sequence_id)
    if max(out.values()) <= 1:  # ESMFold pLDDT is in b-factor field, scaled by [0,1]
        out = {r: b * 100 for r, b in out.items()}
    return out


def get_confident_af_residues(
    sequence_id: str, confidence_threshold: int = 70
) -> set[int]:
    """Set of residues with CA B-factor (pLDDT) above `confidence_threshold`.

    PyMOL-based replacement for V0's BioPython PDBParser re-read.
    AlphaFold writes per-atom rounded b-factors that aren't perfectly
    uniform per residue; matching V0's "first atom of each residue" via
    `name N` filter keeps the threshold-cut bit-equivalent.
    """
    bf = _first_atom_bfactors(sequence_id)
    return {r for r, b in bf.items() if b >= confidence_threshold}


def get_all_confidence_values(sequence_id: str) -> list[float]:
    """All per-residue confidence values (first-atom B-factors)."""
    bf = _first_atom_bfactors(sequence_id)
    return list(bf.values())


def get_confident_residue_mappings(
    filename_2_known_regions_completed: dict[str, list[MappedRegion]],
    file_2_all_residues: dict[str, set[str]],
    domain_2_threshold: dict[str, dict[str, int]],
) -> dict[str, list[MappedRegion]]:
    filename_2_known_regions_completed_confident = {}
    for filename, regions in tqdm(
        filename_2_known_regions_completed.items(), desc="Filtering confident residues"
    ):
        conf_residues = get_confident_af_residues(filename)
        if len(conf_residues) < 0.6 * len(file_2_all_residues[filename]):
            logger.warning(
                f"Too few confident residues for {filename}, leaving top-80% most confident residues"
            )
            all_confidence_values = get_all_confidence_values(filename)
            conf_residues = get_confident_af_residues(
                filename, np.percentile(all_confidence_values, 20)
            )
        new_regions = []
        for mapped_region_init in regions:
            new_residues_mapping = {
                res: res_dom
                for res, res_dom in mapped_region_init.residues_mapping.items()
                if res in conf_residues
            }
            if (
                len(new_residues_mapping)
                >= domain_2_threshold[mapped_region_init.domain]["min_align_len"]
            ):
                new_regions.append(
                    MappedRegion(  # pylint: disable=R0801
                        module_id=mapped_region_init.module_id,
                        domain=mapped_region_init.domain,
                        tmscore=mapped_region_init.tmscore,
                        residues_mapping=new_residues_mapping,
                    )
                )
        filename_2_known_regions_completed_confident[filename] = new_regions
    return filename_2_known_regions_completed_confident


def pick_disjoint_domains(sorted_domains: list[MappedRegion]) -> list[MappedRegion]:
    if len(sorted_domains) == 0:
        return []
    picked = []
    for domain in sorted_domains:
        pick = True
        replace = None
        for picked_domain in picked:
            if is_similar_to_known_region(
                picked_domain, domain, threshold_recall_threshold=0.2
            ):
                if picked_domain.tmscore < domain.tmscore + 0.01 and len(picked_domain.residues_mapping) * 1.15 < len(domain.residues_mapping):
                    replace = picked_domain
                else:
                    pick = False
                    break
        if pick:
            if replace is not None:
                picked = [d if d != replace else domain for d in picked]
            else:
                picked.append(domain)
    return picked