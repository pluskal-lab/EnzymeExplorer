"""FastAPI app — single-protein TPS prediction API.

Endpoint URLs and response shapes are kept on the legacy contract so the
published ``notebooks/EnzymeExplorer_*.ipynb`` colabs continue to work without
modification:

* ``POST /detect_domains/`` — multipart upload (``file=<pdb>``) + form field
  ``is_bfactor_confidence``. Returns the legacy 6-key payload.
* ``POST /predict_tps/`` — same upload form. Returns
  ``{"predictions": [{<raw class name>: prob, ...}]}`` where the class names
  are the *raw* model labels (``isTPS``, ``precursor substr``, substrate
  SMILES) — the notebooks rely on those keys.
* ``GET /download_pdb/{aligned_pdb_name}`` — serves a per-domain PDB written
  during ``/detect_domains/`` and removes it after the response.

Domain *type* prediction is currently stubbed (``domain_type_predictions: {}``)
pending the new domain-type predictor; the notebooks gate their visualization
on this field, so an empty dict cleanly skips that branch without errors.
Other metadata fields previously sourced from legacy auxiliary pickles
(``closest_id_reaction_types``, ``closest_id_kingdom``,
``whole_structure_domain_config``) are returned as ``None`` for the same
reason — the gate also keeps them from being dereferenced.
"""

from __future__ import annotations

# --- Critical import order ---
# PyMOL must load *before* numpy/pandas/BioPython so the conda env's
# libstdc++.so.6 (with GLIBCXX_3.4.30, required by libvtkm_cont-1.8.so.1) is
# the one resolved by the dynamic loader. If pandas/numpy load first they
# pull in the system libstdc++ and the later PyMOL import fails. The
# standalone ``detect_domains`` CLI works without setting LD_LIBRARY_PATH for
# the same reason — it imports PyMOL on line 11 of its module.
from pymol import cmd as _pymol_cmd  # type: ignore  # noqa: F401, E402

import logging  # noqa: E402
import pickle  # noqa: E402
import re  # noqa: E402
import tempfile  # noqa: E402
import uuid  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from shutil import copyfile, rmtree  # noqa: E402

import pandas as pd  # type: ignore  # noqa: E402
from Bio import SeqIO  # type: ignore  # noqa: E402
from fastapi import (  # noqa: E402
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.responses import FileResponse  # noqa: E402

from enzymeexplorer.src.evaluation.classes import SHORT_TO_SMILES  # noqa: E402
from enzymeexplorer.src.prediction.domains import (  # noqa: E402
    detect_and_align_domains,
)
from enzymeexplorer.src.prediction.embeddings import load_plm_embedder  # noqa: E402
from enzymeexplorer.src.prediction.ensemble import load_fold_bundle  # noqa: E402
from enzymeexplorer.src.prediction.pipeline import (  # noqa: E402
    DEFAULT_PLM_DOMAINS_BUNDLE,
    DEFAULT_PLM_ONLY_BUNDLE,
    DEFAULT_REFERENCE_DOMAINS_PICKLE,
    DEFAULT_REFERENCE_DOMAINS_STRUCTURES_DIR,
    predict_with_structures,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


# ---------------------------------------------------------------------------
# Startup state
# ---------------------------------------------------------------------------
PLM_MODEL = "esm-1v-finetuned-subseq"
TEMP_DIR = Path("_temp")
TEMP_DIR.mkdir(exist_ok=True)

# Per-protein metadata used to enrich /detect_domains responses (kingdom and
# reaction types of the closest known domain's parent protein). Loaded once
# from the martsDB reactions CSV. Falls back to empty maps if the file is
# missing — clients see ``None`` for these fields.
DATASET_CSV = Path("data/martsDB_reactions_2026_02_22_preprocessed.csv")
_DATASET_ID_COL = "Enzyme_marts_ID"


def _load_dataset_metadata(
    csv_path: Path,
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Build ``(id_2_kingdom, id_2_reaction_types)`` from the martsDB CSV.

    The CSV has multiple rows per protein (one per substrate/product
    reaction). We take the first non-null Kingdom for each ID and the unique
    sorted list of ``Type`` values.
    """
    if not csv_path.exists():
        logger.warning(
            "Dataset CSV %s not found — closest_id_kingdom / "
            "closest_id_reaction_types will be returned as None.",
            csv_path,
        )
        return {}, {}
    df = pd.read_csv(csv_path, usecols=[_DATASET_ID_COL, "Type", "Kingdom"])
    id_2_kingdom = (
        df.dropna(subset=["Kingdom"])
        .drop_duplicates(subset=[_DATASET_ID_COL])
        .set_index(_DATASET_ID_COL)["Kingdom"]
        .to_dict()
    )
    id_2_reaction_types = (
        df.dropna(subset=["Type"])
        .groupby(_DATASET_ID_COL)["Type"]
        .apply(lambda s: sorted(set(s.astype(str))))
        .to_dict()
    )
    return id_2_kingdom, id_2_reaction_types


logger.info("Loading per-protein metadata from %s…", DATASET_CSV)
ID_2_KINGDOM, ID_2_REACTION_TYPES = _load_dataset_metadata(DATASET_CSV)
logger.info(
    "Loaded metadata for %d proteins (kingdom) / %d proteins (reaction types)",
    len(ID_2_KINGDOM),
    len(ID_2_REACTION_TYPES),
)

logger.info("Loading PLM embedder (%s)…", PLM_MODEL)
embedder = load_plm_embedder(PLM_MODEL)

# Eagerly load the bundles so request latency is dominated by the actual
# work, not pickle.load. The PLM_Domains bundle is the largest (~3 GB) but
# is shared across all requests.
logger.info("Loading PLM_Domains fold bundle…")
_PLM_DOMAINS_FOLDS = load_fold_bundle(DEFAULT_PLM_DOMAINS_BUNDLE)
logger.info("Loading PLM-only fold bundle…")
_PLM_ONLY_FOLDS = load_fold_bundle(DEFAULT_PLM_ONLY_BUNDLE)

app = FastAPI(title="EnzymeExplorer prediction API")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@dataclass
class _MotifDetection:
    start: int
    end: int
    motif: str
    class_tps: str


# Regex motifs from the legacy implementation, unchanged.
_MOTIF_PATTERNS: list[tuple[str, str, str]] = [
    ("DD..D", "DDxxD", "class I"),
    ("[ND]D..[ST]...E", "NSE/DTE", "class I"),
    ("D.DD", "DxDD", "class II"),
]


def _detect_motifs(sequence: str) -> list[dict]:
    out: list[dict] = []
    for pattern, label, class_tps in _MOTIF_PATTERNS:
        for match in re.finditer(pattern, sequence):
            out.append(
                {
                    "start": match.start() + 1,
                    "end": match.end() + 1,
                    "motif": label,
                    "class_tps": class_tps,
                }
            )
    return out


def _normalise_pdb_id(filename: str) -> str:
    """Mirror the legacy ID normalisation: drop ``(...)`` substrings, dashes
    and whitespace from the filename stem."""
    stem = Path(filename).stem
    stem = re.sub(r"\(.*?\)", "", stem)
    return "".join(stem.replace("-", "").split())


def _save_pdb_upload(contents: bytes, filename: str) -> tuple[Path, str]:
    """Save the upload into a fresh temp workdir; return (input_dir, pdb_id)."""
    pdb_id = _normalise_pdb_id(filename)
    if not pdb_id:
        raise HTTPException(status_code=400, detail="filename has no usable stem")
    workdir = Path(tempfile.mkdtemp(prefix="api_"))
    input_dir = workdir / "input"
    input_dir.mkdir(parents=True)
    (input_dir / f"{pdb_id}.pdb").write_bytes(contents)
    return workdir, pdb_id


def _sequence_from_pdb(pdb_path: Path) -> str:
    records = list(SeqIO.parse(str(pdb_path), "pdb-atom"))
    seqs = list({str(r.seq) for r in records if str(r.seq)})
    if not seqs:
        raise HTTPException(
            status_code=400,
            detail=f"Could not extract any sequence from {pdb_path.name}",
        )
    if len(seqs) > 1:
        logger.warning(
            "Multiple distinct chains in %s; using the first", pdb_path.name
        )
    return seqs[0]


def _domain_to_dict(region) -> dict:
    return {
        "module_id": getattr(region, "module_id", None),
        "domain": getattr(region, "domain", None),
        "tmscore": getattr(region, "tmscore", None),
        "residues_mapping": dict(getattr(region, "residues_mapping", {})),
    }


def _legacy_predictions_from_table(table: pd.DataFrame) -> dict:
    """Wide-form prediction table → ``{raw_class_name: prob}`` dict using the
    legacy keys (``isTPS``, ``precursor substr``, substrate SMILES) the
    notebooks consume."""
    if table.empty:
        return {}
    row = table.iloc[0].to_dict()
    out: dict[str, float] = {}
    for col, val in row.items():
        if not isinstance(col, str) or not col.endswith("_score"):
            continue
        short_name = col[: -len("_score")]
        legacy_name = SHORT_TO_SMILES.get(short_name, short_name)
        out[legacy_name] = float(val)
    return out


def _load_secondary_structure_residues(
    workdir: Path, pdb_id: str
) -> list[str] | None:
    """Read back the per-protein residue set the detector cached in the
    workdir."""
    ssr_path = workdir / "_work" / "secondary_structure_residues.pkl"
    if not ssr_path.exists():
        return None
    with ssr_path.open("rb") as f:
        ssr = pickle.load(f)
    if pdb_id not in ssr:
        return None
    return sorted(ssr[pdb_id], key=lambda r: int(r) if str(r).lstrip("-").isdigit() else 0)


def _closest_known_per_detection(
    alignment_df: pd.DataFrame, detected_for_pdb: list
) -> dict[str, dict]:
    """For each detection, return the row of the foldseek alignment table
    with the highest ``alntmscore``: ``{module_id: {target, alntmscore,
    target_domain_type, ...}}``.
    """
    if alignment_df is None or alignment_df.empty:
        return {}
    detection_module_ids = {
        getattr(d, "module_id", None) for d in detected_for_pdb
    }
    detection_module_ids.discard(None)
    sub = alignment_df[alignment_df["query"].isin(detection_module_ids)]
    if sub.empty:
        return {}
    out: dict[str, dict] = {}
    for module_id, group in sub.groupby("query"):
        best = group.loc[group["alntmscore"].idxmax()]
        out[str(module_id)] = best.to_dict()
    return out


def _superpose_and_save(
    detected_pdb: Path,
    closest_pdb: Path,
    out_path: Path,
) -> float | None:
    """TMalign-superpose ``detected_pdb`` onto ``closest_pdb`` inside PyMOL
    (via psico) and write a combined ``ref + rotated mobile`` PDB to
    ``out_path``. Returns the TMalign score, or ``None`` if alignment failed.
    """
    from psico.fitting import tmalign  # type: ignore

    suffix = uuid.uuid4().hex[:8]
    ref_obj = f"ref_{suffix}"
    mobile_obj = f"mob_{suffix}"
    aln_obj = f"aln_{suffix}"
    combined_obj = f"cmb_{suffix}"
    try:
        _pymol_cmd.load(str(closest_pdb), ref_obj)
        _pymol_cmd.load(str(detected_pdb), mobile_obj)
        try:
            tm_score = float(
                tmalign(mobile_obj, ref_obj, object=aln_obj, quiet=1)
            )
        except Exception as exc:  # pragma: no cover — best-effort logging
            logger.warning(
                "TMalign failed for %s vs %s: %s",
                detected_pdb.name,
                closest_pdb.name,
                exc,
            )
            return None
        _pymol_cmd.create(combined_obj, f"({ref_obj}) or ({mobile_obj})")
        _pymol_cmd.save(str(out_path), combined_obj)
        return tm_score
    finally:
        for obj in (ref_obj, mobile_obj, aln_obj, combined_obj):
            try:
                _pymol_cmd.delete(obj)
            except Exception:
                pass


def _build_aligned_pdb_filepaths(
    detected_for_pdb: list,
    domain_structures_dir: Path,
    alignment_df: pd.DataFrame,
    reference_domains_dir: Path,
) -> dict | None:
    """Build the legacy ``aligned_pdb_filepaths`` mapping with real TMalign
    superpositions of each detected domain onto its closest known reference.

    For every detection:

      1. Find the closest known reference module via the foldseek alignment
         table (highest ``alntmscore`` row keyed by ``query==module_id``).
      2. Superpose the detected domain onto that reference using psico's
         ``tmalign`` (PyMOL-internal — no external binary needed).
      3. Save the combined ref + rotated-mobile PDB to ``TEMP_DIR/<module_id>.pdb``
         so ``/download_pdb/<module_id>`` can serve it.

    ``closest_id_reaction_types`` and ``closest_id_kingdom`` are read from
    the in-memory maps populated at startup from
    ``data/EnzymeExplorer_Dataset.csv`` (matched by the closest known
    domain's parent protein ID). ``whole_structure_domain_config`` was
    previously sourced from a separate auxiliary pickle that's no longer
    maintained — it stays ``None`` until the new domain-type predictor lands;
    the notebooks gate on ``domain_type_predictions`` (currently stubbed)
    before dereferencing it.
    """
    if not detected_for_pdb:
        return None
    closest = _closest_known_per_detection(alignment_df, detected_for_pdb)
    out: dict = {}
    for det in detected_for_pdb:
        module_id = getattr(det, "module_id", None)
        if module_id is None:
            continue
        detected_pdb = domain_structures_dir / f"{module_id}.pdb"
        if not detected_pdb.exists():
            continue

        best = closest.get(module_id)
        closest_module_id = (
            str(best["target"]) if best is not None else None
        )
        # ``target_seq_id`` is populated by ``get_foldseek_alignment_df`` —
        # use it directly rather than splitting the module_id, which is
        # brittle for IDs that themselves contain underscores
        # (e.g. ``marts_E00001`` vs the legacy single-token ``1ps1``).
        closest_pdb_id = (
            str(best["target_seq_id"]) if best is not None else None
        )
        closest_domain_type = (
            best.get("target_domain_type") if best is not None else None
        )

        out_path = TEMP_DIR / f"{module_id}.pdb"
        tm_score: float | None = None
        if closest_module_id:
            closest_pdb = reference_domains_dir / f"{closest_module_id}.pdb"
            if closest_pdb.exists():
                tm_score = _superpose_and_save(
                    detected_pdb, closest_pdb, out_path
                )
        if tm_score is None:
            # Fallback: serve the unaligned detected domain so /download_pdb
            # still has something to return.
            copyfile(detected_pdb, out_path)
            tm_score = (
                float(best["alntmscore"]) if best is not None else None
            )

        reaction_types = (
            ID_2_REACTION_TYPES.get(closest_pdb_id) if closest_pdb_id else None
        )
        if reaction_types is not None:
            # Mirror the legacy normalisation — uppercase "Class" was used
            # for some entries and notebooks expected the lowercase form.
            reaction_types = [t.replace("Class", "class") for t in reaction_types]
        out[module_id] = {
            "closest_known_domain_pdb_id": closest_pdb_id,
            "whole_structure_domain_config": None,
            "closest_domain_type": closest_domain_type
            or getattr(det, "domain", None),
            "closest_id_reaction_types": reaction_types,
            "closest_id_kingdom": (
                ID_2_KINGDOM.get(closest_pdb_id) if closest_pdb_id else None
            ),
            "tm_score": tm_score,
            "aligned_pdb_name": module_id,
        }
    return out or None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/detect_domains/")
async def detect_domains_endpoint(
    file: UploadFile = File(...),
    is_bfactor_confidence: bool = Form(...),
):
    contents = await file.read()
    workdir, pdb_id = _save_pdb_upload(contents, file.filename or "")
    pdb_path = workdir / "input" / f"{pdb_id}.pdb"
    try:
        result = detect_and_align_domains(
            structures_dir=workdir / "input",
            protein_ids=[pdb_id],
            reference_domains_pickle=DEFAULT_REFERENCE_DOMAINS_PICKLE,
            reference_domains_structures_dir=DEFAULT_REFERENCE_DOMAINS_STRUCTURES_DIR,
            workdir=workdir / "_work",
            keep_intermediate=True,  # we read SSR + per-domain PDBs back
            is_bfactor_confidence=is_bfactor_confidence,
        )
        detected = result.detected_domains.get(pdb_id, [])
        domains = {pdb_id: [_domain_to_dict(d) for d in detected]}

        # Sequence-derived motifs — same regexes as the legacy implementation.
        sequence = _sequence_from_pdb(pdb_path)
        motif_detections = _detect_motifs(sequence) if detected else None

        return {
            "domains": domains,
            "secondary_structure_residues": _load_secondary_structure_residues(
                workdir, pdb_id
            ),
            "motif_detections": motif_detections,
            # The new pipeline does foldseek alignment internally but the
            # per-detection comparison table the legacy field carried isn't
            # directly consumed by the notebooks; left as None.
            "comparison_to_known_domains": None,
            # STUB until the new domain-type predictor lands. The notebooks
            # gate ``show_domains`` on this field being truthy, so an empty
            # dict cleanly skips that branch.
            "domain_type_predictions": {},
            "aligned_pdb_filepaths": _build_aligned_pdb_filepaths(
                detected,
                domain_structures_dir=result.domain_structures_dir,
                alignment_df=result.alignment_df,
                reference_domains_dir=Path(
                    DEFAULT_REFERENCE_DOMAINS_STRUCTURES_DIR
                ),
            ),
        }
    finally:
        rmtree(workdir, ignore_errors=True)


@app.get("/download_pdb/{aligned_pdb_name}")
async def download_aligned_pdb(
    aligned_pdb_name: str, background_tasks: BackgroundTasks
):
    aligned_pdb_path = TEMP_DIR / f"{aligned_pdb_name}.pdb"
    if not aligned_pdb_path.exists():
        # The legacy app returned this exact JSON shape on miss.
        return {"error": "File not found"}
    background_tasks.add_task(aligned_pdb_path.unlink)
    return FileResponse(
        aligned_pdb_path,
        media_type="application/octet-stream",
        filename=aligned_pdb_path.name,
    )


@app.post("/predict_tps/")
async def predict_tps_endpoint(
    file: UploadFile = File(...),
    is_bfactor_confidence: bool = Form(...),
):
    contents = await file.read()
    workdir, pdb_id = _save_pdb_upload(contents, file.filename or "")
    pdb_path = workdir / "input" / f"{pdb_id}.pdb"
    try:
        sequence = _sequence_from_pdb(pdb_path)
        sequences_df = pd.DataFrame({"id": [pdb_id], "sequence": [sequence]})

        plm_domains_table, plm_only_table = predict_with_structures(
            sequences_df,
            structures_dir=workdir / "input",
            reference_domains_pickle=DEFAULT_REFERENCE_DOMAINS_PICKLE,
            reference_domains_structures_dir=(
                DEFAULT_REFERENCE_DOMAINS_STRUCTURES_DIR
            ),
            plm_domains_bundle_path=DEFAULT_PLM_DOMAINS_BUNDLE,
            plm_only_bundle_path=DEFAULT_PLM_ONLY_BUNDLE,
            embedder=embedder,
            plm_only_embedder=embedder,
            plm_model=PLM_MODEL,
            plm_only_model=PLM_MODEL,
            workdir=workdir / "_work",
            keep_intermediate=False,
        )
        if not plm_domains_table.empty:
            preds = _legacy_predictions_from_table(plm_domains_table)
        elif not plm_only_table.empty:
            preds = _legacy_predictions_from_table(plm_only_table)
        else:
            raise HTTPException(
                status_code=500, detail="Prediction returned no rows"
            )
        return {"predictions": [preds]}
    finally:
        rmtree(workdir, ignore_errors=True)
