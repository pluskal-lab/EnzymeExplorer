"""In-process orchestration of domain detection + foldseek alignment.

End-to-end: given a directory of structures and a list of protein IDs to score,
runs ``run_domain_detection`` to detect TPS-family domains, then
``run_structural_feature_computation`` to TM-align each detection against a set
of training-time reference domains. The structural-features array returned by
the latter is *already* in the column layout the trained classifiers consume,
so we hand it straight through without re-deriving the feature matrix.
"""

from __future__ import annotations

import logging
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np  # type: ignore
import pandas as pd  # type: ignore


# NB: ``run_domain_detection`` and ``run_structural_feature_computation`` are
# imported lazily inside ``detect_and_align_domains`` because they pull in
# PyMOL — which is heavy and fails on environments without the right libstdc++.
# Keeping them lazy means callers that only need the sequence-only flow can
# import this package without paying that cost.

logger = logging.getLogger(__name__)


@dataclass
class DomainAlignmentResult:
    """Outputs of :func:`detect_and_align_domains`.

    Attributes:
        detected_domains: ``{pdb_id: list[MappedRegion]}`` from the domain
            detector — useful for debugging/inspection.
        structural_features: ``[n_query, n_features]`` array of ``1 - tmscore``
            values produced by ``get_structural_features``. Unaligned cells
            default to ``1.0``. Column layout matches the classifier's
            training-time layout (same reference-domain pickle).
        query_seq_ids: row labels for ``structural_features``.
        alignment_df: per-(query_module, ref_module) foldseek alignment table
            produced by ``get_structural_features`` — needed by the API to
            look up each detected domain's closest known reference for the
            TMalign-based superposition.
        domain_structures_dir: directory holding ``<module_id>.pdb`` files for
            every detected domain, written during the detection step.
    """

    detected_domains: dict
    structural_features: np.ndarray
    query_seq_ids: list[str]
    alignment_df: pd.DataFrame
    domain_structures_dir: Path


def detect_and_align_domains(
    *,
    structures_dir: str | Path,
    protein_ids: list[str],
    reference_domains_pickle: str | Path,
    reference_domains_structures_dir: str | Path,
    workdir: str | Path | None = None,
    n_jobs: int = 10,
    keep_intermediate: bool = False,
    prefilter_pdbs_by_foldseek: bool = True,
    prefilter_e_value: float = 10.0,
    is_bfactor_confidence: bool = True,
) -> DomainAlignmentResult:
    """Detect domains and align them to known reference domains.

    Defaults are tuned for prediction on small batches:
    ``prefilter_pdbs_by_foldseek=True`` cuts the (query × template) TMalign
    workload by ~5-10× by skipping pairs with no plausible foldseek
    alignment. The training pipeline keeps prefiltering off (its YAML config)
    because at training time we want exhaustive sensitivity, but at predict
    time the speedup is large and the recall loss negligible (a query with
    *no* foldseek hit to a TPS-family template is essentially never a TPS).

    A scratch directory is created under ``workdir`` (defaulting to a temp
    directory) and removed on exit unless ``keep_intermediate`` is True.
    """
    structures_dir = Path(structures_dir).absolute()
    reference_domains_pickle = Path(reference_domains_pickle).absolute()
    reference_domains_structures_dir = Path(reference_domains_structures_dir).absolute()

    from enzymeexplorer.src.structure_processing.domain_detections import (
        run_domain_detection,
    )
    from enzymeexplorer.src.structure_processing.get_structural_features import (
        run_structural_feature_computation,
    )

    cleanup = False
    if workdir is None:
        workdir = Path(tempfile.mkdtemp(prefix="enzyme_explorer_predict_"))
        cleanup = not keep_intermediate
    workdir = Path(workdir).absolute()
    workdir.mkdir(parents=True, exist_ok=True)

    try:
        # Domain detection requires a CSV of needed proteins; synthesize one
        # so callers can pass a plain list of IDs.
        needed_proteins_csv = workdir / "needed_proteins.csv"
        pd.DataFrame({"id": list(protein_ids)}).to_csv(needed_proteins_csv, index=False)

        detections_pickle = workdir / "detected_domains.pkl"
        detected_domain_structures_root = workdir / "detected_domain_structures"

        logger.info(
            "Running domain detection on %d proteins (prefilter_by_foldseek=%s)",
            len(protein_ids),
            prefilter_pdbs_by_foldseek,
        )
        detected_domains = run_domain_detection(
            input_directory_with_structures=structures_dir,
            needed_proteins_csv_path=needed_proteins_csv,
            csv_id_column="id",
            detections_output_path=detections_pickle,
            detected_regions_root_path=detected_domain_structures_root,
            domains_output_path=detected_domain_structures_root,
            n_jobs=n_jobs,
            secondary_structure_residues_path=str(workdir / "secondary_structure_residues.pkl"),
            prefilter_pdbs_by_foldseek=prefilter_pdbs_by_foldseek,
            prefilter_e_value=prefilter_e_value,
            is_bfactor_confidence=is_bfactor_confidence,
        )

        # Drop proteins whose detection list ended up empty after post-filtering
        # so we never feed a feature row to the classifier for a protein that
        # has no genuine detections. This also guarantees the
        # structural-features rows produced below correspond 1:1 with
        # proteins that have at least one detected domain.
        detected_domains = {
            pid: regions
            for pid, regions in detected_domains.items()
            if regions
        }

        if not detected_domains:
            logger.warning("No domains detected for any of the input proteins.")
            return DomainAlignmentResult(
                detected_domains={},
                structural_features=np.zeros((0, 0), dtype=np.float32),
                query_seq_ids=[],
                alignment_df=pd.DataFrame(),
                domain_structures_dir=detected_domain_structures_root,
            )

        # Re-pickle the filtered detections so get_structural_features only
        # processes proteins that actually have detections.
        with detections_pickle.open("wb") as f:
            pickle.dump(detected_domains, f)

        feature_dir = workdir / "structural_features"
        logger.info(
            "Running foldseek alignment of detected domains vs reference domains"
        )
        result = run_structural_feature_computation(
            query_domains_file_path=detections_pickle,
            reference_domains_file_path=reference_domains_pickle,
            query_domains_structures_directory=detected_domain_structures_root,
            reference_domains_structures_directory=reference_domains_structures_dir,
            output_directory=feature_dir,
        )
        query_seq_ids = list(result["query_seq_ids"])
        # Sanity check: every row in structural_features should map to a
        # protein that actually has detections.
        stray = [pid for pid in query_seq_ids if pid not in detected_domains]
        if stray:
            raise RuntimeError(
                f"{len(stray)} structural-feature row(s) reference proteins "
                f"with no detections (e.g. {stray[:3]}); domain pickle and "
                f"structural-features output disagree."
            )
        return DomainAlignmentResult(
            detected_domains=detected_domains,
            structural_features=np.asarray(
                result["structural_features"], dtype=np.float32
            ),
            query_seq_ids=query_seq_ids,
            alignment_df=result["alignment_df"],
            domain_structures_dir=detected_domain_structures_root,
        )
    finally:
        if cleanup and workdir.exists():
            from shutil import rmtree

            rmtree(workdir, ignore_errors=True)


def load_detected_domains(path: str | Path) -> dict:
    """Helper for callers that only have the ``detected_domains.pkl``."""
    with Path(path).open("rb") as f:
        return pickle.load(f)
