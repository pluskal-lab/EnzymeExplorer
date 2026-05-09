"""Shared prediction pipeline used by the CLI scripts and the FastAPI app.

Submodules:
    inputs    — load sequences from FASTA/CSV
    embeddings — PLM embedding helpers wrapping ``esm_transformer_utils``
    domains   — orchestrate domain detection + foldseek alignment in-process
    ensemble  — load fold-bundled classifiers and average ``predict_proba``
    calibration — apply per-class beta calibrators from fit_summary.csv
    pipeline  — high-level entry points (``predict_with_structures``,
                ``predict_sequences_only``)
"""
