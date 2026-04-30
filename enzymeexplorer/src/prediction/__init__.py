"""Shared prediction pipeline used by the CLI scripts and the FastAPI app.

Submodules:
    inputs    — load sequences from FASTA/CSV
    embeddings — PLM embedding helpers wrapping ``esm_transformer_utils``
    domains   — orchestrate domain detection + foldseek alignment in-process
    ensemble  — load fold-bundled classifiers and average ``predict_proba``
    tiers     — assign confidence-tier labels from confidence_tiers.csv
    pipeline  — high-level entry points (``predict_with_structures``,
                ``predict_sequences_only``)
"""
