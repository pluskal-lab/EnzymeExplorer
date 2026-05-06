"""Catalytic-role analysis — which TPS domain types drive each reaction class.

Hypotheses
----------
H1 (Class 1)
    Class 1 reactions are determined by the **alpha** domain alone.
    beta / gamma / delta / epsilon contribute nothing to the
    Class-1 reaction outcome.

H2 (Class 2)
    Class 2 reactions are determined by the beta / delta / gamma /
    epsilon domains; the alpha domain contributes nothing. Within each
    Class-2 enzyme the participating non-alpha domains contribute equally.

Preprocessing applied to every analysis
---------------------------------------
* Detected domains of type ``zeta`` are ignored entirely.
* ``alpha`` and ``ids`` detections are merged under a single canonical
  type ``"alpha"``.
* MartsDB rows whose ``OriginalType`` is in ``{pt, psy, sqs}`` are
  dropped (these are head-group / chain-elongation enzymes, not
  cyclases).
* MartsDB rows whose substrate SMILES contains a ``.`` (multi-substrate
  reactions) are dropped.

Three similarity targets
------------------------
For each pair of sequences we compute, restricted to the target reaction
class:

  reaction_jaccard
      Jaccard similarity over the set of OriginalType labels of each
      sequence's reactions of that class.

  substrate_max_tanimoto
      Max Tanimoto (Morgan-FP r=2, 2048 bits) over (substrate_a × substrate_b)
      pairs from class-target reactions of each sequence.

  product_max_tanimoto_shared_ot
      Max Tanimoto over (product_a × product_b) pairs **restricted to
      products of the SAME OriginalType** in both sequences. This
      controls for substrate size — pairs that don't share any
      OriginalType in the target class get NaN. The reaction-similarity
      claim is strongest on this target because it isolates "given the
      same substrate class, do these enzymes make similar products?"

Per-axis correlations
---------------------
For each domain axis (alpha, beta, gamma, delta, epsilon) we compute
the pairwise TM-score where both sequences possess that domain and then
fit:

  marginal Pearson r vs each similarity target
  partial Pearson r  vs each target, controlling linearly for the
                      OTHER axes in the analysis (multi-control
                      residualisation).

The hypothesis is supported when the *expected* axis has high partial r
and the *other* axes' partial r values collapse to ~0.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

logger = logging.getLogger(__name__)


# Analysis-specific canonicalisation. ``zeta`` is intentionally
# excluded (mapped to ``None``) so its detected domains are dropped from
# this analysis. ``alpha`` and ``ids`` collapse to a single ``"alpha"``.
ANALYSIS_CANONICAL_DOMAIN_TYPE = {
    "alpha": "alpha",
    "ids": "alpha",
    "zeta": None,
    "beta": "beta",
    "gamma": "gamma",
    "delta": "delta",
    "epsilon": "epsilon",
}

# OriginalType values dropped from every analysis (head-group / chain-
# elongation enzymes, not cyclases).
EXCLUDED_ORIGINAL_TYPES = frozenset({"pt", "psy", "sqs"})

DOMAIN_AXES_ALL = ("alpha", "beta", "gamma", "delta", "epsilon")


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def build_filtered_reactions(martsdb_csv: str | Path) -> pd.DataFrame:
    """Apply the per-row preprocessing filters described in the module docstring."""
    df = pd.read_csv(martsdb_csv)
    sub_col = "SMILES_substrate_canonical_no_stereo"
    prod_col = "SMILES_product_canonical_no_stereo"
    for c in (sub_col, prod_col, "OriginalType", "Class", "Enzyme_marts_ID"):
        if c not in df.columns:
            raise ValueError(f"required MartsDB column missing: {c}")

    n0 = len(df)
    df = df[~df["OriginalType"].isin(EXCLUDED_ORIGINAL_TYPES)]
    n1 = len(df)

    multi_subs_mask = df[sub_col].fillna("").astype(str).str.contains(".", regex=False)
    df = df[~multi_subs_mask]
    n2 = len(df)

    df = df[df["Class"].isin([1, 2])]
    df["Class"] = df["Class"].astype(int)
    n3 = len(df)
    logger.info(
        "Reaction filter: %d → drop pt/psy/sqs → %d → drop multi-substrate → %d → keep Class 1/2 → %d",
        n0, n1, n2, n3,
    )
    return df.reset_index(drop=True)


def build_sequence_summary(
    metadata_df: pd.DataFrame,
    filtered_reactions: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate per-sequence info: domain modules + per-class reaction tags / SMILES."""
    md = metadata_df.copy()
    md["analysis_canonical"] = md["domain_type"].map(ANALYSIS_CANONICAL_DOMAIN_TYPE)
    md = md[md["analysis_canonical"].notna()].copy()

    rows: list[dict] = []
    for seq_id, group in md.groupby("seq_id"):
        kingdom = group["kingdom"].iloc[0] if "kingdom" in group else ""
        modules: dict[str, list[str]] = defaultdict(list)
        for _, r in group.iterrows():
            modules[r["analysis_canonical"]].append(r["module_id"])
        rec = {"seq_id": seq_id, "kingdom": kingdom}
        for ax in DOMAIN_AXES_ALL:
            rec[f"n_{ax}"] = len(modules.get(ax, []))
            mids = modules.get(ax, [])
            rec[f"module_{ax}"] = mids[0] if len(mids) == 1 else None
        rows.append(rec)
    seq_df = pd.DataFrame(rows).set_index("seq_id")

    sub_col = "SMILES_substrate_canonical_no_stereo"
    prod_col = "SMILES_product_canonical_no_stereo"
    for cls in (1, 2):
        cls_rxn = filtered_reactions[filtered_reactions["Class"] == cls]
        ot_per_seq = (
            cls_rxn.groupby("Enzyme_marts_ID")["OriginalType"].agg(set).to_dict()
        )
        sub_per_seq = (
            cls_rxn.groupby("Enzyme_marts_ID")[sub_col].agg(set).to_dict()
        )
        # Group products by EXACT SUBSTRATE SMILES rather than by OriginalType.
        # The OT-conditioned analysis still works as a fallback (kept under
        # ``products_by_ot_class{N}`` for backwards compat), but the headline
        # product-similarity metric conditions on shared *substrate* now —
        # finer-grained than shared OriginalType (which lumps e.g. multiple
        # sesquiterpene substrates into one bucket).
        prod_by_ot: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        prod_by_substrate: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        for _, r in cls_rxn.iterrows():
            seq = r["Enzyme_marts_ID"]
            prod_by_ot[seq][r["OriginalType"]].add(str(r[prod_col]))
            prod_by_substrate[seq][str(r[sub_col])].add(str(r[prod_col]))
        seq_df[f"originaltypes_class{cls}"] = seq_df.index.map(
            lambda s: frozenset(ot_per_seq.get(s, set()))
        )
        seq_df[f"substrates_class{cls}"] = seq_df.index.map(
            lambda s: frozenset(str(x) for x in sub_per_seq.get(s, set()))
        )
        seq_df[f"products_by_ot_class{cls}"] = seq_df.index.map(
            lambda s: {k: frozenset(v) for k, v in prod_by_ot.get(s, {}).items()}
        )
        seq_df[f"products_by_substrate_class{cls}"] = seq_df.index.map(
            lambda s: {k: frozenset(v) for k, v in prod_by_substrate.get(s, {}).items()}
        )
    seq_df = seq_df.reset_index()
    return seq_df


# ---------------------------------------------------------------------------
# Fingerprint precomputation + similarity helpers
# ---------------------------------------------------------------------------

def _smiles_to_fp(smiles: str, radius: int = 2, n_bits: int = 2048):
    from rdkit import Chem  # type: ignore
    from rdkit.Chem import AllChem  # type: ignore

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def precompute_substrate_fps(
    seq_df: pd.DataFrame, target_class: int,
) -> dict[str, list]:
    fps_per_seq: dict[str, list] = {}
    for _, row in seq_df.iterrows():
        smis = row[f"substrates_class{target_class}"]
        fps = []
        for s in smis:
            fp = _smiles_to_fp(s)
            if fp is not None:
                fps.append(fp)
        fps_per_seq[row["seq_id"]] = fps
    return fps_per_seq


# ---------------------------------------------------------------------------
# Substrate-specific similarity matrices (carbon-count, atom-pair, MCS)
# ---------------------------------------------------------------------------

# All three substrate similarity methods. The Morgan-FP-Tanimoto target was
# replaced because the isoprenoid-substrate space saturates Morgan-2
# fingerprints (see plots.plot_role_axes_scatter for diagnostic).
SUBSTRATE_SIMILARITY_METHODS = ("mcs",)


def _heavy_carbon_count(smiles: str) -> int | None:
    from rdkit import Chem  # type: ignore

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "C")


def _heavy_atom_count(smiles: str) -> int | None:
    from rdkit import Chem  # type: ignore

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return mol.GetNumHeavyAtoms()


def _atom_pair_fp(smiles: str, n_bits: int = 2048):
    from rdkit import Chem  # type: ignore
    from rdkit.Chem import AllChem  # type: ignore

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=n_bits)


def build_substrate_similarity_matrices(
    unique_smiles: list[str],
    *,
    methods: tuple[str, ...] = SUBSTRATE_SIMILARITY_METHODS,
    mcs_timeout: int = 30,
) -> dict[str, np.ndarray]:
    """Build square similarity matrices over the unique substrate SMILES list.

    Returns ``{method: matrix}`` where each matrix is shape (n, n) with
    rows/cols indexed by the SMILES list. Missing values (parse failures
    etc.) become ``np.nan``. With only ~30 unique substrates in the
    martsDB dataset, even MCS computation is trivial (≤ 1 minute).
    """
    from rdkit import Chem, DataStructs  # type: ignore
    from rdkit.Chem import rdFMCS  # type: ignore

    n = len(unique_smiles)
    out: dict[str, np.ndarray] = {}

    if "carbon_count" in methods:
        ccs = [_heavy_carbon_count(s) for s in unique_smiles]
        M = np.full((n, n), np.nan)
        for i in range(n):
            ci = ccs[i]
            if ci is None:
                continue
            for j in range(n):
                cj = ccs[j]
                if cj is None:
                    continue
                if ci == 0 and cj == 0:
                    M[i, j] = 1.0
                else:
                    M[i, j] = 1.0 - abs(ci - cj) / max(ci, cj)
        out["carbon_count"] = M
        logger.info(
            "Built carbon-count similarity matrix (n=%d, range=[%.3f, %.3f])",
            n, np.nanmin(M), np.nanmax(M),
        )

    if "atom_pair" in methods:
        fps = [_atom_pair_fp(s) for s in unique_smiles]
        M = np.full((n, n), np.nan)
        for i in range(n):
            if fps[i] is None:
                continue
            for j in range(n):
                if fps[j] is None:
                    continue
                M[i, j] = float(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
        out["atom_pair"] = M
        logger.info(
            "Built atom-pair Tanimoto matrix (n=%d, range=[%.3f, %.3f])",
            n, np.nanmin(M), np.nanmax(M),
        )

    if "mcs" in methods:
        atom_counts = [_heavy_atom_count(s) for s in unique_smiles]
        mols = [Chem.MolFromSmiles(s) for s in unique_smiles]
        M = np.full((n, n), np.nan)
        for i in range(n):
            if mols[i] is None or atom_counts[i] is None:
                continue
            M[i, i] = 1.0
            for j in range(i + 1, n):
                if mols[j] is None or atom_counts[j] is None:
                    continue
                res = rdFMCS.FindMCS(
                    [mols[i], mols[j]], timeout=mcs_timeout,
                )
                if res.canceled or res.numAtoms == 0:
                    sim = 0.0
                else:
                    union_atoms = atom_counts[i] + atom_counts[j] - res.numAtoms
                    sim = res.numAtoms / max(union_atoms, 1)
                M[i, j] = sim
                M[j, i] = sim
        out["mcs"] = M
        logger.info(
            "Built MCS Tanimoto matrix (n=%d, range=[%.3f, %.3f])",
            n, np.nanmin(M), np.nanmax(M),
        )

    return out


def collect_unique_substrate_smiles(
    seq_df: pd.DataFrame, target_classes: tuple[int, ...] = (1, 2),
) -> list[str]:
    """Sorted list of unique substrate SMILES across the requested classes."""
    seen: set[str] = set()
    for cls in target_classes:
        col = f"substrates_class{cls}"
        for s in seq_df[col]:
            seen.update(s)
    return sorted(seen)


def precompute_substrate_indices(
    seq_df: pd.DataFrame, target_class: int, smiles_to_idx: dict[str, int],
) -> dict[str, list[int]]:
    """For each sequence, list of integer indices into the unique-SMILES table."""
    out: dict[str, list[int]] = {}
    for _, row in seq_df.iterrows():
        smis = row[f"substrates_class{target_class}"]
        out[row["seq_id"]] = [
            smiles_to_idx[s] for s in smis if s in smiles_to_idx
        ]
    return out


def _max_matrix_lookup(
    idxs_a: list[int], idxs_b: list[int], matrix: np.ndarray,
) -> float:
    """Max similarity over the cross-product (idxs_a × idxs_b)."""
    if not idxs_a or not idxs_b:
        return float("nan")
    best = float("-inf")
    seen = False
    for ia in idxs_a:
        for ib in idxs_b:
            v = matrix[ia, ib]
            if not np.isnan(v):
                seen = True
                if v > best:
                    best = v
    return best if seen else float("nan")


def precompute_product_fps_by_ot(
    seq_df: pd.DataFrame, target_class: int,
) -> dict[str, dict[str, list]]:
    out: dict[str, dict[str, list]] = {}
    for _, row in seq_df.iterrows():
        per_ot: dict[str, list] = {}
        for ot, smis in row[f"products_by_ot_class{target_class}"].items():
            fps = [_smiles_to_fp(s) for s in smis]
            per_ot[ot] = [f for f in fps if f is not None]
        out[row["seq_id"]] = per_ot
    return out


def precompute_product_fps_by_substrate(
    seq_df: pd.DataFrame, target_class: int,
) -> dict[str, dict[str, list]]:
    """Per-sequence Morgan FPs of products grouped by their substrate SMILES.

    Used so that pairwise product similarity can be computed only over
    products of *the same substrate* in both sequences — finer-grained
    than the shared-OriginalType condition.
    """
    out: dict[str, dict[str, list]] = {}
    for _, row in seq_df.iterrows():
        per_sub: dict[str, list] = {}
        for sub_smi, smis in row[f"products_by_substrate_class{target_class}"].items():
            fps = [_smiles_to_fp(s) for s in smis]
            per_sub[sub_smi] = [f for f in fps if f is not None]
        out[row["seq_id"]] = per_sub
    return out


def _max_tanimoto(fps1: list, fps2: list) -> float:
    from rdkit import DataStructs  # type: ignore

    if not fps1 or not fps2:
        return float("nan")
    best = 0.0
    for f1 in fps1:
        sims = DataStructs.BulkTanimotoSimilarity(f1, fps2)
        if sims:
            best = max(best, max(sims))
    return float(best)


def _jaccard(s1: frozenset, s2: frozenset) -> float:
    if not s1 and not s2:
        return float("nan")
    return len(s1 & s2) / len(s1 | s2)


# ---------------------------------------------------------------------------
# Pair feature computation
# ---------------------------------------------------------------------------

def compute_pair_features(
    seq_df: pd.DataFrame,
    pairwise_tm: dict[tuple[str, str], float],
    *,
    target_class: int,
    domain_axes: tuple[str, ...],
    substrate_indices_per_seq: dict[str, list[int]],
    substrate_similarity_matrices: dict[str, np.ndarray],
    product_fps_by_substrate: dict[str, dict[str, list]],
) -> pd.DataFrame:
    """For every ordered-by-index pair (i<j) in ``seq_df``, compute axis TMs
    and the similarity targets restricted to the target reaction class.

    Substrate similarities are looked up from precomputed (unique-SMILES ×
    unique-SMILES) matrices via per-sequence index lists.

    Product similarity is conditioned on **shared substrate** (same
    canonical SMILES in both sequences' Class-N reaction set) rather than
    shared OriginalType — finer control for substrate identity.
    """
    seqs = seq_df.reset_index(drop=True)
    n = len(seqs)
    seq_ids = seqs["seq_id"].tolist()
    module_per_axis = {
        ax: seqs[f"module_{ax}"].tolist() for ax in domain_axes
    }
    ot_per_seq = seqs[f"originaltypes_class{target_class}"].tolist()
    sub_idx_per_seq = [substrate_indices_per_seq.get(s, []) for s in seq_ids]
    rows: list[dict] = []
    for i in range(n):
        for j in range(i + 1, n):
            row = {"seq_a": seq_ids[i], "seq_b": seq_ids[j]}
            for ax in domain_axes:
                a = module_per_axis[ax][i]
                b = module_per_axis[ax][j]
                if a is None or b is None:
                    row[f"tm_{ax}"] = float("nan")
                elif a == b:
                    row[f"tm_{ax}"] = 1.0
                else:
                    key = (a, b) if a <= b else (b, a)
                    row[f"tm_{ax}"] = pairwise_tm.get(key, float("nan"))

            row["reaction_jaccard"] = _jaccard(ot_per_seq[i], ot_per_seq[j])

            for method, matrix in substrate_similarity_matrices.items():
                row[f"substrate_{method}_sim"] = _max_matrix_lookup(
                    sub_idx_per_seq[i], sub_idx_per_seq[j], matrix,
                )

            # Product similarity: per shared substrate, take the MAX Tanimoto
            # over the cross-product of products that A and B make from THAT
            # substrate; then average those per-substrate maxes across all
            # shared substrates. If they share NO substrates, similarity =
            # NaN — those pairs are excluded from the correlation analysis
            # (every downstream consumer dropna's or uses pairwise-complete).
            prod_a = product_fps_by_substrate.get(seq_ids[i], {})
            prod_b = product_fps_by_substrate.get(seq_ids[j], {})
            shared_substrates = set(prod_a.keys()) & set(prod_b.keys())
            if shared_substrates:
                per_substrate_sims = []
                for sub_smi in shared_substrates:
                    sim = _max_tanimoto(prod_a.get(sub_smi, []),
                                        prod_b.get(sub_smi, []))
                    if not np.isnan(sim):
                        per_substrate_sims.append(sim)
                if per_substrate_sims:
                    row["product_avg_tanimoto_shared_substrate"] = float(
                        np.mean(per_substrate_sims)
                    )
                else:
                    row["product_avg_tanimoto_shared_substrate"] = float("nan")
                row["n_shared_substrates"] = len(shared_substrates)
            else:
                row["product_avg_tanimoto_shared_substrate"] = float("nan")
                row["n_shared_substrates"] = 0

            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Correlations (marginal + multi-control partial Pearson)
# ---------------------------------------------------------------------------

def _partial_pearson_multivariate(
    x: np.ndarray, y: np.ndarray, controls: np.ndarray,
):
    """Pearson r of x and y after linearly residualising both on ``controls``.

    Falls back to ordinary Pearson when ``controls`` is empty.
    """
    from scipy.stats import pearsonr  # type: ignore

    if controls.ndim == 1:
        controls = controls.reshape(-1, 1)
    if controls.size == 0:
        return pearsonr(x, y)
    Z = np.column_stack([np.ones_like(x), controls])
    beta_x, *_ = np.linalg.lstsq(Z, x, rcond=None)
    beta_y, *_ = np.linalg.lstsq(Z, y, rcond=None)
    rx = x - Z @ beta_x
    ry = y - Z @ beta_y
    return pearsonr(rx, ry)


SIMILARITY_TARGETS = (
    ("reaction_jaccard", "reaction Jaccard"),
    ("substrate_mcs_sim", "substrate MCS Tanimoto"),
    ("product_avg_tanimoto_shared_substrate",
     "product avg-Tanimoto (per shared substrate, then mean)"),
)


def correlation_summary(
    pair_df: pd.DataFrame, axes: tuple[str, ...],
    targets: tuple[tuple[str, str], ...] = SIMILARITY_TARGETS,
    *,
    weights_per_target: dict[str, str] | None = None,
) -> pd.DataFrame:
    """For each (axis, target), report n, marginal r/p, Spearman r/p, and
    partial r/p with all OTHER axes in ``axes`` held constant.

    When ``weights_per_target`` is supplied (mapping ``target_col`` →
    ``weight_column_name``), all statistics are computed weighted: rows with
    weight ≤ 0 are dropped, and Pearson / partial / R² use weighted least-
    squares. Spearman is left unweighted (rank-based) and just reported on
    the surviving rows. ``n`` is the raw row count, ``n_eff`` is the
    effective sample size ``(Σw)² / Σw²``.
    """
    from scipy.stats import pearsonr, spearmanr  # type: ignore

    rows: list[dict] = []
    for axis in axes:
        for target_col, target_label in targets:
            other_axes = [a for a in axes if a != axis]
            need_cols = [f"tm_{axis}", target_col] + [f"tm_{a}" for a in other_axes]
            wcol = (weights_per_target or {}).get(target_col)
            extra_cols = [wcol] if wcol else []
            sub = pair_df[need_cols + extra_cols].dropna()
            if wcol:
                sub = sub[sub[wcol] > 0]
            if len(sub) < 5:
                rows.append({
                    "axis": axis, "target": target_label, "n": len(sub),
                    "n_eff": float("nan"),
                    "marginal_r": float("nan"), "marginal_p": float("nan"),
                    "spearman_r": float("nan"), "spearman_p": float("nan"),
                    "partial_r": float("nan"), "partial_p": float("nan"),
                    "controls": ",".join(other_axes),
                    "weighted": bool(wcol), "weight_col": wcol or "",
                })
                continue
            x = sub[f"tm_{axis}"].values.astype(float)
            y = sub[target_col].values.astype(float)
            ctrl = (
                sub[[f"tm_{a}" for a in other_axes]].values.astype(float)
                if other_axes else np.empty((len(sub), 0), dtype=float)
            )
            if wcol:
                w = sub[wcol].values.astype(float)
                mr, mp, n_eff = _weighted_pearson(x, y, w)
                pr, pp, _ = _partial_pearson_multivariate_weighted(x, y, ctrl, w)
            else:
                mr, mp = pearsonr(x, y)
                pr, pp = _partial_pearson_multivariate(x, y, ctrl)
                n_eff = float(len(sub))
            sr, sp = spearmanr(x, y)
            rows.append({
                "axis": axis, "target": target_label, "n": len(sub),
                "n_eff": float(n_eff),
                "marginal_r": float(mr), "marginal_p": float(mp),
                "spearman_r": float(sr), "spearman_p": float(sp),
                "partial_r": float(pr), "partial_p": float(pp),
                "controls": ",".join(other_axes),
                "weighted": bool(wcol), "weight_col": wcol or "",
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Variance partitioning + stratified correlation (diagnostic helpers)
# ---------------------------------------------------------------------------


def _multiple_r2(
    X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None,
) -> float:
    """R² of an OLS / WLS regression of ``y`` on ``X`` (with intercept).

    When ``w`` is supplied, weighted R² is computed: ``ss_res`` and
    ``ss_tot`` are weighted by ``w`` and the mean is the weighted mean.
    """
    if X.size == 0 or len(y) == 0:
        return 0.0
    Z = np.column_stack([np.ones(len(y)), X])
    if w is None:
        beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
        yhat = Z @ beta
        ss_res = float(((y - yhat) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
    else:
        sw = np.sqrt(w)
        Zs = Z * sw[:, None]
        ys = y * sw
        beta, *_ = np.linalg.lstsq(Zs, ys, rcond=None)
        yhat = Z @ beta
        wsum = float(w.sum())
        ymean = float((w * y).sum() / wsum) if wsum > 0 else 0.0
        ss_res = float((w * (y - yhat) ** 2).sum())
        ss_tot = float((w * (y - ymean) ** 2).sum())
    if ss_tot <= 0:
        return 0.0
    return max(0.0, 1.0 - ss_res / ss_tot)


def _weighted_pearson(
    x: np.ndarray, y: np.ndarray, w: np.ndarray,
) -> tuple[float, float, float]:
    """Weighted Pearson r with a Fisher-z p-value using ``n_eff = (Σw)²/Σw²``."""
    from scipy.stats import norm  # type: ignore

    w = np.asarray(w, dtype=float)
    wsum = float(w.sum())
    if wsum <= 0 or len(x) < 5:
        return float("nan"), float("nan"), 0.0
    mx = float((w * x).sum() / wsum)
    my = float((w * y).sum() / wsum)
    cov_xy = float((w * (x - mx) * (y - my)).sum() / wsum)
    var_x = float((w * (x - mx) ** 2).sum() / wsum)
    var_y = float((w * (y - my) ** 2).sum() / wsum)
    if var_x <= 0 or var_y <= 0:
        return float("nan"), float("nan"), 0.0
    r = cov_xy / np.sqrt(var_x * var_y)
    r = float(np.clip(r, -1.0, 1.0))
    n_eff = float(wsum * wsum / float((w * w).sum())) if (w * w).sum() > 0 else 0.0
    if n_eff <= 3 or abs(r) >= 1:
        p = float("nan")
    else:
        z = 0.5 * np.log((1 + r) / (1 - r))
        se = 1.0 / np.sqrt(n_eff - 3)
        p = 2.0 * float(norm.sf(abs(z) / se))
    return r, p, n_eff


def _partial_pearson_multivariate_weighted(
    x: np.ndarray, y: np.ndarray,
    controls: np.ndarray, w: np.ndarray,
) -> tuple[float, float, float]:
    """Weighted Pearson r of ``x`` and ``y`` after WLS-residualising on ``controls``."""
    if controls.ndim == 1:
        controls = controls.reshape(-1, 1)
    if controls.size == 0:
        return _weighted_pearson(x, y, w)
    sw = np.sqrt(w)
    Z = np.column_stack([np.ones_like(x), controls])
    Zs = Z * sw[:, None]
    beta_x, *_ = np.linalg.lstsq(Zs, x * sw, rcond=None)
    beta_y, *_ = np.linalg.lstsq(Zs, y * sw, rcond=None)
    rx = x - Z @ beta_x
    ry = y - Z @ beta_y
    return _weighted_pearson(rx, ry, w)


# ---------------------------------------------------------------------------
# Cluster loading + per-pair weight attachment
# ---------------------------------------------------------------------------

def load_domain_cluster_map(cluster_tsv: str | Path) -> dict[str, str]:
    """Load a two-column ``rep<TAB>member`` TSV into ``module_id → rep``.

    The TSV format matches what generic clustering tools (foldseek/mmseqs2
    easy-cluster, etc.) emit; any such file works as long as both columns
    are module identifiers from this analysis.
    """
    df = pd.read_csv(cluster_tsv, sep="\t", header=None, names=["rep", "member"])
    return dict(zip(df["member"].astype(str), df["rep"].astype(str)))


def build_seq_to_cluster(
    seq_df: pd.DataFrame,
    cluster_map: dict[str, str],
    cluster_axes: str | tuple[str, ...],
) -> dict[str, str]:
    """Map seq_id → cluster representative.

    When ``cluster_axes`` is a tuple, the per-axis cluster reps are joined
    with ``|`` to produce a **composite cluster ID** — two sequences are
    assigned to the same cluster only when they share the cluster
    representative on *every* axis. This is the right redundancy control
    for multi-domain analyses (e.g., H1.1 needs (α, β, γ) joint clustering;
    α-only clustering leaves γ-paralog redundancy intact).

    For single-axis input (string) the behaviour is identical to the
    earlier single-axis version: just the cluster_rep on that axis.

    Sequences with no clustered module on a given axis get the literal
    string ``"none"`` for that axis position; sequences with no clustered
    module on *any* axis fall back to a self-cluster (``"<seq_id>__unclustered"``)
    so they contribute weight 1.
    """
    if isinstance(cluster_axes, str):
        cluster_axes = (cluster_axes,)

    out: dict[str, str] = {}
    for _, row in seq_df.iterrows():
        seq_id = row["seq_id"]
        parts: list[str] = []
        any_clustered = False
        for ax in cluster_axes:
            mid = row.get(f"module_{ax}")
            if mid and mid in cluster_map:
                parts.append(cluster_map[mid])
                any_clustered = True
            else:
                parts.append("none")
        out[seq_id] = "|".join(parts) if any_clustered else f"{seq_id}__unclustered"
    return out


_SUBSTRATE_BIN_EDGES = (0.0, 0.4, 0.6, 0.8, 0.999, 1.001)
_SUBSTRATE_BIN_LABELS = ("[0.00,0.40)", "[0.40,0.60)", "[0.60,0.80)",
                         "[0.80,1.00)", "[1.00,1.00]")


def _bin_substrate(v: float) -> str:
    if pd.isna(v):
        return "nan"
    for lo, hi, lbl in zip(_SUBSTRATE_BIN_EDGES[:-1], _SUBSTRATE_BIN_EDGES[1:],
                           _SUBSTRATE_BIN_LABELS):
        if lo <= v < hi:
            return lbl
    return _SUBSTRATE_BIN_LABELS[-1]


def _normalise_to_mean_one(w: np.ndarray) -> np.ndarray:
    """Rescale a non-negative weight vector to mean 1 (over its support)."""
    w = np.where(w > 0, w, 0.0).astype(float)
    pos = w[w > 0]
    if pos.size == 0:
        return w
    return w / pos.mean()


def attach_pair_weight_columns(
    pair_df: pd.DataFrame,
    *,
    seq_to_cluster: dict[str, str],
    seq_to_ot: dict[str, frozenset],
    cluster_size_cap: int = 5,
) -> pd.DataFrame:
    """Add cluster / OT-pair / substrate-bin columns and per-target weights.

    Weighting strategy (chosen to avoid tiny strata dominating the regression
    while still de-weighting redundancy and over-represented strata):

      w_cluster        = 1 / (min(size_a, K) · min(size_b, K))
                          → large clusters are capped at K (default 5) so a
                            316-member cluster does not vanish; redundancy is
                            still removed but no pair is reduced to ε weight.
      w_ot_pair        = 1 / sqrt(N_pairs(ot_pair_class))
                          → square-root inverse-frequency: balances classes
                            without letting a class of N=1 outweigh a class
                            of N=1000 by 1000×.
      w_substrate_bin  = 1 / sqrt(N_pairs(substrate_bin))   (same rationale)
      w_n_shared       = log(1 + n_shared_substrates)
                          → up-weights pairs sharing many substrates (richer
                            information for the product target).

    Per-target combined weights are normalised to mean 1 over their support
    so cross-target effective sample sizes are comparable:

      w_target_reaction_jaccard
        = w_cluster · w_ot_pair                              (every pair)
      w_target_substrate_mcs_sim
        = w_cluster · w_substrate_bin · 1[mcs < 1.0]
          (excludes the saturated same-substrate cluster — no within-bin
           chemistry variance there)
      w_target_product_avg_tanimoto_shared_substrate
        = w_cluster · w_n_shared · w_substrate_bin · 1[n_shared ≥ 1]
    """
    cluster_sizes: dict[str, int] = {}
    for c in seq_to_cluster.values():
        cluster_sizes[c] = cluster_sizes.get(c, 0) + 1

    pair_df = pair_df.copy()
    pair_df["cluster_a"] = pair_df["seq_a"].map(seq_to_cluster)
    pair_df["cluster_b"] = pair_df["seq_b"].map(seq_to_cluster)
    pair_df["cluster_size_a"] = pair_df["cluster_a"].map(cluster_sizes).fillna(1).astype(int)
    pair_df["cluster_size_b"] = pair_df["cluster_b"].map(cluster_sizes).fillna(1).astype(int)

    def _ot_key(seq_id: str) -> tuple[str, ...]:
        ot = seq_to_ot.get(seq_id, frozenset())
        return tuple(sorted(ot)) if ot else ("none",)

    pair_df["ot_pair_class"] = [
        str(tuple(sorted([_ot_key(a), _ot_key(b)])))
        for a, b in zip(pair_df["seq_a"], pair_df["seq_b"])
    ]

    pair_df["substrate_bin"] = (
        pair_df["substrate_mcs_sim"].apply(_bin_substrate)
        if "substrate_mcs_sim" in pair_df.columns else "nan"
    )

    # ---- component weights ----
    cap = float(cluster_size_cap)
    sa = np.minimum(pair_df["cluster_size_a"].values.astype(float), cap)
    sb = np.minimum(pair_df["cluster_size_b"].values.astype(float), cap)
    pair_df["w_cluster"] = _normalise_to_mean_one(1.0 / (sa * sb))

    ot_counts = pair_df["ot_pair_class"].value_counts().to_dict()
    pair_df["w_ot_pair"] = _normalise_to_mean_one(np.array(
        [1.0 / np.sqrt(ot_counts[c]) for c in pair_df["ot_pair_class"]]
    ))

    bin_counts = pair_df["substrate_bin"].value_counts().to_dict()
    pair_df["w_substrate_bin"] = _normalise_to_mean_one(np.array(
        [1.0 / np.sqrt(bin_counts[b]) if b != "nan" else 0.0
         for b in pair_df["substrate_bin"]]
    ))

    n_shared = pair_df.get("n_shared_substrates",
                           pd.Series(0, index=pair_df.index)).fillna(0).astype(int)
    pair_df["w_n_shared"] = np.log1p(n_shared.values.astype(float))

    # ---- per-target combined weights ----
    w_rxn = pair_df["w_cluster"].values * pair_df["w_ot_pair"].values
    pair_df["w_target_reaction_jaccard"] = _normalise_to_mean_one(w_rxn)

    if "substrate_mcs_sim" in pair_df.columns:
        not_saturated = (pair_df["substrate_mcs_sim"].fillna(0) < 0.999).astype(float).values
    else:
        not_saturated = np.ones(len(pair_df), dtype=float)
    w_sub = pair_df["w_cluster"].values * pair_df["w_substrate_bin"].values * not_saturated
    pair_df["w_target_substrate_mcs_sim"] = _normalise_to_mean_one(w_sub)

    has_shared = (n_shared.values > 0).astype(float)
    w_prod = (pair_df["w_cluster"].values * pair_df["w_n_shared"].values
              * pair_df["w_substrate_bin"].values * has_shared)
    pair_df["w_target_product_avg_tanimoto_shared_substrate"] = _normalise_to_mean_one(w_prod)

    return pair_df


def variance_partition(
    pair_df: pd.DataFrame,
    axes: tuple[str, ...],
    target_col: str,
    *,
    weights_col: str | None = None,
) -> dict:
    """Decompose variance of ``target_col`` into unique-per-axis + shared + unexplained.

    For ``n`` axes:
      * unique_<axis>  = R²(all axes) − R²(all axes except this one) ≥ 0
                          (the contribution that goes away if you drop the axis).
      * shared         = R²(all axes) − Σ unique_<axis>.
                          Can be **negative** when there's a suppressor effect
                          (i.e. the axes cancel rather than reinforce); we
                          report the raw value rather than clipping so the
                          plot can show that explicitly.
      * unexplained    = 1 − R²(all axes).

    All values sum to 1.0 modulo float noise.
    """
    needed = [target_col] + [f"tm_{ax}" for ax in axes]
    extra = [weights_col] if weights_col else []
    sub = pair_df[needed + extra].dropna()
    if weights_col:
        sub = sub[sub[weights_col] > 0]
    if len(sub) < len(axes) + 5:
        return {
            **{f"unique_{ax}": float("nan") for ax in axes},
            "shared": float("nan"),
            "unexplained": float("nan"),
            "r2_full": float("nan"),
            "n": len(sub),
            "n_eff": float("nan"),
            "weighted": bool(weights_col),
        }
    y = sub[target_col].values.astype(float)
    X_full = sub[[f"tm_{ax}" for ax in axes]].values.astype(float)
    w = sub[weights_col].values.astype(float) if weights_col else None
    r2_full = _multiple_r2(X_full, y, w)

    if w is not None:
        wsum = float(w.sum())
        n_eff = wsum * wsum / float((w * w).sum()) if (w * w).sum() > 0 else 0.0
    else:
        n_eff = float(len(sub))

    out = {"r2_full": r2_full, "n": len(sub), "n_eff": float(n_eff),
           "weighted": bool(weights_col)}
    total_unique = 0.0
    for i, ax in enumerate(axes):
        others = [j for j in range(len(axes)) if j != i]
        if others:
            r2_without = _multiple_r2(X_full[:, others], y, w)
        else:
            r2_without = 0.0
        u = max(0.0, r2_full - r2_without)
        out[f"unique_{ax}"] = u
        total_unique += u
    out["shared"] = r2_full - total_unique
    out["unexplained"] = 1.0 - r2_full
    return out


def stratified_correlations(
    pair_df: pd.DataFrame,
    axes: tuple[str, ...],
    target_col: str,
    *,
    bin_col: str = "substrate_mcs_sim",
    bins: tuple[float, ...] = (0.0, 0.4, 0.6, 0.8, 0.999, 1.001),
    bin_labels: tuple[str, ...] | None = None,
    weights_col: str | None = None,
) -> pd.DataFrame:
    """Pearson r of (axis-TM, target) within bins of ``bin_col``.

    Default bin edges split substrate-MCS into:
      [0, 0.4)   distantly related substrates (cross-substrate-class pairs)
      [0.4, 0.6) intermediate
      [0.6, 0.8) somewhat similar
      [0.8, 1.0) very similar (same broad class)
      [1.0]      identical substrate

    A flat r across bins means the marginal correlation is driven by the
    overall data-generating process equally; a strong r in only some bins
    means the signal is bin-specific (e.g. comes mostly from
    cross-substrate-class pairs and disappears within "same substrate").
    """
    from scipy.stats import pearsonr  # type: ignore

    if bin_labels is None:
        bin_labels = tuple(
            f"[{bins[i]:.2f}, {bins[i + 1]:.2f})"
            for i in range(len(bins) - 1)
        )
    rows = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (pair_df[bin_col] >= lo) & (pair_df[bin_col] < hi)
        drop_cols = [f"tm_{a}" for a in axes] + [target_col]
        if weights_col:
            drop_cols = drop_cols + [weights_col]
        sub = pair_df[mask].dropna(subset=drop_cols)
        if weights_col:
            sub = sub[sub[weights_col] > 0]
        for ax in axes:
            x = sub[f"tm_{ax}"].values.astype(float)
            y = sub[target_col].values.astype(float)
            if len(x) >= 5 and x.std() > 0 and y.std() > 0:
                if weights_col:
                    w = sub[weights_col].values.astype(float)
                    r, p, n_eff = _weighted_pearson(x, y, w)
                else:
                    r, p = pearsonr(x, y)
                    n_eff = float(len(x))
            else:
                r, p, n_eff = float("nan"), float("nan"), float(len(x))
            rows.append({
                "axis": ax,
                "bin_label": bin_labels[i],
                "bin_lo": lo,
                "bin_hi": hi,
                "n": len(sub),
                "n_eff": float(n_eff),
                "pearson_r": float(r),
                "pearson_p": float(p),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Configuration filtering
# ---------------------------------------------------------------------------

def filter_to_subset(
    seq_df: pd.DataFrame,
    *,
    require_axes: tuple[str, ...],
    require_class_reactions: int | None,
) -> pd.DataFrame:
    """Restrict to sequences with exactly one detected domain of each axis in
    ``require_axes`` and at least one reaction of ``require_class_reactions``.

    With ``require_class_reactions=None`` only the domain filter applies.
    """
    sub = seq_df.copy()
    for ax in require_axes:
        sub = sub[sub[f"n_{ax}"] == 1]
    # Disallow extra domains so we can isolate the configuration cleanly.
    extras = [a for a in DOMAIN_AXES_ALL if a not in require_axes]
    for ax in extras:
        sub = sub[sub[f"n_{ax}"] == 0]
    if require_class_reactions is not None:
        col = f"originaltypes_class{require_class_reactions}"
        sub = sub[sub[col].apply(lambda s: len(s) > 0)]
    return sub.reset_index(drop=True)
