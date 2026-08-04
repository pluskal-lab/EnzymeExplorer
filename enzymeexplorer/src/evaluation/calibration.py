"""Per-class probability calibration on pooled out-of-fold predictions.

Replaces the previous confidence-tier system. Calibration sits entirely in
the evaluation pipeline: nothing about base-classifier training changes.

Family selection
----------------
For each ``(classifier, target_class)``, four calibrator families compete:

* **identity**     — no transformation; ``p̂(s) = s``. Used when raw scores
  are already on the diagonal (RF on a well-separated class).
* **temperature**  — 1 parameter: ``p̂(s) = σ(logit(s) / T)``. Smooth
  global rescaling; can't drift the way beta's intercept can.
* **logit_platt**  — 2 parameters: ``p̂(s) = σ(a·logit(s) + b)``. Adds a
  bias on top of temperature.
* **beta**         — 3 parameters: ``p̂(s) = σ(a·log(s) − b·log(1−s) + c)``.
  Asymmetric tail flexibility (Kull, Filho, Flach 2017).

Selection rule: pick the simplest family whose LOFO log-loss is within a
small tolerance of the best (Occam: simpler wins ties). Tolerance defaults
to 1e-3 in log-loss units. The LOFO log-loss IS the cross-validated
selection criterion — no extra hold-out needed.

Honest evaluation
-----------------
LOFO predictions are held out by construction: for each fold ``k``, refit
on the remaining 4 folds and predict on fold ``k``. Concatenated LOFO
probabilities feed the reliability diagram and the calibration metrics
that the family selection is *also* computed from.

Cluster bootstrap
-----------------
Resample folds with replacement (size = n_folds), refit the chosen
family, evaluate ``p̂(s)`` on a score grid. Pointwise 2.5/97.5 percentiles
form the CI ribbon. Captures between-fold (model) variance.

Per-fold drift
--------------
The selected family is refit on each LOFO holdout's training set; the
parameter spreads ``(max - min) / (1 + |mean|)`` across folds are
reported. ``drift_flagged`` fires when at least ``ceil(n_params/2)`` of
them exceed the threshold (so ``≥2 of 3`` for beta).

Sample-size grading (status field)
----------------------------------
Independent of family choice:

* ``n_pos < min_n_pos`` → ``skipped_low_n_pos``
* ``n_pos < 50``        → ``fit_borderline``
* ``n_pos < 100``       → ``fit_caveat``
* otherwise             → ``fit``
* fit failed for any reason → ``fit_failed``
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from scipy.special import expit  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore

from enzymeexplorer.src.evaluation.io import FoldDfs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants and status grades
# ---------------------------------------------------------------------------

DEFAULT_MIN_N_POS: int = 30
DEFAULT_SCORE_EPS: float = 1.0e-6
DEFAULT_N_BOOTSTRAP: int = 1000
DEFAULT_BOOTSTRAP_SEED: int = 0
DEFAULT_N_RELIABILITY_BINS: int = 10
DEFAULT_TOP_K_FP: int = 25
DEFAULT_BOTTOM_K_FN: int = 25
DEFAULT_CI: float = 0.95
DEFAULT_FOLD_DRIFT_THRESHOLD: float = 0.5
DEFAULT_FAMILY_TOLERANCE: float = 1.0e-3
# Minimum samples per reliability bin when capping n_bins for small classes.
# n_bins_eff = min(n_reliability_bins, max(MIN_BINS, n_pos // MIN_PER_BIN)).
MIN_SAMPLES_PER_BIN: int = 15
MIN_RELIABILITY_BINS: int = 2

# Family identifiers.
FAMILY_IDENTITY = "identity"
FAMILY_TEMPERATURE = "temperature"
FAMILY_LOGIT_PLATT = "logit_platt"
FAMILY_BETA = "beta"
DEFAULT_FAMILIES: tuple[str, ...] = (
    FAMILY_IDENTITY, FAMILY_TEMPERATURE, FAMILY_LOGIT_PLATT, FAMILY_BETA,
)
# Selection prefers earlier entries on ties (simpler-first).
_FAMILY_ORDER: tuple[str, ...] = (
    FAMILY_IDENTITY, FAMILY_TEMPERATURE, FAMILY_LOGIT_PLATT, FAMILY_BETA,
)

CAVEAT_NPOS_THRESHOLD: int = 100
BORDERLINE_NPOS_THRESHOLD: int = 50

STATUS_FIT = "fit"
STATUS_CAVEAT = "fit_caveat"
STATUS_BORDERLINE = "fit_borderline"
STATUS_SKIPPED = "skipped_low_n_pos"
STATUS_FAILED = "fit_failed"

Z_TWO_SIDED_95 = 1.96


def _grade_n_pos(n_pos: int, min_n_pos: int) -> tuple[str, str | None]:
    """Map ``n_pos`` to a status grade + a one-line reason."""
    if n_pos < min_n_pos:
        return STATUS_SKIPPED, f"n_pos={n_pos} < min_n_pos={min_n_pos}"
    if n_pos < BORDERLINE_NPOS_THRESHOLD:
        return STATUS_BORDERLINE, (
            f"n_pos={n_pos} in [{min_n_pos}, {BORDERLINE_NPOS_THRESHOLD}); "
            "calibration borderline — recommend deployment via upper p̂ only"
        )
    if n_pos < CAVEAT_NPOS_THRESHOLD:
        return STATUS_CAVEAT, (
            f"n_pos={n_pos} in [{BORDERLINE_NPOS_THRESHOLD}, "
            f"{CAVEAT_NPOS_THRESHOLD}); calibration fitted with caveat"
        )
    return STATUS_FIT, None


# ---------------------------------------------------------------------------
# Wilson interval primitive (used by reliability bins)
# ---------------------------------------------------------------------------

def wilson_interval(
    successes: int, trials: int, *, z: float = Z_TWO_SIDED_95
) -> tuple[float, float]:
    """Two-sided Wilson interval clipped to [0, 1]."""
    if trials <= 0:
        return (float("nan"), float("nan"))
    phat = successes / trials
    denom = 1.0 + z * z / trials
    centre = (phat + z * z / (2 * trials)) / denom
    margin = (z / denom) * math.sqrt(
        phat * (1.0 - phat) / trials + z * z / (4 * trials * trials)
    )
    return (max(0.0, centre - margin), min(1.0, centre + margin))


# ---------------------------------------------------------------------------
# OOF frame
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OofFrame:
    """Pooled out-of-fold predictions for one (classifier, target_class)."""
    df: pd.DataFrame  # columns: id, fold, label (int8), score (float64 in [0,1])
    classifier: str
    target_class: str

    @property
    def n_pos(self) -> int:
        return int(self.df["label"].sum())

    @property
    def n_neg(self) -> int:
        return int((self.df["label"] == 0).sum())

    @property
    def n_total(self) -> int:
        return len(self.df)


def build_oof_frame(
    fold_dfs: Mapping[int, FoldDfs],
    target_class: str,
    classifier: str,
) -> OofFrame:
    """Pool per-fold ``(labels_df, preds_df)`` pairs into one long frame."""
    pieces: list[pd.DataFrame] = []
    for fold, (lab, prd) in fold_dfs.items():
        if target_class not in lab.columns or target_class not in prd.columns:
            continue
        pieces.append(pd.DataFrame({
            "id": lab["ID"].to_numpy(),
            "fold": fold,
            "label": lab[target_class].to_numpy().astype(np.int8),
            "score": np.clip(
                prd[target_class].to_numpy(), 0.0, 1.0
            ).astype(np.float64),
        }))
    if not pieces:
        return OofFrame(
            pd.DataFrame(columns=["id", "fold", "label", "score"]),
            classifier, target_class,
        )
    return OofFrame(pd.concat(pieces, ignore_index=True), classifier, target_class)


# ---------------------------------------------------------------------------
# Calibrator family classes (common interface)
# ---------------------------------------------------------------------------

def _logit(s: np.ndarray, eps: float) -> np.ndarray:
    s_clipped = np.clip(s.astype(np.float64), eps, 1.0 - eps)
    return np.log(s_clipped / (1.0 - s_clipped))


def _beta_features(s: np.ndarray, eps: float) -> np.ndarray:
    s_clipped = np.clip(s.astype(np.float64), eps, 1.0 - eps)
    return np.column_stack([np.log(s_clipped), -np.log(1.0 - s_clipped)])


@dataclass(frozen=True)
class IdentityCalibrator:
    """No-op: ``p̂(s) = s``. Used when raw scores are already on-diagonal."""
    eps: float = DEFAULT_SCORE_EPS
    family: str = FAMILY_IDENTITY
    n_params: int = 0

    @classmethod
    def fit(
        cls,
        scores: np.ndarray,
        labels: np.ndarray,
        *,
        eps: float = DEFAULT_SCORE_EPS,
    ) -> "IdentityCalibrator":
        return cls(eps=eps)

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(scores).astype(np.float64), 0.0, 1.0)

    def params_dict(self) -> dict[str, float]:
        return {"a": float("nan"), "b": float("nan"),
                "c": float("nan"), "T": float("nan")}


@dataclass(frozen=True)
class TemperatureCalibrator:
    """1 parameter: ``p̂(s) = σ(logit(s) / T)``. T=1 is identity."""
    T: float
    eps: float
    family: str = FAMILY_TEMPERATURE
    n_params: int = 1

    @classmethod
    def fit(
        cls,
        scores: np.ndarray,
        labels: np.ndarray,
        *,
        eps: float = DEFAULT_SCORE_EPS,
    ) -> "TemperatureCalibrator":
        s = np.asarray(scores)
        y = np.asarray(labels).astype(np.int8)
        if s.size == 0 or len(np.unique(y)) < 2:
            raise ValueError(
                "Temperature calibrator requires both label classes."
            )
        z = _logit(s, eps).reshape(-1, 1)
        # fit_intercept=False so we fit just a slope α; T = 1/α.
        model = LogisticRegression(
            C=1.0e6, fit_intercept=False, solver="lbfgs", max_iter=1000,
        )
        model.fit(z, y)
        alpha = float(model.coef_[0, 0])
        if not (alpha > 0):
            raise ValueError(f"temperature slope must be positive (got {alpha})")
        return cls(T=1.0 / alpha, eps=eps)

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        z = _logit(np.asarray(scores), self.eps) / self.T
        return expit(z)

    def params_dict(self) -> dict[str, float]:
        return {"a": float("nan"), "b": float("nan"),
                "c": float("nan"), "T": self.T}


@dataclass(frozen=True)
class LogitPlattCalibrator:
    """2 parameters: ``p̂(s) = σ(a·logit(s) + b)``. Platt on logit-input."""
    a: float
    b: float
    eps: float
    family: str = FAMILY_LOGIT_PLATT
    n_params: int = 2

    @classmethod
    def fit(
        cls,
        scores: np.ndarray,
        labels: np.ndarray,
        *,
        eps: float = DEFAULT_SCORE_EPS,
    ) -> "LogitPlattCalibrator":
        s = np.asarray(scores)
        y = np.asarray(labels).astype(np.int8)
        if s.size == 0 or len(np.unique(y)) < 2:
            raise ValueError(
                "Logit-Platt calibrator requires both label classes."
            )
        z = _logit(s, eps).reshape(-1, 1)
        model = LogisticRegression(C=1.0e6, solver="lbfgs", max_iter=1000)
        model.fit(z, y)
        return cls(
            a=float(model.coef_[0, 0]),
            b=float(model.intercept_[0]),
            eps=eps,
        )

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        z = self.a * _logit(np.asarray(scores), self.eps) + self.b
        return expit(z)

    def params_dict(self) -> dict[str, float]:
        return {"a": self.a, "b": self.b,
                "c": float("nan"), "T": float("nan")}


@dataclass(frozen=True)
class BetaCalibrator:
    """3 parameters: ``p̂(s) = σ(a·log(s) − b·log(1−s) + c)``."""
    a: float
    b: float
    c: float
    eps: float
    family: str = FAMILY_BETA
    n_params: int = 3

    @classmethod
    def fit(
        cls,
        scores: np.ndarray,
        labels: np.ndarray,
        *,
        eps: float = DEFAULT_SCORE_EPS,
    ) -> "BetaCalibrator":
        s = np.asarray(scores)
        y = np.asarray(labels).astype(np.int8)
        if s.size == 0 or len(np.unique(y)) < 2:
            raise ValueError(
                "Beta calibrator requires both label classes."
            )
        X = _beta_features(s, eps)
        model = LogisticRegression(C=1.0e6, solver="lbfgs", max_iter=1000)
        model.fit(X, y)
        return cls(
            a=float(model.coef_[0, 0]),
            b=float(model.coef_[0, 1]),
            c=float(model.intercept_[0]),
            eps=eps,
        )

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        X = _beta_features(np.asarray(scores), self.eps)
        z = self.a * X[:, 0] + self.b * X[:, 1] + self.c
        return expit(z)

    def params_dict(self) -> dict[str, float]:
        return {"a": self.a, "b": self.b, "c": self.c, "T": float("nan")}


_FAMILY_TO_CLASS: dict[str, type] = {
    FAMILY_IDENTITY: IdentityCalibrator,
    FAMILY_TEMPERATURE: TemperatureCalibrator,
    FAMILY_LOGIT_PLATT: LogitPlattCalibrator,
    FAMILY_BETA: BetaCalibrator,
}


def calibrator_class_for(family: str) -> type:
    if family not in _FAMILY_TO_CLASS:
        raise ValueError(
            f"Unknown calibrator family {family!r}; "
            f"valid: {sorted(_FAMILY_TO_CLASS)}"
        )
    return _FAMILY_TO_CLASS[family]


def build_calibrator_from_params(
    family: str,
    params: Mapping[str, float],
    *,
    eps: float,
):
    """Reconstruct a calibrator object from a stored params dict."""
    if family == FAMILY_IDENTITY:
        return IdentityCalibrator(eps=eps)
    if family == FAMILY_TEMPERATURE:
        return TemperatureCalibrator(T=float(params["T"]), eps=eps)
    if family == FAMILY_LOGIT_PLATT:
        return LogitPlattCalibrator(
            a=float(params["a"]), b=float(params["b"]), eps=eps,
        )
    if family == FAMILY_BETA:
        return BetaCalibrator(
            a=float(params["a"]), b=float(params["b"]),
            c=float(params["c"]), eps=eps,
        )
    raise ValueError(f"Unknown calibrator family {family!r}")


# ---------------------------------------------------------------------------
# CalibrationFit
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CalibrationFit:
    """Deployment-side record for one (classifier, target_class)."""
    classifier: str
    target_class: str
    n_pos: int
    n_neg: int
    family: str | None
    a: float | None
    b: float | None
    c: float | None
    T: float | None
    eps: float
    status: str
    skip_reason: str | None
    n_reliability_bins_used: int
    identity_vetoed: bool
    identity_veto_reason: str | None
    selection_diagnostics: list[dict]
    """Per-family selection diagnostics; one entry per candidate family with
    keys: family, lofo_log_loss, lofo_ece, eligible, ineligible_reason,
    selected."""

    @property
    def calibrator(self):
        if self.family is None:
            return None
        params = {"a": self.a, "b": self.b, "c": self.c, "T": self.T}
        return build_calibrator_from_params(self.family, params, eps=self.eps)

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d.pop("selection_diagnostics", None)
        return d


# ---------------------------------------------------------------------------
# Helpers shared with downstream code (LOFO refits, drift, bootstrap)
# ---------------------------------------------------------------------------

def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    eps = 1.0e-12
    pp = np.clip(p, eps, 1.0 - eps)
    return float(-(y * np.log(pp) + (1.0 - y) * np.log(1.0 - pp)).mean())


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(((p - y) ** 2).mean())


def _refit_for_family(
    family: str,
    df_train: pd.DataFrame,
    *,
    eps: float,
):
    """Fit a calibrator of ``family`` on ``df_train`` (or None if impossible)."""
    if df_train.empty:
        return None
    if family != FAMILY_IDENTITY and df_train["label"].nunique() < 2:
        return None
    klass = calibrator_class_for(family)
    try:
        return klass.fit(
            df_train["score"].to_numpy(),
            df_train["label"].to_numpy(),
            eps=eps,
        )
    except ValueError as exc:
        logger.debug("Family %s fit raised: %s", family, exc)
        return None


def effective_n_reliability_bins(
    n_pos: int,
    requested: int,
    *,
    min_per_bin: int = MIN_SAMPLES_PER_BIN,
    floor: int = MIN_RELIABILITY_BINS,
) -> int:
    """Cap reliability bins so each carries at least ``min_per_bin`` LOFO
    samples on average. ``n_pos`` here is the count of positives; the LOFO
    sample size is ``n_pos + n_neg`` but the *positives-per-bin* is what
    drives Wilson interval width on observed positive rate. We use
    ``n_pos // min_per_bin`` as the cap and clamp to ``[floor, requested]``.
    """
    if requested <= floor:
        return floor
    cap = max(floor, n_pos // min_per_bin)
    return int(min(requested, cap))


def _reliability_has_significant_bin(
    y: np.ndarray, p: np.ndarray, *, n_bins: int, z: float = Z_TWO_SIDED_95,
) -> tuple[bool, int]:
    """Return ``(has_sig, n_sig_bins)`` for a LOFO reliability check.

    A bin is "significant" when its mean predicted probability falls
    outside that bin's binomial Wilson CI on the observed positive rate.
    """
    rel = reliability_table(y, p, n_bins=n_bins, z=z)
    if rel.empty:
        return False, 0
    sig = ((rel["p_pred_mean"] < rel["wilson_lo"])
           | (rel["p_pred_mean"] > rel["wilson_hi"]))
    return bool(sig.any()), int(sig.sum())


def _lofo_predict_for_family(
    oof: OofFrame,
    family: str,
    *,
    eps: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Concatenate per-fold held-out predictions for one calibrator family.

    Returns ``(p_lofo, y)`` aligned arrays or ``None`` if any fold's fit
    failed and the family cannot be evaluated.
    """
    folds = sorted(int(f) for f in oof.df["fold"].unique())
    pieces_p: list[np.ndarray] = []
    pieces_y: list[np.ndarray] = []
    for held in folds:
        train = oof.df[oof.df["fold"] != held]
        test = oof.df[oof.df["fold"] == held]
        if test.empty:
            continue
        cal = _refit_for_family(family, train, eps=eps)
        if cal is None:
            return None
        pieces_p.append(cal.predict_proba(test["score"].to_numpy()))
        pieces_y.append(test["label"].to_numpy())
    if not pieces_p:
        return None
    return np.concatenate(pieces_p), np.concatenate(pieces_y)


# ---------------------------------------------------------------------------
# Family selection
# ---------------------------------------------------------------------------

def fit_best_calibrator(
    oof: OofFrame,
    *,
    families: Sequence[str] = DEFAULT_FAMILIES,
    min_n_pos: int = DEFAULT_MIN_N_POS,
    eps: float = DEFAULT_SCORE_EPS,
    family_tolerance: float = DEFAULT_FAMILY_TOLERANCE,
    n_reliability_bins: int = DEFAULT_N_RELIABILITY_BINS,
) -> CalibrationFit:
    """Pick the simplest family within ``family_tolerance`` of the best
    LOFO log-loss, refit on the full pooled OOF, and return a
    ``CalibrationFit``.

    Selection rules
    ---------------
    Two eligibility filters are applied before picking the simplest family
    within tolerance of the best LOFO log-loss:

    * **Identity veto (rule 1)**: If the LOFO reliability table for raw
      scores has at least one bin whose mean predicted probability falls
      outside its own binomial Wilson 95% CI on the observed positive
      rate, identity is disqualified — something must transform the
      scores.
    * **ECE-must-improve (rule 2)**: A non-identity family is eligible
      only if its LOFO ECE does not worsen versus identity.

    The number of reliability bins used for both rule 1 and rule 2 is
    capped via :func:`effective_n_reliability_bins` so each bin carries
    at least ``MIN_SAMPLES_PER_BIN`` positives on average, avoiding
    tail-bin noise from inflating either check on small classes.
    """
    n_pos, n_neg = oof.n_pos, oof.n_neg
    n_bins_eff = effective_n_reliability_bins(n_pos, n_reliability_bins)
    status, reason = _grade_n_pos(n_pos, min_n_pos)

    def _empty_fit(status_: str, reason_: str | None) -> CalibrationFit:
        return CalibrationFit(
            classifier=oof.classifier, target_class=oof.target_class,
            n_pos=n_pos, n_neg=n_neg,
            family=None, a=None, b=None, c=None, T=None, eps=eps,
            status=status_, skip_reason=reason_,
            n_reliability_bins_used=n_bins_eff,
            identity_vetoed=False, identity_veto_reason=None,
            selection_diagnostics=[],
        )

    if status == STATUS_SKIPPED or n_neg == 0:
        return _empty_fit(
            STATUS_SKIPPED if status == STATUS_SKIPPED else STATUS_FAILED,
            reason or "no negatives in OOF",
        )

    # 1) LOFO predictions per candidate family.
    lofo_per_family: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    log_loss_by: dict[str, float] = {}
    ece_by: dict[str, float] = {}
    for f in families:
        if f not in _FAMILY_TO_CLASS:
            logger.warning("Unknown calibrator family %r — skipping.", f)
            continue
        lofo = _lofo_predict_for_family(oof, f, eps=eps)
        if lofo is None:
            log_loss_by[f] = float("inf")
            ece_by[f] = float("inf")
            continue
        p, y = lofo
        lofo_per_family[f] = (p, y)
        log_loss_by[f] = _log_loss(y, p)
        ece, _ = _ece_mce(y, p, n_bins_eff)
        ece_by[f] = ece if not math.isnan(ece) else float("inf")

    if not lofo_per_family:
        return _empty_fit(
            STATUS_FAILED,
            "no calibrator family produced a valid LOFO fit",
        )

    # 2) Rule 1: identity veto.
    identity_vetoed = False
    identity_veto_reason: str | None = None
    if FAMILY_IDENTITY in lofo_per_family:
        p_id, y_id = lofo_per_family[FAMILY_IDENTITY]
        sig, n_sig = _reliability_has_significant_bin(
            y_id, p_id, n_bins=n_bins_eff,
        )
        if sig:
            identity_vetoed = True
            identity_veto_reason = (
                f"raw-score reliability has {n_sig} bin(s) outside Wilson "
                f"95% CI (n_bins_eff={n_bins_eff})"
            )

    ece_identity = ece_by.get(FAMILY_IDENTITY, float("inf"))

    # 3) Eligibility per rules 1 + 2.
    eligibility: dict[str, tuple[bool, str | None]] = {}
    for f in lofo_per_family:
        if f == FAMILY_IDENTITY:
            if identity_vetoed:
                eligibility[f] = (False, identity_veto_reason)
            else:
                eligibility[f] = (True, None)
        else:
            ece_f = ece_by[f]
            if math.isfinite(ece_identity) and ece_f > ece_identity:
                eligibility[f] = (False, (
                    f"LOFO ECE {ece_f:.4f} > identity ECE "
                    f"{ece_identity:.4f}"
                ))
            else:
                eligibility[f] = (True, None)

    # 4) Pick simplest eligible family within tolerance of best eligible LL.
    eligible_families = {f for f, (ok, _) in eligibility.items() if ok}
    if eligible_families:
        best_ll = min(log_loss_by[f] for f in eligible_families)
        winner: str | None = None
        for f in _FAMILY_ORDER:
            if f not in eligible_families:
                continue
            if log_loss_by[f] - best_ll <= family_tolerance:
                winner = f
                break
        if winner is None:
            winner = min(eligible_families, key=lambda k: log_loss_by[k])
    else:
        # Fallback: nothing was eligible (e.g. identity vetoed AND every
        # other family worsens ECE). Pick whichever has lowest log-loss
        # so we still emit a calibrator, but flag this in skip_reason.
        winner = min(log_loss_by, key=lambda k: log_loss_by[k])

    # Build per-family selection diagnostics.
    diagnostics: list[dict] = []
    for f in _FAMILY_ORDER:
        if f not in log_loss_by:
            continue
        ok, why = eligibility.get(f, (False, "not evaluated"))
        diagnostics.append({
            "family": f,
            "lofo_log_loss": log_loss_by[f],
            "lofo_ece": ece_by[f],
            "eligible": ok,
            "ineligible_reason": why,
            "selected": f == winner,
        })

    # 5) Refit winner on the full pooled OOF.
    deploy = _refit_for_family(winner, oof.df, eps=eps)
    if deploy is None:
        return CalibrationFit(
            classifier=oof.classifier, target_class=oof.target_class,
            n_pos=n_pos, n_neg=n_neg,
            family=None, a=None, b=None, c=None, T=None, eps=eps,
            status=STATUS_FAILED,
            skip_reason=f"deployment refit failed for winner {winner!r}",
            n_reliability_bins_used=n_bins_eff,
            identity_vetoed=identity_vetoed,
            identity_veto_reason=identity_veto_reason,
            selection_diagnostics=diagnostics,
        )

    params = deploy.params_dict()
    final_reason = reason
    if not eligible_families:
        prefix = "no family met both eligibility rules"
        final_reason = (
            f"{prefix}; fell back to lowest-log-loss winner"
            if reason is None else f"{reason}; {prefix}"
        )
    return CalibrationFit(
        classifier=oof.classifier, target_class=oof.target_class,
        n_pos=n_pos, n_neg=n_neg,
        family=winner,
        a=params.get("a"), b=params.get("b"),
        c=params.get("c"), T=params.get("T"),
        eps=eps,
        status=status,
        skip_reason=final_reason,
        n_reliability_bins_used=n_bins_eff,
        identity_vetoed=identity_vetoed,
        identity_veto_reason=identity_veto_reason,
        selection_diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Leave-one-fold-out evaluation (using winning family)
# ---------------------------------------------------------------------------

def per_fold_lofo_metrics(
    lofo_df: pd.DataFrame,
    *,
    n_bins: int,
) -> pd.DataFrame:
    """Per-fold log_loss / Brier / ECE / MCE on held-out predictions.

    Consumes the long ``lofo_df`` emitted by :func:`evaluate_lofo`
    (columns include ``fold``, ``p_lofo``, ``label``) and returns one
    row per fold plus a summary row (fold = ``"__mean_sd__"``) with
    mean ± SD across folds so paper tables can be produced without
    extra pandas gymnastics.
    """
    if lofo_df.empty:
        return pd.DataFrame(columns=[
            "fold", "n", "n_pos", "log_loss", "brier", "ece", "mce",
        ])
    rows: list[dict] = []
    for fold, sub in lofo_df.groupby("fold"):
        y = sub["label"].to_numpy().astype(np.int8)
        p = sub["p_lofo"].to_numpy().astype(np.float64)
        e, m = _ece_mce(y, p, n_bins)
        rows.append({
            "fold": int(fold),
            "n": int(len(y)),
            "n_pos": int(y.sum()),
            "log_loss": _log_loss(y, p),
            "brier": _brier(y, p),
            "ece": e,
            "mce": m,
        })
    per_fold = pd.DataFrame(rows)
    # Append mean/SD row for legibility. Skip NaN so degenerate folds
    # (e.g. no positives) don't poison the summary.
    if not per_fold.empty:
        agg: dict = {"fold": "__mean_sd__",
                     "n": int(per_fold["n"].sum()),
                     "n_pos": int(per_fold["n_pos"].sum())}
        for col in ("log_loss", "brier", "ece", "mce"):
            v = per_fold[col].to_numpy(dtype=float)
            mean = float(np.nanmean(v))
            sd = float(np.nanstd(v, ddof=1)) if np.isfinite(v).sum() > 1 else float("nan")
            agg[col] = f"{mean:.6f}"
            agg[f"{col}_mean"] = mean
            agg[f"{col}_sd"] = sd
        per_fold = pd.concat(
            [per_fold, pd.DataFrame([agg])], ignore_index=True,
        )
    return per_fold


def per_fold_reliability(
    lofo_df: pd.DataFrame,
    *,
    n_bins: int,
) -> pd.DataFrame:
    """Per-fold reliability table — one reliability table per held-out fold.

    Used by the curve-overlap plot: overlaying five per-fold reliability
    curves lets a reviewer eyeball whether the pooled LOFO reliability
    (which the paper reports) is representative or driven by one fold."""
    if lofo_df.empty:
        return pd.DataFrame()
    pieces: list[pd.DataFrame] = []
    for fold, sub in lofo_df.groupby("fold"):
        rel = reliability_table(
            sub["label"].to_numpy(),
            sub["p_lofo"].to_numpy(),
            n_bins=n_bins,
        )
        if rel.empty:
            continue
        rel.insert(0, "fold", int(fold))
        pieces.append(rel)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def evaluate_lofo(
    oof: OofFrame,
    family: str,
    *,
    eps: float = DEFAULT_SCORE_EPS,
) -> pd.DataFrame:
    """Held-out calibrated probabilities for every id, using ``family``.

    Long-form ``[id, fold, raw_score, p_lofo, label]``.
    """
    out_pieces: list[pd.DataFrame] = []
    folds = sorted(int(f) for f in oof.df["fold"].unique())
    for held in folds:
        train = oof.df[oof.df["fold"] != held]
        test = oof.df[oof.df["fold"] == held]
        if test.empty:
            continue
        cal = _refit_for_family(family, train, eps=eps)
        if cal is None:
            continue
        out_pieces.append(pd.DataFrame({
            "id": test["id"].to_numpy(),
            "fold": held,
            "raw_score": test["score"].to_numpy(),
            "p_lofo": cal.predict_proba(test["score"].to_numpy()),
            "label": test["label"].to_numpy(),
        }))
    if not out_pieces:
        return pd.DataFrame(columns=["id", "fold", "raw_score", "p_lofo", "label"])
    return pd.concat(out_pieces, ignore_index=True)


# ---------------------------------------------------------------------------
# Per-fold parameter drift (family-aware)
# ---------------------------------------------------------------------------

def per_fold_param_drift(
    oof: OofFrame,
    family: str,
    *,
    eps: float = DEFAULT_SCORE_EPS,
) -> pd.DataFrame:
    """Calibrator parameters refit on each LOFO holdout's training set.

    Columns include only the params relevant to ``family``; others are NaN.
    Returns empty DataFrame for the identity family (no params).
    """
    folds = sorted(int(f) for f in oof.df["fold"].unique())
    rows: list[dict] = []
    for held in folds:
        train = oof.df[oof.df["fold"] != held]
        cal = _refit_for_family(family, train, eps=eps)
        if cal is None:
            continue
        params = cal.params_dict()
        rows.append({
            "held_out_fold": held,
            "family": family,
            "a": params["a"], "b": params["b"],
            "c": params["c"], "T": params["T"],
            "n_train": int(len(train)),
            "n_pos_train": int(train["label"].sum()),
        })
    return pd.DataFrame(rows)


def detect_fold_drift(
    per_fold_params: pd.DataFrame,
    *,
    family: str,
    threshold: float = DEFAULT_FOLD_DRIFT_THRESHOLD,
) -> tuple[bool, dict[str, float]]:
    """Family-aware drift detector.

    Returns ``(flagged, spreads_by_param)``. The flag fires when at least
    ``ceil(n_params/2)`` parameters' normalised spread exceeds ``threshold``.
    Identity (0 params) is never flagged.
    """
    if per_fold_params.empty:
        return False, {}
    klass = _FAMILY_TO_CLASS.get(family)
    if klass is None or klass.n_params == 0:
        return False, {}
    relevant = {
        FAMILY_TEMPERATURE: ("T",),
        FAMILY_LOGIT_PLATT: ("a", "b"),
        FAMILY_BETA: ("a", "b", "c"),
    }[family]
    spreads: dict[str, float] = {}
    for col in relevant:
        if col not in per_fold_params.columns:
            continue
        vals = per_fold_params[col].dropna().to_numpy()
        if len(vals) < 2:
            continue
        denom = 1.0 + float(abs(vals.mean()))
        spreads[col] = float((vals.max() - vals.min()) / denom)
    if not spreads:
        return False, {}
    n_required = math.ceil(klass.n_params / 2)
    n_exceeding = sum(1 for v in spreads.values() if v > threshold)
    flagged = n_exceeding >= n_required
    return flagged, spreads


# ---------------------------------------------------------------------------
# Cluster bootstrap CI ribbon
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BootstrapRibbon:
    classifier: str
    target_class: str
    family: str
    score_grid: np.ndarray
    p_lo: np.ndarray
    p_hat: np.ndarray
    p_hi: np.ndarray
    n_resamples_used: int
    ci: float

    def to_long_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "classifier": self.classifier,
            "target_class": self.target_class,
            "family": self.family,
            "score": self.score_grid,
            "p_lo": self.p_lo,
            "p_hat": self.p_hat,
            "p_hi": self.p_hi,
        })


def _resample_indices(
    unit: str,
    oof_df: pd.DataFrame,
    rng: np.random.Generator,
    cluster_map: dict[str, str] | None,
) -> np.ndarray:
    """Draw one bootstrap row-index vector according to ``unit``.

    ``fold``     — sample n_folds folds with replacement, concat rows.
    ``cluster``  — sample n_clusters cluster labels with replacement (from
                   the mmseqs 50%-seq-id mapping), concat their rows.
    ``row``      — sample len(oof_df) rows with replacement. Wrong for
                   this problem (homology inflates N) — kept for parity.
    """
    if unit == "fold":
        folds = np.array(sorted(int(f) for f in oof_df["fold"].unique()))
        sampled = rng.choice(folds, size=len(folds), replace=True)
        parts = [
            np.flatnonzero(oof_df["fold"].to_numpy() == int(f)) for f in sampled
        ]
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
    if unit == "cluster":
        if cluster_map is None:
            raise ValueError("cluster bootstrap unit requires cluster_map")
        ids = oof_df["id"].astype(str).to_numpy()
        labels = np.array(
            [cluster_map.get(u, "__missing_cluster__") for u in ids],
            dtype=object,
        )
        order = np.argsort(labels, kind="stable")
        sorted_labels = labels[order]
        change = np.concatenate(([True], sorted_labels[1:] != sorted_labels[:-1]))
        starts = np.flatnonzero(change)
        ends = np.append(starts[1:], len(labels))
        groups = [order[s:e] for s, e in zip(starts, ends)]
        pick = rng.integers(0, len(groups), size=len(groups))
        return np.concatenate([groups[k] for k in pick])
    if unit == "row":
        n = len(oof_df)
        return rng.integers(0, n, size=n)
    raise ValueError(f"Unknown bootstrap unit for calibrator: {unit!r}")


def cluster_bootstrap_calibrator(
    oof: OofFrame,
    deployment,
    *,
    family: str,
    n_iter: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    eps: float = DEFAULT_SCORE_EPS,
    score_grid: np.ndarray | None = None,
    ci: float = DEFAULT_CI,
    bootstrap_unit: str = "cluster",
    cluster_map: dict[str, str] | None = None,
) -> BootstrapRibbon:
    """Bootstrap the chosen calibrator family and report its curve CI.

    ``bootstrap_unit``:

    * ``"cluster"`` (default) — resample 50%-seq-identity cluster reps
      with replacement (from ``cluster_map``), refit the family. Consistent
      with the eval-pipeline block bootstrap and much finer-grained than
      the legacy 5-fold ribbon.
    * ``"fold"`` — legacy behavior: resample folds with replacement.
      Retained for reproducibility of pre-refactor ribbons.
    * ``"row"`` — sample rows with replacement. Wrong under high
      homology (inflates effective N) — parity option only.

    ``p_hat`` is the deployment fit applied to ``score_grid`` — the curve
    that will be served in production. The ribbon (``p_lo``, ``p_hi``)
    marks where a resample-refit could have landed.
    """
    rng = np.random.default_rng(seed)
    if oof.df.empty:
        nan = np.array([], dtype=np.float64)
        return BootstrapRibbon(
            oof.classifier, oof.target_class, family,
            np.array([]), nan, nan, nan, 0, ci,
        )
    if score_grid is None:
        score_grid = np.unique(np.clip(
            oof.df["score"].to_numpy(), eps, 1.0 - eps
        ))
    grid = np.asarray(score_grid)

    accum: list[np.ndarray] = []
    for _ in range(n_iter):
        idx = _resample_indices(bootstrap_unit, oof.df, rng, cluster_map)
        df_k = oof.df.iloc[idx].reset_index(drop=True)
        cal = _refit_for_family(family, df_k, eps=eps)
        if cal is None:
            continue
        accum.append(cal.predict_proba(grid))

    p_hat = deployment.predict_proba(grid)
    if not accum:
        nan = np.full_like(grid, np.nan, dtype=np.float64)
        return BootstrapRibbon(
            oof.classifier, oof.target_class, family,
            grid, nan, p_hat, nan, 0, ci,
        )
    arr = np.vstack(accum)
    alpha = (1.0 - ci) / 2.0
    p_lo = np.quantile(arr, alpha, axis=0)
    p_hi = np.quantile(arr, 1.0 - alpha, axis=0)
    return BootstrapRibbon(
        oof.classifier, oof.target_class, family,
        grid, p_lo, p_hat, p_hi, len(accum), ci,
    )


def _bootstrap_lofo_metrics(
    oof: OofFrame,
    family: str,
    *,
    n_iter: int,
    seed: int,
    eps: float,
    bootstrap_unit: str,
    cluster_map: dict[str, str] | None,
    n_bins: int,
) -> pd.DataFrame:
    """Cluster-block bootstrap CIs on the LOFO calibration metrics.

    Each draw: resample rows per ``bootstrap_unit`` (default cluster),
    then RE-COMPUTE LOFO on the resampled fold structure — that is,
    run the leave-one-fold-out refit inside the draw. Returns one row
    per (metric, draw) so callers can compute quantiles.

    The resample assigns to each drawn row its ORIGINAL fold label, so
    the 5-fold structure is preserved and LOFO stays a real hold-out
    even after resampling.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for b in range(n_iter):
        idx = _resample_indices(bootstrap_unit, oof.df, rng, cluster_map)
        df_k = oof.df.iloc[idx].reset_index(drop=True)
        if df_k["label"].nunique() < 2:
            continue
        oof_k = OofFrame(df_k, oof.classifier, oof.target_class)
        p_and_y = _lofo_predict_for_family(oof_k, family, eps=eps)
        if p_and_y is None:
            continue
        p_lofo, y = p_and_y
        e, m = _ece_mce(y, p_lofo, n_bins)
        rows.append({
            "bootstrap_idx": b,
            "log_loss": _log_loss(y, p_lofo),
            "brier":    _brier(y, p_lofo),
            "ece":      e,
            "mce":      m,
        })
    return pd.DataFrame(rows)


def _bootstrap_calibrator_params(
    oof: OofFrame,
    family: str,
    *,
    n_iter: int,
    seed: int,
    eps: float,
    bootstrap_unit: str,
    cluster_map: dict[str, str] | None,
) -> pd.DataFrame:
    """Cluster-block bootstrap CIs on the winning family's parameters.

    Each draw refits the family on the resample and records
    ``{a, b, c, T}`` (unused params are NaN). Callers take quantiles
    to build parameter CIs."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for b in range(n_iter):
        idx = _resample_indices(bootstrap_unit, oof.df, rng, cluster_map)
        df_k = oof.df.iloc[idx].reset_index(drop=True)
        cal = _refit_for_family(family, df_k, eps=eps)
        if cal is None:
            continue
        p = cal.params_dict()
        rows.append({
            "bootstrap_idx": b,
            "a": p.get("a"), "b": p.get("b"),
            "c": p.get("c"), "T": p.get("T"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reliability and calibration metrics
# ---------------------------------------------------------------------------

def _quantile_bin_edges(p: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile bin edges with duplicates collapsed."""
    if len(p) == 0:
        return np.array([])
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(p, qs))
    if edges[0] > 0.0:
        edges = np.concatenate([[0.0], edges])
    if edges[-1] < 1.0:
        edges = np.concatenate([edges, [1.0]])
    return edges


def reliability_table(
    y: np.ndarray,
    p: np.ndarray,
    *,
    n_bins: int = DEFAULT_N_RELIABILITY_BINS,
    z: float = Z_TWO_SIDED_95,
) -> pd.DataFrame:
    """Per-bin reliability stats with binomial Wilson CIs."""
    y = np.asarray(y).astype(np.int8)
    p = np.asarray(p).astype(np.float64)
    base_cols = ["bin_idx", "edge_lo", "edge_hi", "n", "n_pos",
                 "p_pred_mean", "p_obs", "wilson_lo", "wilson_hi"]
    if len(p) == 0:
        return pd.DataFrame(columns=base_cols)
    edges = _quantile_bin_edges(p, n_bins)
    if len(edges) < 2:
        return pd.DataFrame(columns=base_cols)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, len(edges) - 2)
    rows: list[dict] = []
    for b in range(len(edges) - 1):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            continue
        n_pos = int(y[m].sum())
        wl, wh = wilson_interval(n_pos, n, z=z)
        rows.append({
            "bin_idx": b,
            "edge_lo": float(edges[b]),
            "edge_hi": float(edges[b + 1]),
            "n": n,
            "n_pos": n_pos,
            "p_pred_mean": float(p[m].mean()),
            "p_obs": n_pos / n,
            "wilson_lo": wl,
            "wilson_hi": wh,
        })
    return pd.DataFrame(rows)


def _ece_mce(y: np.ndarray, p: np.ndarray, n_bins: int) -> tuple[float, float]:
    if len(p) == 0:
        return float("nan"), float("nan")
    edges = _quantile_bin_edges(p, n_bins)
    if len(edges) < 2:
        return float("nan"), float("nan")
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, len(edges) - 2)
    n_total = len(p)
    ece = 0.0
    mce = 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            continue
        diff = abs(float(p[m].mean()) - float(y[m].mean()))
        ece += (n / n_total) * diff
        if diff > mce:
            mce = diff
    return ece, mce


def calibration_metrics(
    y: np.ndarray,
    p_raw: np.ndarray,
    p_cal: np.ndarray,
    *,
    n_bins: int = DEFAULT_N_RELIABILITY_BINS,
) -> dict:
    """Log-loss, Brier, ECE, MCE for raw-as-prob vs. calibrated."""
    y = np.asarray(y).astype(np.int8)
    p_raw = np.asarray(p_raw).astype(np.float64)
    p_cal = np.asarray(p_cal).astype(np.float64)
    e_raw, m_raw = _ece_mce(y, p_raw, n_bins)
    e_cal, m_cal = _ece_mce(y, p_cal, n_bins)
    return {
        "log_loss_raw": _log_loss(y, p_raw),
        "log_loss_cal": _log_loss(y, p_cal),
        "brier_raw": _brier(y, p_raw),
        "brier_cal": _brier(y, p_cal),
        "ece_raw": e_raw, "ece_cal": e_cal,
        "mce_raw": m_raw, "mce_cal": m_cal,
        "n": int(len(y)),
        "n_pos": int(y.sum()),
    }


# ---------------------------------------------------------------------------
# Hard-error inspection
# ---------------------------------------------------------------------------

def hard_error_panel(
    lofo_df: pd.DataFrame,
    *,
    top_k_fp: int = DEFAULT_TOP_K_FP,
    bottom_k_fn: int = DEFAULT_BOTTOM_K_FN,
) -> pd.DataFrame:
    """Top-K FPs (label=0, highest p_lofo) and bottom-K FNs (label=1, lowest)."""
    if lofo_df.empty:
        return pd.DataFrame(columns=list(lofo_df.columns) + ["kind"])
    fps = (
        lofo_df[lofo_df["label"] == 0]
        .sort_values("p_lofo", ascending=False).head(top_k_fp).copy()
    )
    fns = (
        lofo_df[lofo_df["label"] == 1]
        .sort_values("p_lofo", ascending=True).head(bottom_k_fn).copy()
    )
    fps["kind"] = "FP"
    fns["kind"] = "FN"
    return pd.concat([fps, fns], ignore_index=True)


# ---------------------------------------------------------------------------
# Multi-classifier orchestration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CalibrationArtefacts:
    fit_summary: pd.DataFrame
    selection_log_loss: pd.DataFrame
    skipped: pd.DataFrame
    lofo_predictions: pd.DataFrame
    reliability: pd.DataFrame          # pooled LOFO reliability
    reliability_per_fold: pd.DataFrame # 5 per-fold reliability tables
    metrics: pd.DataFrame              # pooled LOFO log_loss / Brier / ECE / MCE
    metrics_per_fold: pd.DataFrame     # per-fold LOFO metrics + mean/SD summary row
    lofo_metric_ci: pd.DataFrame       # cluster-block bootstrap CIs on the pooled LOFO metrics
    param_ci: pd.DataFrame             # cluster-block bootstrap CIs on winning-family params
    ribbon: pd.DataFrame               # cluster-block curve ribbon (p_lo, p_hat, p_hi)
    ribbon_coverage: pd.DataFrame      # deployment-in-ribbon check summary per class
    per_fold_params: pd.DataFrame
    fold_drift_summary: pd.DataFrame
    hard_errors: pd.DataFrame


def _fit_to_row(fit: CalibrationFit) -> dict:
    return {
        "classifier": fit.classifier,
        "target_class": fit.target_class,
        "n_pos": fit.n_pos,
        "n_neg": fit.n_neg,
        "family": fit.family,
        "a": fit.a,
        "b": fit.b,
        "c": fit.c,
        "T": fit.T,
        "eps": fit.eps,
        "status": fit.status,
        "skip_reason": fit.skip_reason,
        "n_reliability_bins_used": fit.n_reliability_bins_used,
        "identity_vetoed": fit.identity_vetoed,
        "identity_veto_reason": fit.identity_veto_reason,
    }


def fit_calibration_table(
    oof_per_clf_class: Mapping[str, Mapping[str, OofFrame]],
    *,
    families: Sequence[str] = DEFAULT_FAMILIES,
    family_tolerance: float = DEFAULT_FAMILY_TOLERANCE,
    min_n_pos: int = DEFAULT_MIN_N_POS,
    eps: float = DEFAULT_SCORE_EPS,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    n_reliability_bins: int = DEFAULT_N_RELIABILITY_BINS,
    top_k_fp: int = DEFAULT_TOP_K_FP,
    bottom_k_fn: int = DEFAULT_BOTTOM_K_FN,
    ci: float = DEFAULT_CI,
    fold_drift_threshold: float = DEFAULT_FOLD_DRIFT_THRESHOLD,
    bootstrap_unit: str = "cluster",
    cluster_map: dict[str, str] | None = None,
) -> CalibrationArtefacts:
    """End-to-end calibration over every (classifier, class) pair.

    ``bootstrap_unit``/``cluster_map`` route the curve ribbon and the new
    LOFO-metric/parameter bootstrap CIs to a 50%-seq-identity cluster
    block bootstrap (default) instead of the legacy 5-fold cluster.
    """
    fit_rows: list[dict] = []
    selection_rows: list[dict] = []
    skip_rows: list[dict] = []
    lofo_pieces: list[pd.DataFrame] = []
    reliability_pieces: list[pd.DataFrame] = []
    reliability_pf_pieces: list[pd.DataFrame] = []
    metrics_rows: list[dict] = []
    metrics_pf_pieces: list[pd.DataFrame] = []
    lofo_metric_ci_rows: list[dict] = []
    param_ci_rows: list[dict] = []
    ribbon_pieces: list[pd.DataFrame] = []
    ribbon_coverage_rows: list[dict] = []
    pfp_pieces: list[pd.DataFrame] = []
    drift_rows: list[dict] = []
    hard_pieces: list[pd.DataFrame] = []

    total_cells = sum(len(cm) for cm in oof_per_clf_class.values())
    logger.info(
        "Fitting calibration for %d (classifier, class) cells across %d "
        "candidate families%s",
        total_cells, len(families),
        f", ribbon via cluster-block bootstrap ({n_bootstrap} draws)" if
        bootstrap_unit == "cluster" and cluster_map else "",
    )
    from tqdm.auto import tqdm as _tqdm  # type: ignore
    cells = [
        (clf, cls, oof)
        for clf, cls_map in oof_per_clf_class.items()
        for cls, oof in cls_map.items()
    ]
    for clf, cls, oof in _tqdm(cells, desc="calibration cells"):
        if True:  # keep indentation for the fit block below
            fit = fit_best_calibrator(
                oof, families=families, min_n_pos=min_n_pos,
                eps=eps, family_tolerance=family_tolerance,
                n_reliability_bins=n_reliability_bins,
            )
            n_bins_eff = fit.n_reliability_bins_used

            for diag in fit.selection_diagnostics:
                selection_rows.append({
                    "classifier": clf, "target_class": cls,
                    **diag,
                })

            if fit.calibrator is None:
                fit_rows.append(_fit_to_row(fit))
                skip_rows.append({
                    "classifier": clf,
                    "target_class": cls,
                    "n_pos": fit.n_pos, "n_neg": fit.n_neg,
                    "status": fit.status,
                    "reason": fit.skip_reason,
                })
                continue

            family = fit.family
            assert family is not None

            # LOFO predictions for the winning family.
            lofo = evaluate_lofo(oof, family, eps=eps)
            if not lofo.empty:
                lofo.insert(0, "target_class", cls)
                lofo.insert(0, "classifier", clf)
                lofo.insert(2, "family", family)
                lofo_pieces.append(lofo)

                rel = reliability_table(
                    lofo["label"].to_numpy(),
                    lofo["p_lofo"].to_numpy(),
                    n_bins=n_bins_eff,
                )
                if not rel.empty:
                    rel.insert(0, "target_class", cls)
                    rel.insert(0, "classifier", clf)
                    rel.insert(2, "family", family)
                    reliability_pieces.append(rel)

                # Per-fold reliability (for the 5-CV curve-overlap plot).
                rel_pf = per_fold_reliability(lofo, n_bins=n_bins_eff)
                if not rel_pf.empty:
                    rel_pf.insert(0, "target_class", cls)
                    rel_pf.insert(0, "classifier", clf)
                    rel_pf.insert(2, "family", family)
                    reliability_pf_pieces.append(rel_pf)

                metrics_rows.append({
                    "classifier": clf,
                    "target_class": cls,
                    "family": family,
                    "n_reliability_bins_used": n_bins_eff,
                    **calibration_metrics(
                        lofo["label"].to_numpy(),
                        lofo["raw_score"].to_numpy(),
                        lofo["p_lofo"].to_numpy(),
                        n_bins=n_bins_eff,
                    ),
                })

                # Per-fold LOFO metrics + mean/SD summary row.
                pf_metrics = per_fold_lofo_metrics(lofo, n_bins=n_bins_eff)
                if not pf_metrics.empty:
                    pf_metrics.insert(0, "target_class", cls)
                    pf_metrics.insert(0, "classifier", clf)
                    pf_metrics.insert(2, "family", family)
                    metrics_pf_pieces.append(pf_metrics)

                he = hard_error_panel(
                    lofo, top_k_fp=top_k_fp, bottom_k_fn=bottom_k_fn,
                )
                if not he.empty:
                    hard_pieces.append(he)

            # Cluster-bootstrap ribbon (winning family) + coverage check.
            ribbon = cluster_bootstrap_calibrator(
                oof, fit.calibrator, family=family,
                n_iter=n_bootstrap, seed=bootstrap_seed,
                eps=eps, ci=ci,
                bootstrap_unit=bootstrap_unit,
                cluster_map=cluster_map,
            )
            ribbon_pieces.append(ribbon.to_long_frame())
            if ribbon.p_hat.size:
                outside = np.logical_or(
                    ribbon.p_hat < ribbon.p_lo,
                    ribbon.p_hat > ribbon.p_hi,
                )
                pct_outside = float(np.mean(outside)) * 100.0
            else:
                pct_outside = float("nan")
            ribbon_coverage_rows.append({
                "classifier": clf,
                "target_class": cls,
                "family": family,
                "bootstrap_unit": bootstrap_unit,
                "n_resamples_used": ribbon.n_resamples_used,
                "pct_deployment_outside_ribbon": pct_outside,
                "ci": ci,
            })

            # Bootstrap CIs on LOFO metrics + winning-family parameters.
            metric_boot = _bootstrap_lofo_metrics(
                oof, family, n_iter=n_bootstrap, seed=bootstrap_seed,
                eps=eps, bootstrap_unit=bootstrap_unit,
                cluster_map=cluster_map, n_bins=n_bins_eff,
            )
            if not metric_boot.empty:
                pooled_pt = calibration_metrics(
                    lofo["label"].to_numpy(),
                    lofo["raw_score"].to_numpy(),
                    lofo["p_lofo"].to_numpy(),
                    n_bins=n_bins_eff,
                ) if not lofo.empty else {}
                alpha = (1.0 - ci) / 2.0
                for metric_name, pooled_key in (
                    ("log_loss", "log_loss_cal"),
                    ("brier",    "brier_cal"),
                    ("ece",      "ece_cal"),
                    ("mce",      "mce_cal"),
                ):
                    vals = metric_boot[metric_name].to_numpy(dtype=float)
                    vals = vals[np.isfinite(vals)]
                    if vals.size == 0:
                        continue
                    lofo_metric_ci_rows.append({
                        "classifier": clf,
                        "target_class": cls,
                        "family": family,
                        "metric": metric_name,
                        "point": float(pooled_pt.get(pooled_key, float("nan"))),
                        "ci_low": float(np.quantile(vals, alpha)),
                        "ci_high": float(np.quantile(vals, 1.0 - alpha)),
                        "n_resamples_used": int(vals.size),
                        "ci": ci,
                    })
            param_boot = _bootstrap_calibrator_params(
                oof, family, n_iter=n_bootstrap, seed=bootstrap_seed,
                eps=eps, bootstrap_unit=bootstrap_unit, cluster_map=cluster_map,
            )
            if not param_boot.empty:
                deploy_params = fit.calibrator.params_dict()
                alpha = (1.0 - ci) / 2.0
                for pname in ("a", "b", "c", "T"):
                    col = param_boot[pname].to_numpy(dtype=float)
                    col = col[np.isfinite(col)]
                    if col.size == 0 or deploy_params.get(pname) is None:
                        continue
                    param_ci_rows.append({
                        "classifier": clf,
                        "target_class": cls,
                        "family": family,
                        "parameter": pname,
                        "point": float(deploy_params[pname]),
                        "ci_low": float(np.quantile(col, alpha)),
                        "ci_high": float(np.quantile(col, 1.0 - alpha)),
                        "n_resamples_used": int(col.size),
                        "ci": ci,
                    })

            # Per-fold drift (winning family).
            pfp = per_fold_param_drift(oof, family, eps=eps)
            if not pfp.empty:
                pfp.insert(0, "target_class", cls)
                pfp.insert(0, "classifier", clf)
                pfp_pieces.append(pfp)
            flagged, spreads = detect_fold_drift(
                pfp, family=family, threshold=fold_drift_threshold,
            )
            drift_rows.append({
                "classifier": clf,
                "target_class": cls,
                "family": family,
                "drift_flagged": flagged,
                "spread_a": spreads.get("a"),
                "spread_b": spreads.get("b"),
                "spread_c": spreads.get("c"),
                "spread_T": spreads.get("T"),
                "threshold": fold_drift_threshold,
            })

            fit_rows.append(_fit_to_row(fit))

    def _maybe_concat(pieces: list[pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()

    return CalibrationArtefacts(
        fit_summary=pd.DataFrame(fit_rows),
        selection_log_loss=pd.DataFrame(selection_rows),
        skipped=pd.DataFrame(skip_rows),
        lofo_predictions=_maybe_concat(lofo_pieces),
        reliability=_maybe_concat(reliability_pieces),
        reliability_per_fold=_maybe_concat(reliability_pf_pieces),
        metrics=pd.DataFrame(metrics_rows),
        metrics_per_fold=_maybe_concat(metrics_pf_pieces),
        lofo_metric_ci=pd.DataFrame(lofo_metric_ci_rows),
        param_ci=pd.DataFrame(param_ci_rows),
        ribbon=_maybe_concat(ribbon_pieces),
        ribbon_coverage=pd.DataFrame(ribbon_coverage_rows),
        per_fold_params=_maybe_concat(pfp_pieces),
        fold_drift_summary=pd.DataFrame(drift_rows),
        hard_errors=_maybe_concat(hard_pieces),
    )


# ---------------------------------------------------------------------------
# Inference-side helpers
# ---------------------------------------------------------------------------

def calibrators_from_fit_summary(
    fit_summary: pd.DataFrame,
    classifier: str,
):
    """Build a ``{class: calibrator}`` dispatch for one classifier.

    Each row is interpreted via its ``family`` column. Rows whose family
    is ``None`` / NaN (skipped or failed) are omitted; the caller should
    then leave those classes uncalibrated at inference time.
    """
    out: dict[str, object] = {}
    sub = fit_summary[fit_summary["classifier"] == classifier]
    for _, row in sub.iterrows():
        fam = row.get("family")
        if not isinstance(fam, str) or not fam or fam == "nan":
            continue
        eps = float(row["eps"])
        try:
            cal = build_calibrator_from_params(
                fam,
                {
                    "a": row.get("a"),
                    "b": row.get("b"),
                    "c": row.get("c"),
                    "T": row.get("T"),
                },
                eps=eps,
            )
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning(
                "Could not build calibrator for %s/%s (%s): %s",
                classifier, row.get("target_class"), fam, exc,
            )
            continue
        out[str(row["target_class"])] = cal
    return out
