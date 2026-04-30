"""Class registry for the evaluation pipeline.

Maps short, plot-friendly class names (e.g. "FPP", "TPS") to the substrate
SMILES / sentinel labels stored in the dataset and in saved fold predictions.
This module is the single source of truth for substrate vs detection
groupings and the default plot ordering.
"""

from __future__ import annotations


SHORT_TO_SMILES: dict[str, str] = {
    "FPP": "CC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
    "GPP": "CC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
    "GGPP": "CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
    "EDSQ": "CC(C)=CCCC(C)=CCCC(C)=CCCC=C(C)CCC=C(C)CCC1OC1(C)C",
    "CPP": "CC1(C)CCCC2(C)C1CCC(=C)C2CCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
    "GFPP": "CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
    "2xFPP": (
        "CC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O."
        "CC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O"
    ),
    "2xGGPP": (
        "CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O."
        "CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O"
    ),
    "IDS": "precursor substr",
    "TPS": "isTPS",
}

SMILES_TO_SHORT: dict[str, str] = {v: k for k, v in SHORT_TO_SMILES.items()}

DETECTION_CLASSES: list[str] = ["TPS", "IDS"]

SUBSTRATE_CLASSES: list[str] = [
    "FPP",
    "GPP",
    "GGPP",
    "EDSQ",
    "CPP",
    "GFPP",
    "2xFPP",
    "2xGGPP",
]

ALL_CLASSES: list[str] = DETECTION_CLASSES + SUBSTRATE_CLASSES

DEFAULT_PLOT_ORDER: list[str] = [
    "TPS",
    "IDS",
    "GPP",
    "FPP",
    "GGPP",
    "GFPP",
    "CPP",
    "EDSQ",
    "2xFPP",
    "2xGGPP",
]


def to_short(class_label: str) -> str:
    """Return the short form of a class label, accepting either form."""
    if class_label in SHORT_TO_SMILES:
        return class_label
    return SMILES_TO_SHORT[class_label]


def to_smiles(class_label: str) -> str:
    """Return the SMILES/sentinel form of a class label, accepting either form."""
    if class_label in SMILES_TO_SHORT:
        return class_label
    return SHORT_TO_SMILES[class_label]
