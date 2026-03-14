import numpy as np

# Potential TPS SwissProt IDs without any functional annotation (manually curated)
PUTATIVE_TPS_IDS = [
    "A0A2B7YDW3",
    "P9WEH1",
    "A0A2Z4HPY4",
    "P0CJ42",
    "B8NHE1",
    "A0A084R1K7",
    "A0A6A6H2E0",
    "P9WEH1",
    "Q2UEK4",
    "A0A5Q0QMX6",
    "F0ZHE2",
    "A0A5Q0QRK8",
    "O65688",
    "A0A8H5Z7W4",
    "M2SPA3",
    "A0A348B794",
    "A0A1V0QSA9",
    "P0DXD5",
]


TPS_ECS_BASE = set(["4.2.1.123", "4.2.3.211", "4.2.3.216"])

# GO terms to blacklist for TPS identification even though TPS enzymes may be annotated with them
TPS_GO_BLACKLIST = set(
    [
        "GO:0003723",
        "GO:0009975",
        "GO:0016491",
        "GO:0016740",
        "GO:0016787",
        "GO:0016829",
        "GO:0016853",
        "GO:0042802",
        "GO:0000287",
        "GO:0004452",
        "GO:0016746",
        "GO:0016765",
        "GO:0016791",
        "GO:0016823",
        "GO:0016836",
        "GO:0016866",
        "GO:0016872",
        "GO:0042803",
        "GO:0046872",
        "GO:0030955",
        "GO:0005506",
        "GO:0030145",
        "GO:0016838",
    ]
)

MAJOR_CLASSES = [
    "CC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",  # FPP
    "CC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",  # GPP
    "CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",  # GGPP
    "CC(C)=CCCC(C)=CCCC(C)=CCCC=C(C)CCC=C(C)CCC1OC1(C)C",  # squalene epoxide
    "CC1(C)CCCC2(C)C1CCC(=C)C2CCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",  # copalyl PP
    "CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",  # GFPP
    "CC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O.CC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",  # FPP + FPP
    "CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O.CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",  # GGPP + GGPP
    "precursor substr",
]

METRICS_2_FUNC = {
    "confident_residues": lambda plddts: np.where(plddts >= 70)[0].size / len(plddts),
    "mean": lambda plddts: plddts.mean(),
    "median": lambda plddts: np.median(plddts),
    "conf_segments": lambda plddts: (confidence_segment_lengths(plddts)/len(plddts)).mean() if confidence_segment_lengths(plddts).size > 0 else 0,
    "high_conf_segments": lambda plddts: (confidence_segment_lengths(plddts, threshold=90)/len(plddts)).mean() if confidence_segment_lengths(plddts).size > 0 else 0
}