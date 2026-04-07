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


TPS_ECS_TO_SUBSTRATES_BASE = {
    "2.5.1.30": "precursor substr",
    "2.5.1.67": "precursor substr",
    "2.5.1.69": "precursor substr",
    "2.5.1.82": "precursor substr",
    "2.5.1.83": "precursor substr",
    "2.5.1.84": "precursor substr",
    "2.5.1.85": "precursor substr",
    "2.5.1.90": "precursor substr",
    "2.5.1.91": "precursor substr",
    "2.5.1.150": "precursor substr",
    "3.1.7.5": "CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
    "3.1.7.10": "CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
    "3.1.7.12": "CC1CCC2(C)C(CCC=C2C)C1(C)CCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
    "4.2.1.123": "CC(C)=CCCC(C)=CCCC(C)=CCCC=C(C)CCC=C(C)CCC=C(C)C",
    "4.2.1.138": "CC1CCC2C(CC2(C)C)C(=C)CCC=1",
    "4.2.3.8": "CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
    "4.2.3.41": "CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
    "4.2.3.63": "CC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
    "4.2.3.64": "CC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
    "4.2.3.142": "CC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
    "4.2.3.151": "CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
    "4.2.3.156": "CC(CCC=C(C)CCC=C(C)C)=CC1C(COP([O-])(=O)OP([O-])([O-])=O)C1(C)CCC=C(C)CCC=C(C)C",
    "4.2.3.205": "CC1C(C)C(C)C(C)(CCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O)C=1C",
    "4.2.3.207": "CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
    "4.2.3.208": "CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
    "4.2.3.209": "CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
    "4.2.3.210": "CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
    "4.2.3.211": "CC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
    "4.2.3.212": "CC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
    "4.2.3.213": "CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
    "4.2.3.216": "CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
    "4.2.3.217": "CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
    "4.2.3.222": "CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
    "4.2.3.224": "CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
    "4.2.3.229": "CC1(C)CCCC2(C)C1CCC(=C)C2CCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
    "5.5.1.8": "CC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
}

NON_TPS_ECS = {"2.5.1.142", "2.5.1.28", "2.5.1.68", "2.5.1.92", "4.1.99.16"}

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