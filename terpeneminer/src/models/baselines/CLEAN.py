# pylint: disable=C0103
""" This is a wrapper to use the CLEAN model for substrate prediction.
Please note, that before using this wrapper you would need to install CLEAN as per https://github.com/tttianhao/CLEAN
"""
import os
from collections import defaultdict
from shutil import copyfile
from dataclasses import dataclass
from typing import Type, Optional
from pathlib import Path
from uuid import uuid4
import sys
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import wget  # type: ignore

from rdkit.Chem import MolToSmiles, rdChemReactions  # type: ignore

# remove additional 'data' folder from CLEAN's codebase (at the time of my experiments, the CLEAN's scripts were unrunnable without fixes of paths)
from CLEAN.utils import (  # type: ignore
    prepare_infer_fasta,
)

# also remove additional 'data' folder from CLEAN's codebase
from CLEAN.infer import (  # type: ignore
    infer_maxsep,
)

from terpeneminer.src.models.ifaces import BaseModel, BaseConfig
from terpeneminer.src.utils.msa import get_fasta_seqs
from terpeneminer.src.utils.data import get_canonical_smiles


@dataclass
class CLEANConfig(BaseConfig):
    """
    A data class to store CLEAN-model attributes
    """

    clean_installation_root: Path
    rhea2ec_link: str
    rhea_reaction_smiles_link: str
    rhea_directions_link: str
    clean_working_dir: str
    seq_col_name: str


class CLEAN(BaseModel):
    """
    CLEAN model wrapper for prediction of TPS substrates
    """

    def __init__(self, config: CLEANConfig):
        super().__init__(config=config)
        self.working_path = Path(config.clean_working_dir)
        if not self.working_path.exists():
            self.working_path.mkdir()
        rhea2ec_path = self.working_path / "rhea2ec.tsv"
        if not rhea2ec_path.exists():
            wget.download(config.rhea2ec_link, str(rhea2ec_path))
        rhea_reaction_smiles_path = self.working_path / "rhea-reaction-smiles.tsv"
        if not rhea_reaction_smiles_path.exists():
            wget.download(
                config.rhea_reaction_smiles_link, str(rhea_reaction_smiles_path)
            )
        rhea_directions_path = self.working_path / "rhea-directions.tsv"
        if not rhea_directions_path.exists():
            wget.download(config.rhea_directions_link, str(rhea_directions_path))
        self.config: CLEANConfig = config
        self.config.clean_installation_root = Path(self.config.clean_installation_root)

        # rhea reactions mapped to ec numbers
        rhea2ec_df = pd.read_csv(rhea2ec_path, sep="\t")
        ec_2_rheaids = rhea2ec_df.groupby("ID")["RHEA_ID"].agg(set).to_dict()
        rhea2smiles_df = pd.read_csv(rhea_reaction_smiles_path, sep="\t", header=None)
        rheaid_2_rxn = rhea2smiles_df[[0, 1]].set_index(0)[1].to_dict()
        rhea2directed_df = pd.read_csv(rhea_directions_path, sep="\t")
        master_rhea_2_directed = (
            rhea2directed_df[["RHEA_ID_MASTER", "RHEA_ID_LR"]]
            .set_index("RHEA_ID_MASTER")["RHEA_ID_LR"]
            .to_dict()
        )

        self.ec_2_substrates: dict = defaultdict(set)
        for ec_code in ec_2_rheaids:
            ec_class = f"EC:{ec_code}"
            rhea_ids = ec_2_rheaids[ec_code]
            for rhea_id in rhea_ids:
                if rhea_id in master_rhea_2_directed:
                    rhea_id = master_rhea_2_directed[rhea_id]
                if rhea_id in rheaid_2_rxn:
                    rxn_smiles = rheaid_2_rxn[rhea_id]
                    rxn = rdChemReactions.ReactionFromSmarts(rxn_smiles, useSmiles=True)
                    trxn = rdChemReactions.ChemicalReaction(rxn)
                    substrates = trxn.GetReactants()
                    substrates_canonical = {
                        get_canonical_smiles(MolToSmiles(substr))
                        for substr in substrates
                    }
                    self.ec_2_substrates[ec_class] = self.ec_2_substrates[
                        ec_class
                    ].union(substrates_canonical)

        tps_df = pd.read_csv(config.tps_cleaned_csv_path)
        tps_df.loc[
            tps_df["Type (mono, sesq, di, …)"].isin(
                {"ggpps", "fpps", "gpps", "gfpps", "hsqs"}
            ),
            config.target_col_name,
        ] = "precursor substr"
        self.precursor_smiles = set(
            tps_df.loc[
                tps_df["Type (mono, sesq, di, …)"].isin(
                    {"ggpps", "fpps", "gpps", "gfpps", "hsqs"}
                ),
                config.target_col_name,
            ].values
        )

        self.tps_substrate_smiles = {
            substr
            for substr in tps_df[config.target_col_name].values
            if substr not in {"Unknown", "Negative"}
        }
        sys.path.insert(0, str(self.config.clean_installation_root / "app" / "src"))

    def fit_core(self, train_df: pd.DataFrame, class_name: str = None):
        """Just a placeholder for the interface compatibility"""

    def predict_proba(
        self,
        val_df: pd.DataFrame,
        selected_class_name: Optional[str] = None,
    ) -> np.ndarray:
        """
        Function to predict the class probabilities for the given validation data using the CLEAN model.

        :param val_df: A pandas DataFrame containing the validation data.
                       The DataFrame must contain sequence and ID columns as specified in the configuration.
        :param selected_class_name: An optional parameter for selecting a class. Defaults to None.
                                    Note: This model does not support class selection and will raise an assertion error if a class name is provided.

        :return: A numpy ndarray containing the predicted class probabilities.
        """

        assert (
            selected_class_name is None
        ), "This model does not support class selection."

        seqs = val_df[self.config.seq_col_name].values
        ids = val_df[self.config.id_col_name].values
        fasta_str = get_fasta_seqs(seqs, ids)

        temp_fasta_path = (
            self.config.clean_installation_root / "app" / f"_temp_msa_{uuid4()}.fasta"
        )
        with open(temp_fasta_path, "w", encoding="utf-8") as file:
            file.writelines(fasta_str.replace("'", "").replace('"', ""))
        # maybe some locations are redundant,
        # but CLEAN codebase tends to look into multiple places for the same input, so to be safe:
        copyfile(
            temp_fasta_path,
            self.config.clean_installation_root
            / "app"
            / "data"
            / "inputs"
            / temp_fasta_path.name,
        )
        copyfile(
            temp_fasta_path,
            self.config.clean_installation_root / "app" / "data" / temp_fasta_path.name,
        )
        cwd = os.getcwd()
        os.chdir(self.config.clean_installation_root / "app")

        clean_name_convention = str(temp_fasta_path.stem)
        prepare_infer_fasta(clean_name_convention)
        infer_maxsep(
            "split100",
            clean_name_convention,
            report_metrics=False,
            pretrained=True,
            gmm="data/pretrained/gmm_ensumble.pkl",
        )

        with open(
            f"results/{temp_fasta_path.stem}_maxsep.csv", "r", encoding="utf-8"
        ) as file:
            clean_pred_lines = file.readlines()
        os.chdir(cwd)
        os.remove(temp_fasta_path)
        id_2_class_2_conf: dict = defaultdict(dict)
        for line in clean_pred_lines:
            line_splitted = line.split(",")
            for ec_classes in line_splitted[1:]:
                ec_class, dist = ec_classes.replace("\n", "").split("/")
                id_2_class_2_conf[line_splitted[0]][ec_class] = 10 - float(dist)

        id_2_substr_2_conf: dict = defaultdict(dict)
        for uni_id, ec_num_2_conf in id_2_class_2_conf.items():
            for ec_num, conf in ec_num_2_conf.items():
                if ec_num in self.ec_2_substrates and len(
                    self.ec_2_substrates[ec_num].intersection(self.tps_substrate_smiles)
                ):
                    ec_num_substrates = self.ec_2_substrates[ec_num]
                    if len(self.precursor_smiles.intersection(ec_num_substrates)):
                        ec_num_substrates.add("precursor substr")
                    substrates = ec_num_substrates.intersection(
                        self.tps_substrate_smiles
                    )
                    for substr in substrates:
                        id_2_substr_2_conf[uni_id][substr] = conf
        assert isinstance(
            val_df, pd.DataFrame
        ), "the CLEAN requires Uniprot ID and sequences, np.array of numerical representations is not a possible input"
        val_df["substr_2_conf"] = val_df[self.config.id_col_name].map(
            lambda x: {} if x not in id_2_substr_2_conf else id_2_substr_2_conf[x]
        )
        val_proba_np = np.zeros((len(val_df), len(self.config.class_names)))
        for class_i, class_name in enumerate(self.config.class_names):
            if class_name == "isTPS":
                val_proba_np[:, class_i] = val_df["substr_2_conf"].map(
                    lambda x: max(x.values()) if len(x) else 0
                )
            else:
                val_proba_np[:, class_i] = val_df["substr_2_conf"].map(
                    lambda x: x.get(class_name, 0)
                )
        return val_proba_np

    @classmethod
    def config_class(cls) -> Type[CLEANConfig]:
        """
        A getter of the model-specific config class
        :return:  A dataclass for config storage
        """
        return CLEANConfig
