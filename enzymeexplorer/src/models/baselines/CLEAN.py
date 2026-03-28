# pylint: disable=C0103
"""This is a wrapper to use the CLEAN model for substrate prediction.
Please note, that before using this wrapper you would need to install CLEAN as per https://github.com/tttianhao/CLEAN
"""
import json
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
import logging

from rdkit.Chem import MolToSmiles, rdChemReactions  # type: ignore

# remove additional 'data' folder from CLEAN's codebase (at the time of my experiments, the CLEAN's scripts were unrunnable without fixes of paths)
from CLEAN.utils import (  # type: ignore
    prepare_infer_fasta,
)

# also remove additional 'data' folder from CLEAN's codebase
from CLEAN.infer import (  # type: ignore
    infer_maxsep,
)

from enzymeexplorer.src.models.ifaces import BaseModel, BaseConfig
from enzymeexplorer.src.utils.msa import get_fasta_seqs
from enzymeexplorer.src.utils.data import get_canonical_smiles

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class CLEANConfig(BaseConfig):
    """
    A data class to store CLEAN-model attributes
    """

    clean_installation_root: Path
    ec_2_substrates_json_path: str
    clean_working_dir: str
    seq_col_name: str
    is_halo: bool
    pretrained_model_name: str


class CLEAN(BaseModel):
    """
    CLEAN model wrapper for prediction of TPS substrates
    """

    def __init__(self, config: CLEANConfig):
        super().__init__(config=config)
        self.working_path = Path(config.clean_working_dir)
        if not self.working_path.exists():
            self.working_path.mkdir()

        self.config: CLEANConfig = config
        self.config.clean_installation_root = Path(self.config.clean_installation_root)

        self.ec_2_substrates = json.load(open(config.ec_2_substrates_json_path, "r"))
        self.ec_2_substrates = {
            ec: set(substrates)
            for ec, substrates in self.ec_2_substrates.items()
        }

        data_df = pd.read_csv(config.tps_cleaned_csv_path)
        if hasattr(config, "is_halo"):
            self.is_halo = config.is_halo
        else:
            self.is_halo = False
        if not self.is_halo:
            data_df.loc[
                data_df["Type (mono, sesq, di, …)"].isin(
                    {"ggpps", "fpps", "gpps", "gfpps", "hsqs"}
                ),
                config.target_col_name,
            ] = "precursor substr"
            self.precursor_smiles = set(
                data_df.loc[
                    data_df["Type (mono, sesq, di, …)"].isin(
                        {"ggpps", "fpps", "gpps", "gfpps", "hsqs"}
                    ),
                    config.target_col_name,
                ].values
            )
        else:
            self.precursor_smiles = set()

        self.tps_substrate_smiles = {
            substr
            for substr in data_df[config.target_col_name].values
            if substr not in {"Unknown", "Negative"}
        }
        sys.path.insert(0, str(self.config.clean_installation_root / "app" / "src"))

    def fit_core(self, train_df: pd.DataFrame, class_name: str = None):
        """Just a placeholder for the interface compatibility"""
        pass

    def predict_proba(
        self,
        val_df: pd.DataFrame,
        selected_class_name: Optional[str] = None,
        fold_idx: Optional[int] = None,
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
        logger.info(f"Running CLEAN on {clean_name_convention}")
        prepare_infer_fasta(clean_name_convention)
        train_data = "split100"
        pretrained_model = None
        gmm = "data/pretrained/gmm_ensumble.pkl"
        if self.config.pretrained_model_name is not None:
            train_data = self.config.pretrained_model_name + f"_{fold_idx}_train"
            pretrained_model = self.config.pretrained_model_name + f"_{fold_idx}"
            gmm = f"data/pretrained/gmm_{self.config.pretrained_model_name}_{fold_idx}.pkl"
        infer_maxsep(
            train_data,
            clean_name_convention,
            report_metrics=False,
            pretrained=True,
            model_name=pretrained_model,
            gmm=gmm,
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
                id_2_class_2_conf[line_splitted[0]][ec_class] = float(dist)

        id_2_substr_2_conf: dict = defaultdict(dict)
        if not self.is_halo:
            for uni_id, ec_num_2_conf in id_2_class_2_conf.items():
                for ec_num, conf in ec_num_2_conf.items():
                    if ec_num in self.ec_2_substrates:
                        if len(self.ec_2_substrates[ec_num] - {"precursor substr"}):
                            id_2_substr_2_conf[uni_id]["isTPS"] = max(
                                conf, id_2_substr_2_conf[uni_id].get("isTPS", 0)
                            )
                        if len(
                            self.ec_2_substrates[ec_num].intersection(
                                self.tps_substrate_smiles
                            )
                        ):
                            ec_num_substrates = self.ec_2_substrates[ec_num]
                            substrates = ec_num_substrates.intersection(
                                self.tps_substrate_smiles
                            )
                            for substr in substrates:
                                id_2_substr_2_conf[uni_id][substr] = conf
        else:
            for uni_id, ec_num_2_conf in id_2_class_2_conf.items():
                for ec_num, conf in ec_num_2_conf.items():
                    ec_num = ec_num.replace("EC:", "")
                    if ec_num in self.config.class_names:
                        id_2_substr_2_conf[uni_id][ec_num] = conf
        assert isinstance(
            val_df, pd.DataFrame
        ), "the CLEAN requires Uniprot ID and sequences, np.array of numerical representations is not a possible input"
        val_df["substr_2_conf"] = val_df[self.config.id_col_name].map(
            lambda x: {} if x not in id_2_substr_2_conf else id_2_substr_2_conf[x]
        )
        val_proba_np = np.zeros((len(val_df), len(self.config.class_names)))
        for class_i, class_name in enumerate(self.config.class_names):
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
