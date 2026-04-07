# pylint: disable=C0103
"""This is a wrapper to use the CLEAN model for substrate prediction.
Please note, that before using this wrapper you would need to install CLEAN as per https://github.com/tttianhao/CLEAN
"""
import json
import logging
import os
import shutil
import sys
import glob
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from shutil import copyfile
from typing import Optional, Type
from uuid import uuid4

import gdown
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

# remove additional 'data' folder from CLEAN's codebase (at the time of my experiments, the CLEAN's scripts were unrunnable without fixes of paths)
from CLEAN.utils import (  # type: ignore
    prepare_infer_fasta,
)

# also remove additional 'data' folder from CLEAN's codebase
from CLEAN.infer import (  # type: ignore
    infer_maxsep,
)

from enzymeexplorer.src.models.ifaces import BaseConfig, BaseModel
from enzymeexplorer.src.utils.msa import get_fasta_seqs

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class CLEANConfig(BaseConfig):
    """
    A data class to store CLEAN-model attributes
    """

    clean_installation_root: Path
    ec_2_substrates_json_path: str
    seq_col_name: str
    is_halo: bool
    pretrained_models_link: str


class CLEAN(BaseModel):
    """
    CLEAN model wrapper for prediction of TPS substrates
    """

    def __init__(self, config: CLEANConfig):
        super().__init__(config=config)
        self.config: CLEANConfig = config
        self.config.clean_installation_root = Path(self.config.clean_installation_root)

        with open(config.ec_2_substrates_json_path, "r", encoding="utf-8") as file:
            self.ec_2_substrates = json.load(file)
        self.ec_2_substrates = {
            ec: set(substrates) for ec, substrates in self.ec_2_substrates.items()
        }

        data_df = pd.read_csv(config.tps_cleaned_csv_path)
        self.is_halo = getattr(config, "is_halo", False)
        if not self.is_halo:
            data_df.loc[
                data_df["Type"].isin(
                {"pt"}
            ),
            config.target_col_name,
            ] = "precursor substr"
            self.precursor_smiles = set(
                data_df.loc[
                    data_df["Type"].isin(
                        {"pt"}
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

        self.pretrained_models_dir = (
            self.config.clean_installation_root / "pretrained_models"
        )
        self._download_and_unpack_pretrained_models()

        sys.path.insert(0, str(self.config.clean_installation_root / "app" / "src"))

    def _download_and_unpack_pretrained_models(self):
        pretrained_zip_path = self.config.clean_installation_root / "pretrained_models.zip"
        self.pretrained_models_dir.mkdir(parents=True, exist_ok=True)
        for f in glob.glob(str(self.pretrained_models_dir / "*")):
            os.remove(f)

        gdown.download(
            self.config.pretrained_models_link,
            str(pretrained_zip_path),
            quiet=False,
        )
        shutil.unpack_archive(pretrained_zip_path, self.pretrained_models_dir)
        logger.info(
            "Pretrained models downloaded and unpacked to %s",
            self.pretrained_models_dir,
        )

    def _stage_pretrained_files(self, app_root: Path, fold_idx: int):
        pretrained_dir = app_root / "data" / "pretrained"
        pretrained_dir.mkdir(parents=True, exist_ok=True)

        source_to_destination = {
            self.pretrained_models_dir
            / f"FOLD_{fold_idx}_MODEL.pth": pretrained_dir
            / "split100.pth",
            self.pretrained_models_dir
            / f"FOLD_{fold_idx}_GMM.pkl": pretrained_dir
            / "gmm_ensumble.pkl",
            self.pretrained_models_dir
            / f"FOLD_{fold_idx}_EMBEDDINGS.pt": pretrained_dir
            / "100.pt",
            self.pretrained_models_dir
            / f"FOLD_{fold_idx}_TRAIN_DATA.csv": app_root
            / "data"
            / "split100.csv",
        }
        for source_path, destination_path in source_to_destination.items():
            if not source_path.exists():
                raise FileNotFoundError(
                    "Missing CLEAN pretrained asset for fold "
                    f"{fold_idx}: {source_path}"
                )
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            copyfile(source_path, destination_path)

    def _read_clean_predictions(
        self, results_path: Path
    ) -> dict[str, list[tuple[str, float]]]:
        id_2_ec_scores: dict[str, list[tuple[str, float]]] = defaultdict(list)
        with open(results_path, "r", encoding="utf-8") as file:
            for line in file:
                entries = line.strip().split(",")
                if not entries or not entries[0]:
                    continue
                protein_id = entries[0]
                for entry in entries[1:]:
                    if not entry:
                        continue
                    ec_class, score = entry.split("/")
                    normalized_ec = ec_class.replace("EC:", "")
                    confidence = float(score)
                    if confidence == 0:
                        confidence = 1e-6
                    id_2_ec_scores[protein_id].append((normalized_ec, confidence))
        return id_2_ec_scores

    def _convert_clean_output_to_probabilities(
        self, id_2_ec_scores: dict[str, list[tuple[str, float]]], ids: np.ndarray
    ) -> np.ndarray:
        val_proba_np = np.zeros((len(ids), len(self.config.class_names)))
        class_name_2_idx = {
            class_name: class_i
            for class_i, class_name in enumerate(self.config.class_names)
        }

        for row_i, protein_id in enumerate(ids):
            ec_scores = id_2_ec_scores.get(protein_id, [])
            if self.is_halo:
                for ec_num, conf in ec_scores:
                    class_i = class_name_2_idx.get(ec_num)
                    if class_i is not None:
                        val_proba_np[row_i, class_i] = conf
                continue

            neg_score = 0.0
            tps_pos_score = 0.0
            class_2_pos_score: dict[str, float] = defaultdict(float)

            for ec_num, conf in ec_scores:
                substrates = self.ec_2_substrates.get(ec_num, set())
                if not substrates:
                    neg_score += conf
                    continue
                if substrates - {"precursor substr"}:
                    tps_pos_score += conf
                for substrate in substrates.intersection(self.tps_substrate_smiles):
                    class_2_pos_score[substrate] += conf

            tps_denominator = tps_pos_score + neg_score
            if "isTPS" in class_name_2_idx and tps_denominator > 0:
                val_proba_np[row_i, class_name_2_idx["isTPS"]] = (
                    tps_pos_score / tps_denominator
                )

            for class_name, class_i in class_name_2_idx.items():
                if class_name == "isTPS":
                    continue
                pos_score = class_2_pos_score.get(class_name, 0.0)
                denominator = pos_score + neg_score
                if denominator > 0:
                    val_proba_np[row_i, class_i] = pos_score / denominator

        return val_proba_np

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
        assert fold_idx is not None, "CLEAN inference requires a validation fold index."
        assert isinstance(
            val_df, pd.DataFrame
        ), "the CLEAN requires Uniprot ID and sequences, np.array of numerical representations is not a possible input"

        seqs = val_df[self.config.seq_col_name].values
        ids = val_df[self.config.id_col_name].values
        fasta_str = get_fasta_seqs(seqs, ids)

        app_root = self.config.clean_installation_root / "app"
        temp_fasta_path = app_root / f"_temp_msa_{uuid4()}.fasta"
        with open(temp_fasta_path, "w", encoding="utf-8") as file:
            file.writelines(fasta_str.replace("'", "").replace('"', ""))

        copyfile(temp_fasta_path, app_root / "data" / "inputs" / temp_fasta_path.name)
        copyfile(temp_fasta_path, app_root / "data" / temp_fasta_path.name)

        cwd = os.getcwd()
        results_path = app_root / "results" / f"{temp_fasta_path.stem}_maxsep.csv"
        try:
            self._stage_pretrained_files(app_root, fold_idx)
            os.chdir(app_root)

            clean_name_convention = str(temp_fasta_path.stem)
            logger.info("Running CLEAN on %s", clean_name_convention)
            prepare_infer_fasta(clean_name_convention)
            infer_maxsep(
                "split100",
                clean_name_convention,
                report_metrics=False,
                pretrained=True,
                model_name=None,
                gmm="data/pretrained/gmm_ensumble.pkl",
            )
            id_2_ec_scores = self._read_clean_predictions(results_path)
            return self._convert_clean_output_to_probabilities(id_2_ec_scores, ids)
        finally:
            os.chdir(cwd)
            if temp_fasta_path.exists():
                os.remove(temp_fasta_path)

    @classmethod
    def config_class(cls) -> Type[CLEANConfig]:
        """
        A getter of the model-specific config class
        :return:  A dataclass for config storage
        """
        return CLEANConfig
