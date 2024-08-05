""" This is a wrapper to use TPS Pfam and SUPFAM models for TPS detection. """
from dataclasses import dataclass
from pathlib import Path
from typing import Type, Optional
from uuid import uuid4
import logging
import subprocess

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from src.models.ifaces import BaseModel, BaseConfig
from src.utils.msa import get_fasta_seqs

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


@dataclass
class PFamSUPFAMConfig(BaseConfig):
    """
    A data class to store Blast-model attributes
    """

    e_threshold: float
    root_path_to_models: str
    working_directory: str
    seq_col_name: str
    n_jobs: Optional[int] = 64


class PfamSUPFAM(BaseModel):
    """
    Pfam SUPFAM profile HMMs for TPS detection
    """

    def __init__(self, config: PFamSUPFAMConfig):
        super().__init__(config=config)
        self.working_path = Path(config.working_directory)
        if not self.working_path.exists():
            self.working_path.mkdir()
        self.paths_to_models = list(Path(config.root_path_to_models).glob("*.hmm"))
        self.e_threshold = config.e_threshold
        self.config: PFamSUPFAMConfig = config

    def fit_core(self, train_df: pd.DataFrame, class_name: str = None):
        """A placeholder for the compatibility with the BaseModel interface"""

    def predict_proba(
        self,
        val_df: pd.DataFrame,
        selected_class_name: Optional[str] = None,
    ) -> np.ndarray:
        """
        Function to predict class probabilities for the given validation data using profile Hidden Markov Models (pHMM).

        :param val_df: A pandas DataFrame containing the validation data.
        :param selected_class_name: An optional parameter for selecting a class. Defaults to None.
                                    Note: This model does not support class selection and will raise an assertion error if a class name is provided.

        :return: A numpy ndarray containing the predicted class probabilities.
        """

        assert isinstance(
            val_df, pd.DataFrame
        ), "This model does not support class selection."
        assert (
            selected_class_name is None
        ), "This model does not support class selection."
        fasta_str = get_fasta_seqs(
            val_df[self.config.seq_col_name].values,
            val_df[self.config.id_col_name].values,
        )
        temp_fasta_path = self.working_path / f"_temp_msa_{uuid4()}.fasta"
        with open(temp_fasta_path, "w", encoding="utf-8") as file:
            file.writelines(fasta_str.replace("'", "").replace('"', ""))
        id_2_min_eval: dict = {}
        for model_path in self.paths_to_models:
            logger.info("Processing pHMM model %s...", str(model_path))
            output_path = self.working_path / model_path.stem
            subprocess.run(
                f"hmmsearch --noali --notextw --tblout {output_path} -E {self.e_threshold} --cpu {self.config.n_jobs} {model_path} {temp_fasta_path}".split(),
                check=False,
            )
            with open(output_path, "r", encoding="utf-8") as file:
                lines = file.readlines()
            for line in lines:
                if line[0] != "#":
                    entries = line.split()
                    id_2_min_eval[entries[0]] = max(
                        float(entries[5]), id_2_min_eval.get(entries[0], -1000000)
                    )
        val_df["isTPS"] = val_df["Uniprot ID"].map(
            lambda x: id_2_min_eval.get(x, -1000000)
        )
        val_proba_np = np.zeros((len(val_df), len(self.config.class_names)))
        for class_i, class_name in enumerate(self.config.class_names):
            if class_name == "isTPS":
                val_proba_np[:, class_i] = val_df["isTPS"].values
        return val_proba_np

    @classmethod
    def config_class(cls) -> Type[PFamSUPFAMConfig]:
        """
        A getter of the model-specific config class
        :return:  A dataclass for config storage
        """
        return PFamSUPFAMConfig
