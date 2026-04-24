# TODO: Adjust evaluation logic for the tps prediction

""" This is a wrapper to use TPS Pfam and SUPFAM models for TPS detection. """
from dataclasses import dataclass
from pathlib import Path
from typing import Type, Optional
from uuid import uuid4
import logging
import subprocess
import tempfile
from enzymeexplorer.src.data_preparation.hmmer_wrapper import HMMerWrapper
import os
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from enzymeexplorer.src.models.ifaces import BaseModel, BaseConfig
from enzymeexplorer.src.utils.msa import get_fasta_seqs

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


@dataclass
class PFamSUPFAMConfig(BaseConfig):
    """
    A data class to store Blast-model attributes
    """

    bitscore: float
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
        self.root_path_to_models = config.root_path_to_models
        self.bitscore = config.bitscore
        self.config: PFamSUPFAMConfig = config
        self.hmmer = HMMerWrapper(threads=config.n_jobs if config.n_jobs is not None else 8)

    def fit_core(self, train_df: pd.DataFrame, class_name: str = None):
        """A placeholder for the compatibility with the BaseModel interface"""

    def predict_proba(
        self,
        val_df: pd.DataFrame,
        selected_class_name: Optional[str] = None,
        fold_idx: Optional[int] = None,
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
        
        with tempfile.TemporaryDirectory() as tmpdir:
            uuid = str(uuid4())
            input_fasta_path = os.path.join(tmpdir, f"input_{uuid}.fasta")
            with open(input_fasta_path, "w") as fasta_file:
                for _, row in val_df.iterrows():
                    fasta_file.write(f">{row[self.config.id_col_name]}\n{row[self.config.seq_col_name]}\n")

            pfam_db_path = os.path.join(tmpdir, f"pfam_models_{uuid}")
            self.hmmer.hmm_concat(self.root_path_to_models, pfam_db_path)
            self.hmmer.hmmpress(pfam_db_path)
            pfam_hits_df = self.hmmer.hmmscan(
                query_fasta=input_fasta_path,
                hmm_path=pfam_db_path,
                output=os.path.join(tmpdir, f"pfam_scan_{uuid}.tbl"),
                bitscore=self.bitscore,
            )
        hit_seqs = set(pfam_hits_df["query_name"].unique())
        val_df["isTPS"] = val_df[self.config.id_col_name].apply(lambda x: 1 if x in hit_seqs else 0)
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
