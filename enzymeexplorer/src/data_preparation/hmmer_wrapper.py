import logging
import subprocess
import pandas as pd
from Bio import SeqIO
import glob

logger = logging.getLogger(__file__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class HMMerWrapper:
    def __init__(self, threads: int = 8):
        self.threads = threads

    def hmm_concat(self, hmms_dir: str, output_hmm: str):
        hmm_files = glob.glob(f"{hmms_dir}/*.hmm")
        if not hmm_files:
            raise RuntimeError(f"No HMM files found in directory {hmms_dir}")

        concatted_hmm_lines = []

        for hmm_file in hmm_files:
            with open(hmm_file, "r") as f:
                concatted_hmm_lines.extend(f.readlines())

        with open(output_hmm, "w") as f:
            f.writelines(concatted_hmm_lines)

    def hmmpress(self, hmm_file: str):
        cmd = ["hmmpress", hmm_file]

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.returncode != 0:
            logger.error("hmmpress stderr: %s", result.stderr)
            raise RuntimeError(f"hmmpress failed with return code {result.returncode}")

    def hmmscan(
        self,
        query_fasta: str,
        hmm_path: str,
        output: str,
        bitscore: float = 25,
    ) -> pd.DataFrame:
        cmd = [
            "hmmscan",
            "--noali",
            "--notextw",
            "--tblout",
            output,
            "--cpu",
            str(self.threads),
        ]
        if bitscore is not None:
            cmd.extend(["-T", str(bitscore)])

        cmd.extend([hmm_path, query_fasta])

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.returncode != 0:
            logger.error("hmmscan stderr: %s", result.stderr)
            raise RuntimeError(f"hmmscan failed with return code {result.returncode}")

        columns = [
            "target_name",
            "target_accession",
            "query_name",
            "query_accession",
            "E-value_full",
            "score_full",
            "bias_full",
            "E-value_best_dom",
            "score_best_dom",
            "bias_best_dom",
        ]

        hits_df = pd.read_csv(
            output, sep="\s+", comment="#", header=None, names=columns, usecols=range(10)
        )
        return hits_df.sort_values(by=["score_best_dom"], ascending=False).drop_duplicates(subset=["query_name"])
