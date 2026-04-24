import logging
import subprocess
import pandas as pd
from Bio import SeqIO
import glob

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    
class MMSeqs2Wrapper:
    def __init__(self, mmseqs_path: str = "mmseqs", threads: int = 8):
        self.mmseqs_path = mmseqs_path
        self.threads = threads

    def easy_cluster(
        self,
        input_fasta: str,
        output: str,
        tmp: str,
        min_seq_id: float,
        coverage: float,
        coverage_mode: int = 0,
        max_seqs: int = 15000,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        cmd = [
            self.mmseqs_path,
            "easy-cluster",
            input_fasta,
            output,
            tmp,
            "--min-seq-id", str(min_seq_id),
            "-c", str(coverage),
            "--cov-mode", str(coverage_mode),
            "--threads", str(self.threads),
            "--max-seqs", str(max_seqs)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        if result.returncode != 0:
            logger.error("mmseqs stderr: %s", result.stderr)
            raise RuntimeError(f"mmseqs easy-cluster failed with return code {result.returncode}")
    
        clusters_df = pd.read_csv(f"{output}_cluster.tsv", sep="\t", header=None, names=["Representative", "Member"])
        representatives = list(SeqIO.parse(f"{output}_rep_seq.fasta", "fasta"))
        representatives_df = pd.DataFrame([{"Representative": repr.id, "Sequence": repr.seq} for repr in representatives])
        return clusters_df, representatives_df
    
    def easy_search(
        self,
        query_fasta: str,
        target_fasta: str,
        output: str,
        tmp: str,
        e_value: float | None = None,
        seq_id: float | None = None,
        num_iterations: int = 1,
        coverage: float | None = None,
        coverage_mode: int | None = None,
        alignment_mode: int | None = None,
        sensitivity: float | None = None,
        get_best_hit: bool = True,
        max_seqs: int | None = None,
    ) -> pd.DataFrame:
        cmd = [
            self.mmseqs_path,
            "easy-search",
            query_fasta,
            target_fasta,
            output,
            tmp,
            "--num-iterations", str(num_iterations),
            "--threads", str(self.threads),
        ]
        if seq_id is not None:
            cmd.extend(["--min-seq-id", str(seq_id)])
        if e_value is not None:
            cmd.extend(["-e", str(e_value)])
        if coverage is not None:
            cmd.extend(["-c", str(coverage)])
        if coverage_mode is not None:
            cmd.extend(["--cov-mode", str(coverage_mode)])
        if alignment_mode is not None:
            cmd.extend(["--alignment-mode", str(alignment_mode)])
        if sensitivity is not None:
            cmd.extend(["-s", str(sensitivity)])
        if max_seqs is not None:
            cmd.extend(["--max-seqs", str(max_seqs)])
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        if result.returncode != 0:
            logger.error("mmseqs stderr: %s", result.stderr)
            raise RuntimeError(f"mmseqs easy-search failed with return code {result.returncode}")
    
        search_results_df = pd.read_csv(output, sep="\t", header=None, names=["query", "target", "perc_identity", "alignment_length", "mismatches", "gap_opens", "q_start", "q_end", "s_start", "s_end", "evalue", "bit_score"])
        if get_best_hit:
            search_results_df.sort_values("evalue", inplace=True)
            search_results_df.drop_duplicates("query", inplace=True)
        return search_results_df