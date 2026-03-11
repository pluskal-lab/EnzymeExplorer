import logging
import subprocess
import pandas as pd

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    
class FoldseekWrapper:
    def __init__(self, foldseek_path: str = "foldseek", threads: int = 8):
        self.foldseek_path = foldseek_path
        self.threads = threads

    def easy_search(
        self,
        query_dir: str,
        target_dir: str,
        output: str,
        tmp_dir: str,
        max_seqs: int = 5000,
        e_value: float = 1,
        sensitivity: int = 10,
    ) -> pd.DataFrame:
        cmd = [
            self.foldseek_path,
            "easy-search",
            query_dir,
            target_dir,
            output,
            tmp_dir,
            "-e", str(e_value),
            "--max-seqs", str(max_seqs),
            "-s", str(sensitivity),
            "--format-output", "query,target,alntmscore"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        if result.returncode != 0:
            logger.error("foldseek stderr: %s", result.stderr)
            raise RuntimeError(f"foldseek easy-search failed with return code {result.returncode}")
    
        search_results_df = pd.read_csv(output, sep="\t", header=None, names=["query", "target", "alntmscore"])
        return search_results_df