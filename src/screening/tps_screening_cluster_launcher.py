"""This script scores the models against UniProt proteins"""
import argparse
import subprocess
import time

import GPUtil  # type: ignore


def parse_args() -> argparse.Namespace:
    """
    This function parses arguments
    :return: current argparse.Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta", type=int, default=40000)
    parser.add_argument("--session-i", type=int, default=1)
    parser.add_argument("--n-gpus", type=int, default=8)
    parser.add_argument("--fasta-path", type=str, default="data/uniprot_trembl.fasta")
    parser.add_argument("--output-root", type=str, default="trembl_screening")
    return parser.parse_args()


class GpuAllocator:
    """
    Class to manage GPU allocation on a cluster node with 8 GPUs
    """

    def __init__(self, n_gpus: int = 8):
        self.available_gpus = set(
            GPUtil.getAvailable(
                order="PCI_BUS_ID",
                limit=100,  # big M (i.e. unreachable upper bound)
                maxLoad=0.5,
                maxMemory=0.5,
                includeNan=False,
                excludeID=[],
                excludeUUID=[],
            )
        )
        self.process_id_2_gpu_id: dict[subprocess.Popen, int] = {}
        self.n_gpus = n_gpus

    def check_dead_processes(self):
        """
        Check if any of the processes have finished and free the GPU
        """
        for process in list(self.process_id_2_gpu_id.keys()):
            if process.poll() is not None:
                self.available_gpus.add(self.process_id_2_gpu_id[process])
                del self.process_id_2_gpu_id[process]

    def assign_process_to_gpu(self, process: subprocess.Popen, gpu_id: int):
        """
        Assign a process to a GPU
        """
        self.process_id_2_gpu_id[process] = gpu_id
        self.available_gpus.remove(gpu_id)

    def is_gpu_available(self):
        """
        Check if any GPU is available
        :return:
        """
        return len(self.available_gpus) > 0

    def are_all_gpus_available(self):
        """
        Check if all GPUs are available
        :return:
        """
        return len(self.available_gpus) == self.n_gpus

    def get_available_gpu(self):
        """
        Get the first available GPU
        :return:
        """
        assert self.is_gpu_available(), "No gpus"
        return list(self.available_gpus)[0]

    def wait_for_free_gpu(self):
        """
        Actively wait until a GPU is available
        """
        while not self.is_gpu_available():
            self.check_dead_processes()
            time.sleep(5)

    def wait_for_complete_finish(self):
        """
        Actively wait until all processes are finished
        """
        while not self.are_all_gpus_available():
            self.check_dead_processes()
            time.sleep(120)


if __name__ == "__main__":
    args = parse_args()

    gpu_allocator = GpuAllocator()

    starting_i = (args.karolina_session - 1) * args.n_gpus * args.delta

    for gpu_i in range(args.n_gpus):
        with subprocess.Popen(
            [
                "python",
                "-m",
                "src.screening.tps_predict_fasta",
                "--gpu",
                str(gpu_i),
                "--starting-i",
                str(starting_i + gpu_i * args.delta),
                "--end-i",
                str(starting_i + (gpu_i + 1) * args.delta),
                "--batch-size",
                "32",
                "--clf-batch-size",
                "10000",
                "--session-id",
                str(args.karolina_session),
                "--fasta-path",
                args.fasta_path,
                "--output-root",
                args.output_root,
            ]
        ) as current_process:
            gpu_allocator.assign_process_to_gpu(current_process, gpu_i)

    gpu_allocator.wait_for_complete_finish()
