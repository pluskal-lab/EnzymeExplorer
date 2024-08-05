"""This script scores the models against UniProt proteins"""
import argparse
import subprocess
import time
import logging

# import GPUtil  # type: ignore

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


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
    parser.add_argument("--model", type=str, default="esm-1v-finetuned-subseq")
    parser.add_argument("--detection-threshold", type=float, default=0.2)
    parser.add_argument("--detect-precursor-synthases", action="store_true")

    return parser.parse_args()


def get_amd_gpu_utilization():
    """
    Function to retrieve the utilization of AMD GPUs using the `rocm-smi` tool.

    :return: A dictionary mapping GPU IDs to their utilization rates as a float value between 0 and 1.
    """

    result = subprocess.run(
        ["rocm-smi", "--showuse"], capture_output=True, text=True, check=False
    )
    gpu_utilization = {}
    lines = result.stdout.split("\n")
    for line in lines:
        if "GPU use" in line:
            parts = line.split()
            gpu_id = int(parts[0].replace("GPU[", "").replace("]", ""))
            gpu_usage = float(parts[5]) / 100.0
            gpu_utilization[gpu_id] = gpu_usage
    return gpu_utilization


class GpuAllocator:
    """
    Class to manage GPU allocation on a cluster node with 8 GPUs
    """

    def __init__(self, n_gpus: int = 8):
        # self.available_gpus = set(
        #     GPUtil.getAvailable(
        #         order="PCI_BUS_ID",
        #         limit=100,  # big M (i.e. unreachable upper bound)
        #         maxLoad=0.5,
        #         maxMemory=0.5,
        #         includeNan=False,
        #         excludeID=[],
        #         excludeUUID=[],
        #     )
        # )

        def get_amd_gpu_memory_usage():
            result = subprocess.run(
                ["rocm-smi", "--showmemuse"],
                capture_output=True,
                text=True,
                check=False,
            )
            gpu_memory_usage = {}
            lines = result.stdout.split("\n")
            for line in lines:
                if "GPU memory use" in line:
                    parts = line.split()
                    gpu_id = int(parts[0].replace("GPU[", "").replace("]", ""))
                    memory_usage = float(parts[6]) / 100.0
                    gpu_memory_usage[gpu_id] = memory_usage
            return gpu_memory_usage

        def get_available_amd_gpus(max_load=0.5, max_memory=0.5):
            gpu_utilization = get_amd_gpu_utilization()
            gpu_memory_usage = get_amd_gpu_memory_usage()
            available_gpus = set()
            for gpu_id, current_utilization in gpu_utilization.items():
                if (
                    current_utilization <= max_load
                    and gpu_memory_usage.get(gpu_id, 1.0) <= max_memory
                ):
                    available_gpus.add(gpu_id)
            return available_gpus

        self.available_gpus = get_available_amd_gpus(max_load=0.5, max_memory=0.5)
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

    starting_i = (args.session_i - 1) * args.n_gpus * args.delta

    for gpu_i in range(args.n_gpus):
        # pylint: disable=R1732
        current_process = subprocess.Popen(
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
                "4096",
                "--fasta-path",
                args.fasta_path,
                "--output-root",
                args.output_root,
                "--model",
                args.model,
                "--detection-threshold",
                str(args.detection_threshold),
                "--detect-precursor-synthases",
                str(args.detect_precursor_synthases),
            ]
        )
        gpu_allocator.assign_process_to_gpu(current_process, gpu_i)

    gpu_allocator.wait_for_complete_finish()
