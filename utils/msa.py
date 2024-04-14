"""This module contains msa-related utils"""

import os
from typing import Optional


def get_fasta_seqs(seqs: list[str], ids: Optional[list[str]] = None) -> str:
    """
    The function returns a string corresponding to a fasta file
    :param seqs: a list of amino acid sequences
    :param ids: a list of corresponding ids
    :return: a fasta string
    """
    if ids is None:
        full_entries = [
            f">{i}\n{''.join(tps_seq.split())}" for i, tps_seq in enumerate(seqs)
        ]
    else:
        full_entries = [
            f">{id}\n{''.join(tps_seq.split())}" for id, tps_seq in zip(ids, seqs)
        ]
    return "\n".join(full_entries)


def generate_msa_mafft(
    seqs: list[str],
    ids: Optional[list[str]] = None,
    output_name: str = "_msa.fasta",
    num_workers: int = 26,
):
    """
    The function generates MSA and stores it into a file
    :param seqs: a list of amino acid sequences
    :param ids: a list of corresponding ids
    :param output_name: an MSA output file name
    """
    fasta_str = get_fasta_seqs(seqs, ids)
    with open("_temp_mafft.fasta", "w", encoding="utf-8") as file:
        file.writelines(fasta_str.replace("'", "").replace('"', ""))
    msa_return_value = os.system(
        f"mafft --thread {num_workers} --auto _temp_mafft.fasta > {output_name}"
    )
    if msa_return_value != 0:
        raise ChildProcessError("MSA creation failed!")
    os.remove("_temp_mafft.fasta")


def read_fasta_seqs(msa_file: str) -> tuple[list[str], list[str]]:
    """
    The function reads aligned sequences from the specified fasta file of MSA
    :param msa_file: the path to the MSA fasta file
    :return: a list of protein ids and a list of the corresponding aligned sequences
    """
    with open(msa_file, "r", encoding="utf-8") as file:
        file_lines = file.readlines()
    seq_ids = []
    aligned_seqs = []
    next_seq = None
    for line in file_lines:
        if ">" in line:
            if next_seq is not None:
                aligned_seqs.append("".join(next_seq))
            seq_ids.append(line.replace(">", "").strip())
            next_seq = []
        else:
            next_seq.append(line.strip())
    aligned_seqs.append("".join(next_seq))
    return seq_ids, aligned_seqs


def get_mappings(full_aligned_seq: str) -> tuple[dict, dict]:
    """
    A helping function processing an aligned sequence to get mapping
    between position indices of the original sequence and the aligned one
    :param full_aligned_seq: the aligned amino acid sequences
    :return: a dictionary with mapping of the original (unaligned) sequence indices to the aligned sequences indices,
            and another dictionary with a reverse mapping
    """
    orig_2_msa = {}
    msa_2_orig = {}

    orig_i = 0
    msa_i = 0
    for char in full_aligned_seq:
        if char == " ":
            orig_i += 1
        elif char == "-":
            msa_i += 1
        else:
            orig_2_msa[orig_i] = msa_i
            msa_2_orig[msa_i] = orig_i
            orig_i += 1
            msa_i += 1
    return orig_2_msa, msa_2_orig
