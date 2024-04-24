""" This is the main script running the experiments specified in the configs and/or selected via CLI or GUI """

import argparse

from src.evaluation import evaluate_selected_experiments
from src.experiments_orchestration.experiment_runner import run_experiment
from src.experiments_orchestration.experiment_selector import (
    collect_single_experiment_arguments,
    discover_experiments_from_configs,
)
from src.utils.project_info import ExperimentInfo, get_config_root


def parse_args() -> argparse.Namespace:
    """
    This function parses arguments
    :return: current argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="An entry point for Terpene synthases substrate prediction"
    )
    parser.add_argument("--select-single-experiment", action="store_true")
    subparsers = parser.add_subparsers()
    parser_run = subparsers.add_parser("run", help="Run experiment(s)")
    parser_run.set_defaults(cmd="run")
    parser_eval = subparsers.add_parser("evaluate", help="Evaluate experiment(s)")
    parser_eval.set_defaults(cmd="evaluate")
    parser_eval.add_argument(
        "--classes",
        help="A list of classes to use in evaluation",
        type=str,
        nargs="+",
        default=[
            "CC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O.CC(=C)CCOP(O)(=O)OP(O)(O)=O",
            "CC1(C)CCCC2(C)C1CCC(=C)C2CCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
            "C(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O.C(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
            "CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O.CC(=C)CCOP(O)(=O)OP(O)(O)=O",
            "CC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
            "CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
            "CC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
            "CC(C)=CCCC(C)=CCCC(C)=CCCC=C(C)CCC=C(C)CCC1OC1(C)C",
            "CC(C)=CCOP([O-])(=O)OP([O-])([O-])=O.CC(=C)CCOP(O)(=O)OP(O)(O)=O",
            "CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
            "CC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O.CC(=C)CCOP(O)(=O)OP(O)(O)=O",
        ],
    )
    parser_eval.add_argument(
        "--minimal-count-to-eval",
        type=int,
        help="A minimal number of class representatives in the hold-out fold to include class during eval",
        default=3,
    )
    args = parser.parse_args()
    return args


def run_selected_experiments(args: argparse.Namespace):
    """
    This functions runs the experiments which are enabled in the configs (no .ignore suffix) or the selected experiment
    :param args: parsed argparse name space
    """

    config_root_path = get_config_root()
    if args.select_single_experiment:
        experiment_kwargs = collect_single_experiment_arguments(config_root_path)
        experiment_info = ExperimentInfo(**experiment_kwargs)
        run_experiment(experiment_info)
    else:
        all_enabled_experiments_df = discover_experiments_from_configs(config_root_path)
        for _, experiment_info_row in all_enabled_experiments_df.iterrows():
            experiment_info = ExperimentInfo(**experiment_info_row.to_dict())
            run_experiment(experiment_info)


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.cmd == "run":
        run_selected_experiments(arguments)
    elif arguments.cmd == "evaluate":
        evaluate_selected_experiments(arguments)
