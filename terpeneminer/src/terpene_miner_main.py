""" This is the main script running the experiments specified in the configs and/or selected via CLI or GUI """

import argparse
import logging

from terpeneminer.src.evaluation import evaluate_selected_experiments
from terpeneminer.src.evaluation.plotting import plot_selected_results
from terpeneminer.src.experiments_orchestration.experiment_runner import run_experiment
from terpeneminer.src.experiments_orchestration.experiment_selector import (
    collect_single_experiment_arguments,
    discover_experiments_from_configs,
)
from terpeneminer.src.utils.project_info import ExperimentInfo, get_config_root

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


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
    parser_run.add_argument("--load-hyperparameters", action="store_true")

    parser_eval = subparsers.add_parser("evaluate", help="Evaluate experiment(s)")
    parser_eval.set_defaults(cmd="evaluate")
    parser_eval.add_argument(
        "--classes",
        help="A list of classes to use in evaluation",
        type=str,
        nargs="+",
        default=[
            "CC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
            "CC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
            "CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
            "CC(C)=CCCC(C)=CCCC(C)=CCCC=C(C)CCC=C(C)CCC1OC1(C)C",
            "CC1(C)CCCC2(C)C1CCC(=C)C2CCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
            "CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
            "CC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O.CC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
            (
                "CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O."
                "CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O"
            ),
        ],
    )
    parser_eval.add_argument(
        "--minimal-count-to-eval",
        type=int,
        help="A minimal number of class representatives in the hold-out fold to include class during eval",
        default=3,
    )
    parser_eval.add_argument(
        "--n-folds", help="A number of folds used in CV", type=int, default=5
    )
    parser_eval.add_argument(
        "--output-filename",
        help="A file to save evaluation results",
        type=str,
        default="all_results",
    )
    parser_eval.add_argument(
        "--id-2-category-path",
        help="A path to file containing categories to be evaluated separately (e.g., kingdoms)",
        type=str,
        default=None,
    )

    parser_tune = subparsers.add_parser(
        "tune", help="Run experiments with hyper-parameter tuning"
    )
    parser_tune.set_defaults(cmd="tune")
    parser_tune.add_argument(
        "--hyperparameter-combination-i",
        type=int,
        help="An ordinal number of the hyperparameter combination to run "
        "(for automatic submission of hyperparameter search via job array in slurm)",
        default=0,
    )
    parser_tune.add_argument(
        "--classes",
        help="A list of classes to hyper-tune parameters for",
        type=str,
        nargs="+",
        default=[
            "CC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
            "CC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
            "CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
            "CC(C)=CCCC(C)=CCCC(C)=CCCC=C(C)CCC=C(C)CCC1OC1(C)C",
            "CC1(C)CCCC2(C)C1CCC(=C)C2CCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
            "CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
            "CC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O.CC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O",
            (
                "CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O."
                "CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP([O-])(=O)OP([O-])([O-])=O"
            ),
            "precursor substr",
            "isTPS",
        ],
    )
    parser_tune.add_argument(
        "--n-folds",
        type=int,
        help="A number of folds used in CV",
        default=5,
    )

    parser_vis = subparsers.add_parser("visualize", help="Run visualizations")
    parser_vis.set_defaults(cmd="visualize")
    parser_vis.add_argument(
        "--eval-output-filename",
        help="A file with saved evaluation results",
        type=str,
        default="all_results",
    )
    parser_vis.add_argument(
        "--models",
        help="A list of models for visualization",
        type=str,
        nargs="+",
        default=[
            "CLEAN__with_minor_reactions",
            "HMM__with_minor_reactions",
            "Foldseek__with_minor_reactions",
            "Blastp__with_minor_reactions",
            "PlmDomainsRandomForest__tps_esm-1v-subseq_with_minor_reactions_global_tuning_domains_subset",
        ],
    )
    parser_vis.add_argument(
        "--model-names",
        help="A list of model names to be displayed",
        type=str,
        nargs="+",
        default=["CLEAN", "HMM", "Foldseek", "Blastp", "Ours"],
    )
    parser_vis.add_argument(
        "--subset-name",
        help="A name for comparison",
        type=str,
        default=None,
    )
    parser_vis.add_argument("--plot-tps-detection", action="store_true")
    parser_vis.add_argument("--plot-boxplots-per-type", action="store_true")
    parser_vis.add_argument(
        "--type-detected",
        help="A TPS type to evaluate detection",
        type=str,
        default="isTPS",
    )
    parser_vis.add_argument(
        "--id-2-category-path",
        help="A path to file containing categories to be evaluated separately (e.g., kingdoms)",
        type=str,
        default=None,
    )
    parser_vis.add_argument("--plot-barplots-per-category", action="store_true")
    parser_vis.add_argument(
        "--category-name",
        help="A name of category to be evaluated separately (e.g., Kingdom)",
        type=str,
        default="Kingdom",
    )
    parser_vis.add_argument(
        "--categories-order",
        help="A list of category names to be displayed in the defined order",
        type=str,
        nargs="+",
        default=[
            "Bacteria",
            "Fungi",
            "Plants",
            "Animals",
            "Protists",
            "Viruses",
            "Archaea",
        ],
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
        run_experiment(experiment_info, load_hyperparameters=args.load_hyperparameters)
    else:
        all_enabled_experiments_df = discover_experiments_from_configs(config_root_path)
        for _, experiment_info_row in all_enabled_experiments_df.iterrows():
            experiment_info = ExperimentInfo(**experiment_info_row.to_dict())
            run_experiment(experiment_info, load_hyperparameters=args.load_hyperparameters)


def tune_hyperparameters(args: argparse.Namespace):
    """
    This functions picks experiments which are enabled in the configs (no .ignore suffix)
    and then it generates all possible hyperparameter tuning configuration, separately for each fold.
    :param args: parsed argparse name space
    """
    config_root_path = get_config_root()
    all_enabled_experiments_df = discover_experiments_from_configs(config_root_path)
    all_experiments_to_run = []
    for _, experiment_info_row in all_enabled_experiments_df.iterrows():
        raw_experiment_dict = experiment_info_row.to_dict()
        is_per_class_tuning = (
            "global_tuning" not in raw_experiment_dict["model_version"]
        )
        for fold_i in range(args.n_folds):
            for class_name in args.classes if is_per_class_tuning else ["all_classes"]:
                experiment_info = ExperimentInfo(**raw_experiment_dict)
                if is_per_class_tuning:
                    experiment_info.class_name = class_name
                experiment_info.fold = str(fold_i)
                all_experiments_to_run.append(experiment_info)
    all_experiments_to_run = sorted(all_experiments_to_run)
    run_experiment(all_experiments_to_run[args.hyperparameter_combination_i])


def main():
    """
    This is the main function to run the experiments, evaluate them, tune hyperparameters or visualize the results
    @return:
    """
    arguments = parse_args()
    if arguments.cmd == "run":
        run_selected_experiments(arguments)
    elif arguments.cmd == "evaluate":
        evaluate_selected_experiments(arguments)
    elif arguments.cmd == "tune":
        tune_hyperparameters(arguments)
    elif arguments.cmd == "visualize":
        plot_selected_results(arguments)


if __name__ == "__main__":
    main()
