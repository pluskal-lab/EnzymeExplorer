""" This is the main script running the experiments specified in the configs and/or selected via CLI or GUI """

# --- Dynamic-linker fix (must run before ANY heavy import) --------------
# The ``predict`` subcommand imports PyMOL, whose ``libvtkm_cont-1.8.so.1``
# requires ``GLIBCXX_3.4.30`` — present in the conda env's libstdc++ but
# not in the system copy on RHEL-9 hosts. The dynamic linker reads
# ``LD_LIBRARY_PATH`` at process start, so setting it mid-process would
# not help; the only reliable fix is to prepend ``$CONDA_PREFIX/lib`` and
# re-exec once, before numpy/pandas / anything else pulls in the older
# libstdc++. A sentinel env var guards against exec loops. This is a
# no-op outside conda and a no-op when LD_LIBRARY_PATH already starts
# with the conda lib dir.
import os as _os_bootstrap  # noqa: E402
import sys as _sys_bootstrap  # noqa: E402

_CONDA_PREFIX = _os_bootstrap.environ.get("CONDA_PREFIX")
_ALREADY_REEXECED = _os_bootstrap.environ.get("_ENZYME_EXPLORER_LDFIX") == "1"
if _CONDA_PREFIX and not _ALREADY_REEXECED:
    _CONDA_LIB = _os_bootstrap.path.join(_CONDA_PREFIX, "lib")
    _existing = _os_bootstrap.environ.get("LD_LIBRARY_PATH", "")
    if not _existing.startswith(_CONDA_LIB):
        _env = dict(_os_bootstrap.environ)
        _env["LD_LIBRARY_PATH"] = (
            _CONDA_LIB + (":" + _existing if _existing else "")
        )
        _env["_ENZYME_EXPLORER_LDFIX"] = "1"
        _os_bootstrap.execvpe(_sys_bootstrap.executable,
                              [_sys_bootstrap.executable] + _sys_bootstrap.argv,
                              _env)

import argparse
import logging

from enzymeexplorer.src.evaluation import cli as eval_cli
from enzymeexplorer.src.prediction import cli as predict_cli
from enzymeexplorer.src.experiments_orchestration.experiment_runner import run_experiment
from enzymeexplorer.src.experiments_orchestration.experiment_selector import (
    collect_single_experiment_arguments,
    discover_experiments_from_configs,
)
from enzymeexplorer.src.utils.project_info import ExperimentInfo, get_config_root

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
    parser_run.add_argument("--model", type=str, default=None)
    parser_run.add_argument("--model-version", type=str, default=None)

    eval_cli.add_evaluate_subparser(subparsers)
    eval_cli.add_calibrate_subparser(subparsers)
    eval_cli.add_visualize_subparser(subparsers)
    predict_cli.add_predict_subparser(subparsers)

    # ``detect_domains`` is a passthrough — argparse.REMAINDER is intercepted
    # in :func:`main` before this parser runs, so we register the subcommand
    # here only for --help visibility.
    subparsers.add_parser(
        "detect_domains",
        help=(
            "Detect TPS-family domains in a directory of PDB/CIF "
            "structures. All args after ``detect_domains`` are passed "
            "verbatim to the configargparse CLI in "
            "``enzymeexplorer.src.structure_processing.domain_detections`` "
            "(``-c/--config`` for a YAML config; see "
            "``enzymeexplorer/configs/*_domain_detection_config.yaml``)."
        ),
        add_help=False,
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

    args = parser.parse_args()
    return args


def run_selected_experiments(args: argparse.Namespace):
    """
    This functions runs the experiments which are enabled in the configs (no .ignore suffix) or the selected experiment
    :param args: parsed argparse name space
    """

    config_root_path = get_config_root()
    if args.select_single_experiment:
        if args.model is None or args.model_version is None:
            experiment_kwargs = collect_single_experiment_arguments(config_root_path)
        else:
            experiment_kwargs = {
                "model_type": args.model,
                "model_version": args.model_version,
            }
        experiment_info = ExperimentInfo(**experiment_kwargs)
        run_experiment(experiment_info, load_hyperparameters=args.load_hyperparameters)
    else:
        all_enabled_experiments_df = discover_experiments_from_configs(config_root_path)
        for _, experiment_info_row in all_enabled_experiments_df.iterrows():
            experiment_info = ExperimentInfo(**experiment_info_row.to_dict())
            run_experiment(
                experiment_info, load_hyperparameters=args.load_hyperparameters
            )


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
    # Ensure INFO-level logs from every submodule (evaluation, bootstrap,
    # calibration, categorical_bootstrap, plotting) surface in the console.
    # Some deep modules already call basicConfig at import time; if none of
    # them get imported (e.g. pure `evaluate --config .../calibration.yaml`)
    # we still need a root handler for the timing + step logs added to
    # ``evaluation.cli`` to appear.
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    # ``detect_domains`` is a full passthrough — the target CLI uses
    # configargparse (its own -h, its own --config semantics), so we
    # bypass argparse entirely and forward argv verbatim. Runs before
    # ``parse_args`` so top-level ``-h/--help`` never intercepts a
    # subcommand-level help request.
    import sys as _sys
    if len(_sys.argv) >= 2 and _sys.argv[1] == "detect_domains":
        from enzymeexplorer.src.structure_processing import (
            domain_detections as _dd,
        )
        _sys.argv = [
            "enzyme_explorer_main detect_domains", *_sys.argv[2:],
        ]
        _dd.main()
        return
    arguments = parse_args()
    if arguments.cmd == "run":
        run_selected_experiments(arguments)
    elif arguments.cmd == "evaluate":
        eval_cli.run_evaluate(arguments)
    elif arguments.cmd == "calibrate":
        eval_cli.run_calibrate(arguments)
    elif arguments.cmd == "tune":
        tune_hyperparameters(arguments)
    elif arguments.cmd == "visualize":
        eval_cli.run_visualize(arguments)
    elif arguments.cmd == "predict":
        predict_cli.run_predict(arguments)


if __name__ == "__main__":
    main()
