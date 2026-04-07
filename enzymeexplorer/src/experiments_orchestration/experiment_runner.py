"""This file contains an experiment runner
which is capable of gathering all the required pieces of information for a particular experiment
and consequently performing the computational experiment, i.e. instantiating, training and scoring the selected model
"""

import inspect
import logging
import os.path
import pickle

import pandas as pd  # type: ignore
from tqdm.auto import tqdm  # type: ignore
from tqdm.contrib.logging import logging_redirect_tqdm  # type: ignore

from enzymeexplorer.src import models
from enzymeexplorer.src.models.ifaces import BaseConfig, BaseModel
from enzymeexplorer.src.utils.data import get_folds_from_csv, get_tps_df
from enzymeexplorer.src.utils.project_info import (
    ExperimentInfo,
    get_config_root,
    get_output_root,
)

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

_NON_TPS_LABELS = frozenset({"Unknown", "precursor substr"})

_PRECURSOR_TYPES = frozenset({"ggpps", "fpps", "gpps", "gfpps", "hsqs", "pt"})

_DEFAULT_TYPE_COL = "Type (mono, sesq, di, …)"

_SUBSTRATE_COL = "SMILES_substrate_canonical_no_stereo"


def _remap_substrates_by_type(df: pd.DataFrame, type_col: str) -> None:
    """Override substrate labels for non-TPS protein types.

    * Precursor types (``_PRECURSOR_TYPES``) → ``"precursor substr"``
    * Unknown / negative types → ``"Unknown"``

    This prevents substrate-bearing negatives from being labelled
    ``isTPS=True`` by :func:`assign_is_tps_label`.
    """
    if type_col not in df.columns:
        return
    df.loc[
        df[type_col].isin(_PRECURSOR_TYPES), _SUBSTRATE_COL
    ] = "precursor substr"
    df.loc[df[type_col] == "Unknown", _SUBSTRATE_COL] = "Unknown"


def _normalize_fold_column(df: pd.DataFrame, col: str) -> None:
    """Ensure fold-column values use ``fold_N`` format.

    Old datasets already store ``fold_0``, ``fold_1``, etc.  New datasets
    (e.g. *EnzymeExplorer_Dataset.csv*) store bare integers ``0, 1, …``.
    This helper normalises the latter to ``fold_0, fold_1, …`` in-place so
    that downstream code can use a single format.
    """
    vals = df[col].dropna().astype(str)
    if vals.empty or vals.str.startswith("fold_").any():
        return
    mask = df[col].notna()
    df.loc[mask, col] = "fold_" + df.loc[mask, col].astype(int).astype(str)


def assign_is_tps_label(label_set: set[str]) -> set[str]:
    """Add ``isTPS`` to *label_set* when it contains at least one real TPS substrate.

    A set that only contains non-TPS sentinel values (``Unknown``,
    ``precursor substr``) is returned unchanged.  Any other substrate
    present -- even alongside ``precursor substr`` -- indicates the
    protein is a TPS and should carry the ``isTPS`` flag.
    """
    if label_set.issubset(_NON_TPS_LABELS):
        return label_set
    return label_set | {"isTPS"}


def _load_eval_dataset(config: BaseConfig) -> pd.DataFrame:
    """Load the cross-dataset evaluation CSV and rename its columns
    to match the training-dataset schema so downstream code works
    unchanged."""
    eval_df = pd.read_csv(config.eval_csv_path)
    renames: dict[str, str] = {}
    if config.eval_id_col_name:
        renames[config.eval_id_col_name] = config.id_col_name
    if config.eval_split_col_name:
        renames[config.eval_split_col_name] = config.split_col_name
    for eval_attr, cfg_attr in [
        ("eval_seq_col_name", "seq_col_name"),
        ("eval_type_col_name", "type_col_name"),
        ("eval_group_col_name", "group_column_name"),
    ]:
        eval_col = getattr(config, eval_attr, "")
        cfg_col = getattr(config, cfg_attr, None)
        if eval_col and cfg_col and eval_col != cfg_col:
            renames[eval_col] = cfg_col
    if renames:
        eval_df.rename(columns=renames, inplace=True)
    _normalize_fold_column(eval_df, config.split_col_name)
    type_col = getattr(config, "type_col_name", _DEFAULT_TYPE_COL)
    _remap_substrates_by_type(eval_df, type_col)
    logger.info(
        "Cross-dataset eval: test folds from %s (%d rows)",
        config.eval_csv_path,
        len(eval_df),
    )
    return eval_df


def run_experiment(experiment_info: ExperimentInfo, load_hyperparameters: bool = False):
    """
    This function gathers all the required pieces of information for a particular experiment
    and consequently runs the experiment, i.e. instantiating, training and scoring the selected model
    """
    # retrieve the model class
    try:
        model_class = getattr(models, experiment_info.model_type)
    except AttributeError as ex:
        raise NotImplementedError(
            f"Configured model {experiment_info.model_type} not found. The available models are "
            f"""{','.join([model_type
                  for model_type, model_class in inspect.getmembers(models, inspect.isclass)
                  if issubclass(model_class, BaseModel)])}"""
        ) from ex

    if not issubclass(model_class, BaseModel):
        raise ValueError(
            f'Model class must be a child class of "BaseModel".\n{experiment_info.model_type} is not'
        )
    logger.info(
        "The model class for %s has been successfully retrieved",
        experiment_info.model_type,
    )

    # retrieve the corresponding config
    config_class = model_class.config_class()
    if not issubclass(config_class, BaseConfig):
        raise ValueError(
            f'Config class must be a child class of "BaseConfig".\n{type(config_class)} is not'
        )
    config_path = (
        get_config_root()
        / experiment_info.model_type
        / experiment_info.model_version
        / "config.yaml"
    )
    config_dict = BaseConfig.load(config_path)
    config_dict.update({"experiment_info": experiment_info})
    config = config_class(**config_dict)
    logger.info(
        "The config for %s has been loaded and instantiated",
        experiment_info.model_type,
    )

    # accessing the configured class name, if present
    if experiment_info.class_name != "all_classes":
        config.class_name = experiment_info.class_name

    if hasattr(config, "gpu_id"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)

    if hasattr(config, "is_halo"):
        is_halo = config.is_halo
    else:
        is_halo = False

    # instantiating model
    model = model_class(config)
    if load_hyperparameters:
        try:
            per_class_optimization = model.config.per_class_optimization
        except AttributeError:
            per_class_optimization = False
        if per_class_optimization:
            raise NotImplementedError(
                "Please implement loading of outputs for per-class optimization, it wasn't needed before"
            )
        # in the future, refactor hyperparameters loading as a common routine used here and in hyperparameter tuning
        # pylint: disable=R0801
        logger.info("Looking for hyperparameters optimization results...")
        n_folds = len(
            get_folds_from_csv(
                csv_path=config.tps_cleaned_csv_path,
                split_col_name=config.split_col_name,
            )
        )
        experiment_output_folder_root = (
            get_output_root()
            / experiment_info.model_type
            / experiment_info.model_version
        )
        assert (
            experiment_output_folder_root.exists()
        ), f"Output folder {experiment_output_folder_root} for {experiment_info} does not exist"
        model_version_fold_folders = {
            x.stem for x in experiment_output_folder_root.glob("*")
        }
        if (
            len(model_version_fold_folders.intersection(set(map(str, range(n_folds)))))
            == n_folds
        ):
            logger.info("Found %d fold results for %s", n_folds, str(experiment_info))
            fold_2_root_dir = {
                f"{fold_i}": experiment_output_folder_root / f"{fold_i}"
                for fold_i in range(n_folds)
            }
        elif "all_folds" in model_version_fold_folders:
            logger.info("Found all_folds results for %s", f"{experiment_info}")
            fold_2_root_dir = {
                str(fold_i): experiment_output_folder_root / "all_folds"
                for fold_i in range(n_folds)
            }
        else:
            raise NotImplementedError(
                f"Not all fold outputs found. Please run corresponding experiments ({experiment_info}) before evaluation"
            )

    logger.info(
        "Instantiated the model %s", model.config.experiment_info.get_experiment_name()
    )

    data_df = pd.read_csv(config.tps_cleaned_csv_path)
    _normalize_fold_column(data_df, config.split_col_name)
    if not is_halo:
        type_col = getattr(config, "type_col_name", _DEFAULT_TYPE_COL)
        _remap_substrates_by_type(data_df, type_col)

    cross_dataset_eval = bool(getattr(config, "eval_csv_path", ""))
    eval_data_df = None
    eval_features_df = None
    if cross_dataset_eval:
        eval_data_df = _load_eval_dataset(config)
        eval_repr_path = getattr(config, "eval_representations_path", "")
        if eval_repr_path:
            eval_features_df = pd.read_hdf(eval_repr_path)
            eval_features_df.columns = [config.id_col_name, "Emb"]
            eval_features_df.drop_duplicates(subset=[config.id_col_name], inplace=True)

    try:
        save_trained_model = config.save_trained_model
    except AttributeError:
        save_trained_model = False
    # iterating over folds
    with logging_redirect_tqdm([logger]):
        # pylint: disable=too-many-nested-blocks
        for test_fold in tqdm(
            (
                get_folds_from_csv(
                    csv_path=config.tps_cleaned_csv_path,
                    split_col_name=config.split_col_name,
                )
                if not is_halo
                else [0]
            ),
            desc=f"Iterating over validation folds per {config.split_col_name}..",
        ):
            # selecting a single fold to run if specified
            if experiment_info.fold in {"all_folds", test_fold}:
                logger.info("Fold: %s", test_fold)
                fold_needs_resetting = experiment_info.fold == "all_folds"
                model.config.experiment_info.fold = test_fold
                if not is_halo:
                    trn_folds = [
                        f"fold_{fold_trn}"
                        for fold_trn in get_folds_from_csv(
                            csv_path=config.tps_cleaned_csv_path,
                            split_col_name=config.split_col_name,
                        )
                        if fold_trn != test_fold
                    ]
                else:
                    trn_folds = ["train"]
                trn_df = data_df[
                    data_df[config.split_col_name].isin(set(trn_folds))
                ].copy()
                if not is_halo:
                    trn_df.loc[
                        trn_df[f"{config.split_col_name}_ignore_in_eval"] == 1,
                        config.target_col_name,
                    ] = "other"
                trn_df = (
                    trn_df.groupby(config.id_col_name)[config.target_col_name]
                    .agg(set)
                    .reset_index()
                )
                trn_df[config.target_col_name] = trn_df[config.target_col_name].map(
                    assign_is_tps_label
                )

                if config.run_against_wetlab:
                    test_df_raw = get_tps_df(
                        path_to_file="data/df_wetlab_long_clean.csv",
                        path_to_sampled_negatives="data/sampled_id_2_seq_experimental.pkl",
                        id_col_name="ID",
                        remove_fragments=False,
                    )
                    test_id_column_name = "ID"
                    raw_dataset_id_colunm_name = config.id_col_name
                    trn_df[test_id_column_name] = trn_df[raw_dataset_id_colunm_name]
                    data_df[test_id_column_name] = data_df[raw_dataset_id_colunm_name]
                    model.config.id_col_name = test_id_column_name
                elif cross_dataset_eval:
                    test_df_raw = eval_data_df[
                        eval_data_df[config.split_col_name] == f"fold_{test_fold}"
                    ]
                    _eval_ignore = f"{config.split_col_name}_ignore_in_eval"
                    if not is_halo and _eval_ignore in test_df_raw.columns:
                        test_df_raw = test_df_raw.copy()
                        test_df_raw.loc[
                            test_df_raw[_eval_ignore] == 1,
                            config.target_col_name,
                        ] = "other"
                    test_id_column_name = config.id_col_name
                    model.config.id_col_name = test_id_column_name
                else:
                    test_df_raw = data_df[
                        data_df[config.split_col_name] == f"fold_{test_fold}"
                    ]
                    if not is_halo:
                        test_df_raw.loc[
                            test_df_raw[f"{config.split_col_name}_ignore_in_eval"] == 1,
                            config.target_col_name,
                        ] = "other"
                    test_id_column_name = config.id_col_name
                    model.config.id_col_name = test_id_column_name
                test_df = (
                    test_df_raw.groupby(test_id_column_name)[config.target_col_name]
                    .agg(set)
                    .reset_index()
                )
                test_df[config.target_col_name] = test_df[config.target_col_name].map(
                    assign_is_tps_label
                )

                # checking if the model requires an amino acid sequence or a group (kingdom) column
                _test_source_df = eval_data_df if cross_dataset_eval else data_df
                for optional_column_attribute in ["seq_col_name", "group_column_name"]:
                    if (
                        hasattr(config, optional_column_attribute)
                        and getattr(config, optional_column_attribute) is not None
                    ):
                        id_seq_df = data_df[
                            [
                                config.id_col_name,
                                getattr(config, optional_column_attribute),
                            ]
                        ].drop_duplicates(config.id_col_name)
                        trn_df = trn_df.merge(
                            id_seq_df,
                            on=config.id_col_name,
                        )
                        _col = getattr(config, optional_column_attribute)
                        if _col in _test_source_df.columns:
                            test_id_seq_df = test_df_raw[
                                [test_id_column_name, _col]
                            ].drop_duplicates(test_id_column_name)
                            test_df = test_df.merge(
                                test_id_seq_df,
                                on=test_id_column_name,
                            )
                logger.info(f"A number of training samples: {len(trn_df)}")
                logger.info(f"A number of testing samples: {len(test_df)}")

                # retrieving hyperparameters
                if load_hyperparameters:
                    fold_root_dir = fold_2_root_dir[test_fold]
                    logger.info(
                        "Loading hyperparameters for fold %s with root dir %s",
                        test_fold,
                        str(fold_root_dir),
                    )
                    # in the future, refactor hyperparameters loading as a common routine used here and in hyperparameter tuning
                    # pylint: disable=R0801
                    if per_class_optimization:
                        class_names = [
                            (
                                model.config.class_names
                                if not hasattr(model.config, "class_name")
                                else [model.config.class_name]
                            )
                        ]
                    else:
                        class_names = ["all_classes"]
                    for class_name in class_names:
                        if class_name not in {"Unknown", "other"}:
                            if (fold_root_dir / f"{class_name}").exists():
                                fold_class_path = fold_root_dir / f"{class_name}"
                            elif (fold_root_dir / "all_classes").exists():
                                fold_class_path = fold_root_dir / "all_classes"
                            else:
                                raise ValueError(
                                    f"No fold_class_path found for class {class_name} in folder {fold_root_dir}"
                                )
                            previous_results = list(
                                fold_class_path.glob(
                                    "*/hyperparameters_optimization/optimization_results_detailed_*.pkl"
                                )
                            )
                            if previous_results:
                                logger.info(
                                    "Found previous results for class %s: %s",
                                    class_name,
                                    previous_results,
                                )
                                with open(previous_results[0], "rb") as file:
                                    best_params, _, _ = pickle.load(file)
                                model.set_params(**best_params)
                                logger.info("Loaded previous best hyperparameters")

                # fitting the model
                model.fit(trn_df)
                logger.info(
                    "Trained model %s (%s), fold %s",
                    experiment_info.model_type,
                    experiment_info.model_version,
                    test_fold,
                )
                if save_trained_model:
                    model.save()

                # scoring the model
                _stashed_features = None
                if (
                    cross_dataset_eval
                    and eval_features_df is not None
                    and hasattr(model, "features_df")
                ):
                    _stashed_features = model.features_df
                    model.features_df = eval_features_df
                val_proba_np = model.predict_proba(
                    test_df, fold_idx=int(test_fold)
                )
                if _stashed_features is not None:
                    model.features_df = _stashed_features
                with open(
                    model.output_root / f"fold_{test_fold}_results.pkl", "wb"
                ) as file:
                    pickle.dump((val_proba_np, model.config.class_names, test_df), file)
                if fold_needs_resetting:
                    experiment_info.fold = "all_folds"
