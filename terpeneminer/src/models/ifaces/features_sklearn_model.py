"""Module with interfaces for scikit-learn-compatible predictive models
built on top of numerical protein representations (features)"""
import logging
from typing import Optional

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import sklearn.base  # type: ignore
from sklearn.multioutput import MultiOutputClassifier  # type: ignore
from sklearn.preprocessing import MultiLabelBinarizer  # type: ignore

from .config_baseclasses import SklearnBaseConfig
from .model_baseclass import BaseModel

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class FeaturesSklearnModel(BaseModel):
    """Interfaces for scikit-learn-compatible predictive models
    built on top of numerical protein representations (features)"""

    def __init__(self, config: SklearnBaseConfig):
        super().__init__(config=config)
        try:
            self.per_class_optimization = self.config.per_class_optimization
            if self.per_class_optimization is None:
                self.per_class_optimization = False
            if self.per_class_optimization:
                try:
                    self.per_class_with_multilabel_regularization = (
                        self.config.per_class_with_multilabel_regularization  # type: ignore
                    )
                except AttributeError:
                    self.per_class_with_multilabel_regularization = 0
        except AttributeError:
            self.per_class_optimization = False
        if self.per_class_optimization:
            self.class_2_classifier: dict = {}
        self.classifier: sklearn.base.BaseEstimator
        self.classes_ = None
        self.features_df = None

    def fit_core(self, train_df: pd.DataFrame, class_name: str = None):
        logger.info("Started fitting the model %s", self.__class__)
        assert isinstance(
            self.config, SklearnBaseConfig
        ), "EmbsSklearnModel model expects config of type SklearnBaseConfig"
        train_df_neg_all = train_df[
            train_df[self.config.target_col_name].map(
                lambda classes: self.config.neg_val in classes
            )
        ]
        negs_proportion = len(train_df_neg_all) / len(train_df)
        if negs_proportion > self.config.max_train_negs_proportion:
            positive_count = train_df[self.config.id_col_name].nunique() * (
                1 - negs_proportion
            )
            required_negs_count = (
                positive_count / (1 - self.config.max_train_negs_proportion)
                - positive_count
            )
            train_df_pos = train_df[
                train_df[self.config.target_col_name].map(
                    lambda classes: self.config.neg_val not in classes
                )
            ]
            train_df_neg = train_df_neg_all.sample(int(required_negs_count))
            train_df = pd.concat((train_df_pos, train_df_neg)).sample(frac=1.0)

        assert (
            self.features_df is not None
        ), "self.features_df has not been initialized!"
        logger.info("In fit(), features DF shape is: %d x %d", *self.features_df.shape)
        logger.info(
            "In fit(), features dimension is: %d",
            self.features_df["Emb"].values[0].shape[0],
        )
        train_data = train_df.merge(self.features_df, on=self.config.id_col_name)

        if not self.per_class_optimization:
            try:
                requires_multioutputwrapper_for_multilabel = (
                    self.config.requires_multioutputwrapper_for_multilabel  # type: ignore
                )
            except AttributeError:
                requires_multioutputwrapper_for_multilabel = False
            if requires_multioutputwrapper_for_multilabel:
                self.classifier = MultiOutputClassifier(
                    self.classifier_class(**self.get_model_specific_params())
                )
            else:
                self.classifier = self.classifier_class(
                    **self.get_model_specific_params()
                )
            label_binarizer = MultiLabelBinarizer(classes=self.config.class_names)
            logger.info("Preparing multi-labels...")
            target = label_binarizer.fit_transform(
                train_data[self.config.target_col_name].values
            )
            logger.info("Fitting the model...")

            self.classifier.fit(
                np.stack(train_data["Emb"].values),
                target,
            )
            self.config.class_names = label_binarizer.classes_
        else:
            assert (
                class_name is not None
            ), "For per-class fitting the class name must be provided"
            model_specific_params = list(
                self.get_model_specific_params(class_name=class_name).items()
            )
            model_params = {
                param_name.replace(f"{class_name}_", ""): param_val
                for param_name, param_val in model_specific_params
            }
            is_class_specific_param_present = False
            for param_name, _ in model_specific_params:
                if class_name in param_name:
                    is_class_specific_param_present = True
                    break
            if is_class_specific_param_present:
                classifier = self.classifier_class(**model_params)
                if self.per_class_with_multilabel_regularization:
                    label_binarizer = MultiLabelBinarizer(
                        classes=self.config.class_names
                    )
                    logger.info(
                        "Preparing multi-labels during per class optimization..."
                    )
                    target = label_binarizer.fit_transform(
                        train_data[self.config.target_col_name].values
                    )
                    logger.info(
                        "Fitting the model (multi-label) for class %s...", class_name
                    )

                    classifier.fit(
                        np.stack(train_data["Emb"].values),
                        target,
                    )
                    self.class_2_classifier[class_name] = classifier
                    self.config.class_names = label_binarizer.classes_
                else:
                    y_binary = train_data[self.config.target_col_name].map(
                        lambda x: int(class_name in x)
                    )
                    if sum(y_binary):
                        logger.info(
                            "Fitting the model (binary) for class %s...", class_name
                        )
                        classifier.fit(
                            np.stack(train_data["Emb"].values),
                            y_binary,
                        )
                        self.class_2_classifier[class_name] = classifier
            else:
                raise ValueError(
                    f"During per-class optimization class {class_name} had no parameters specified."
                )

    def predict_proba(
        self,
        val_df: pd.DataFrame | np.ndarray,
        selected_class_name: Optional[str] = None,
    ) -> np.ndarray:
        if isinstance(val_df, pd.DataFrame):  # local validation
            assert isinstance(self.features_df, pd.DataFrame)
            test_df = val_df.merge(
                self.features_df, on=self.config.id_col_name, copy=False, how="left"
            ).set_index(self.config.id_col_name)
            logger.info(
                "In predict_proba(), features DF shape is: %d x %d",
                *self.features_df.shape,
            )
            logger.info(
                "In predict_proba(), features dimension is: %d",
                self.features_df["Emb"].values[0].shape[0],
            )

            average_emb = np.stack(self.features_df["Emb"].values).mean(axis=0)
            test_df["Emb"] = test_df["Emb"].map(
                lambda x: x if isinstance(x, np.ndarray) else average_emb
            )
            test_df = test_df.loc[val_df[self.config.id_col_name]]
            test_embs_np = np.stack(test_df["Emb"].values)
        elif isinstance(val_df, np.ndarray):  # test-time prediction
            test_embs_np = val_df
        else:
            raise NotImplementedError(
                f"Got {type(val_df)} as input to predict_proba function, "
                f"but only pd.DataFrame | np.ndarray are supported"
            )
        try:
            per_class_optimization = self.config.per_class_optimization
        except AttributeError:
            per_class_optimization = False
        val_proba_np = np.zeros((len(val_df), len(self.config.class_names)))
        if not per_class_optimization:
            logger.info(
                "Global model %s predicts proba for %d samples (%d features each)",
                str(self.classifier),
                *test_embs_np.shape,
            )
            y_pred_proba = self.classifier.predict_proba(test_embs_np)
            for class_i in range(len(self.config.class_names)):
                val_proba_np[:, class_i] = (
                    y_pred_proba[class_i][:, -1]
                    if isinstance(y_pred_proba, list)
                    else y_pred_proba[:, class_i]
                )
        else:
            for class_i, class_name in enumerate(self.config.class_names):
                if (
                    selected_class_name is None or class_name == selected_class_name
                ) and class_name in self.class_2_classifier:
                    logger.info("Predicting proba for class %s...", class_name)
                    y_pred_proba = self.class_2_classifier[class_name].predict_proba(
                        test_embs_np
                    )
                    val_proba_np[:, class_i] = (
                        y_pred_proba[class_i][:, -1]
                        if isinstance(y_pred_proba, list)
                        else y_pred_proba[:, 1]
                    )
        return val_proba_np
