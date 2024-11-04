"""This script predicts domain types and novelty based on the TMScore distances between the detected domains and the known ones."""
from collections import defaultdict

from sklearn.ensemble import RandomForestClassifier  # type: ignore
import pickle
import logging
import argparse
import numpy as np  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.metrics import average_precision_score  # type: ignore
from sklearn.preprocessing import MultiLabelBinarizer  # type: ignore

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def parse_args() -> argparse.Namespace:
    """
    This function parses arguments
    :return: current argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="A script to detect novel and known domains"
    )
    parser.add_argument(
        "--tps-classifiers-path",
        type=str,
        default="data/classifier_domain_and_plm_checkpoints.pkl",
    )
    parser.add_argument(
        "--domain-classifiers-path",
        type=str,
        default="data/domain_type_predictors.pkl",
    )
    parser.add_argument("--path-to-domain-comparisons", type=str)
    parser.add_argument("--id", type=str)
    parser.add_argument("--output-path", type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    with open(args.tps_classifiers_path, "rb") as file:
        tps_classifiers = pickle.load(file)
    with open(args.domain_classifiers_path, "rb") as file:
        domain_type_classifiers = pickle.load(file)
    with open(args.path_to_domain_comparisons, "rb") as file:
        comparison_results = pickle.load(file)
    comparison_results = comparison_results[args.id]

    domain_id_2_predictions = {}
    for new_protein_domain_id, domain_comparisons_result in comparison_results.items():
        known_domain_id_2_tmscore = dict(domain_comparisons_result)
        is_novel_predictions = []
        domain_type_2_pred_values = defaultdict(list)
        for FOLD in range(5):
            logger.info('Processing fold: %d', FOLD)
            classifier = tps_classifiers[FOLD]
            fold_domains_order = classifier.order_of_domain_modules
            feat_vector = np.zeros(len(fold_domains_order))
            for i, known_domain_id in enumerate(fold_domains_order):
                if known_domain_id in known_domain_id_2_tmscore:
                    feat_vector[i] = known_domain_id_2_tmscore[known_domain_id]
            X_np = np.array(feat_vector).reshape(1, -1)

            classifier = domain_type_classifiers[FOLD]
            domain_type_pred = classifier.predict_proba(X_np)
            for class_name, pred_val in zip(classifier.classes_, domain_type_pred[0]):
                domain_type_2_pred_values[class_name].append(pred_val)
        domain_type_2_pred = {dom_type: np.mean(vals) for dom_type, vals in domain_type_2_pred_values.items()}
        max_pred = -float('inf')
        gen_type_2_pred = {}
        for type_preds in domain_type_2_pred.values():
            max_pred = max(max_pred, np.max(type_preds))
        domain_type_2_pred.update({"novel": 1 - max_pred})
        domain_id_2_predictions[new_protein_domain_id] = domain_type_2_pred

    with open(args.output_path, "wb") as file:
        pickle.dump(domain_id_2_predictions, file)




