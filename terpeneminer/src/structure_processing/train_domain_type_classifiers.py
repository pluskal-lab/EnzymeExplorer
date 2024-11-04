"""This script trains domain type classifiers and novelty detectors based on the TMScore distances between the detected domains and the known ones."""


from sklearn.ensemble import RandomForestClassifier  # type: ignore
import pickle
import logging
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

if __name__ == "__main__":

    with open('data/classifier_domain_and_plm_checkpoints.pkl', 'rb') as file:
        fold_classifiers = pickle.load(file)
    with open("data/clustering__domain_dist_based_features.pkl", "rb") as file:
        (
            feats_dom_dists,
            all_ids_list_dom,
            uniid_2_column_ids,
            domain_module_id_2_dist_matrix_index,
        ) = pickle.load(file)
    with open("data/domains_subset.pkl", "rb") as file:
        dom_subset, feat_indices_subset = pickle.load(file)
    with open('data/domain_module_id_2_domain_type.pkl', 'rb') as file:
        domain_module_id_2_domain_type = pickle.load(file)
    with open("data/precomputed_tmscores_foldseek.pkl", "rb") as file:
        regions_ids_2_tmscore = pickle.load(file)

    domain_type_classifiers = []
    fold_2_domain_type_predictions = []
    fold_2_predictions = []
    y_is_novel_test_all, y_pred_novel_all = [], []
    for FOLD in range(5):
        hits_count = 0
        miss_count = 0
        classifier = fold_classifiers[FOLD]
        new_fold_domains = [module_id for module_id in domain_module_id_2_dist_matrix_index.keys() if
                            module_id not in classifier.order_of_domain_modules]
        ref_types = {domain_module_id_2_domain_type[mod_id] for mod_id in classifier.order_of_domain_modules}
        y = np.array([domain_module_id_2_domain_type[mod_id] for mod_id in new_fold_domains])
        y_is_novel = np.array([int(dom_type not in ref_types) for dom_type in y])
        # print(Counter(y), Counter(y_is_novel))

        X_list = []

        for mod_id in new_fold_domains:
            dists_current = []
            for ref_mod_id in classifier.order_of_domain_modules:
                dom_ids = tuple(sorted([mod_id, ref_mod_id]))
                try:
                    tmscore = regions_ids_2_tmscore[dom_ids]
                    hits_count += 1
                except KeyError:
                    miss_count += 1
                    tmscore = 0
                dists_current.append(tmscore)
            X_list.append(dists_current)
        X_np = np.array(X_list)

        dom_classifier = RandomForestClassifier(500)
        dom_classifier.fit(X_np, y)
        domain_type_classifiers.append(dom_classifier)

        # novelty detector evaluation
        X_np_novel, y_novel = X_np[y_is_novel == 1], y[y_is_novel == 1]
        X_np_known, y_known = X_np[y_is_novel == 0], y[y_is_novel == 0]
        try:
            X_np_trn, X_np_test, y_trn, y_test = train_test_split(X_np_known, y_known, stratify=y_known)
            X_np_test = np.concatenate((X_np_test, X_np_novel))
            y_is_novel_test = np.concatenate((np.zeros(len(y_test)), np.ones(len(y_novel))))

            dom_classifier = RandomForestClassifier(500)
            dom_classifier.fit(X_np_trn, y_trn)
            y_pred_all = dom_classifier.predict_proba(X_np_test)
            y_pred = 1 - y_pred_all.max(axis=1)
            y_is_novel_test_all.extend(y_is_novel_test)
            y_pred_novel_all.extend(y_pred)
        except ValueError:
            logger.warning(f'Not enough un-covered domain types for fold {FOLD} (it does not influence the final results, the fold is just excluded from the novelty detection evaluation metric)')
    if sum(y_is_novel_test_all):
        logger.info(f'Novelty detection mAP: {average_precision_score(y_is_novel_test_all, y_pred_novel_all):.3f}')

    with open("data/domain_type_predictors_foldseek.pkl", "wb") as file:
        pickle.dump(domain_type_classifiers, file)
