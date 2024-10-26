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
    with open("data/precomputed_tmscores.pkl", "rb") as file:
        regions_ids_2_tmscore = pickle.load(file)


    domain_type_classifiers = []
    novel_domain_detectors = []

    for FOLD in range(5):
        logger.info('Processing fold: %d', FOLD)
        classifier = fold_classifiers[FOLD]
        new_fold_domains = [module_id for module_id in domain_module_id_2_dist_matrix_index.keys() if module_id not in classifier.order_of_domain_modules]
        ref_types = {domain_module_id_2_domain_type[mod_id] for mod_id in classifier.order_of_domain_modules}
        y = np.array([domain_module_id_2_domain_type[mod_id] for mod_id in new_fold_domains])
        y_is_novel = np.array([int(dom_type not in ref_types) for dom_type in y])

        X_list = []

        for mod_id in new_fold_domains:
            dists_current = []
            for ref_mod_id in classifier.order_of_domain_modules:
                dom_ids = tuple(sorted([mod_id, ref_mod_id]))
                tmscore = regions_ids_2_tmscore[dom_ids]
                dists_current.append(tmscore)
            X_list.append(dists_current)
        X_np = np.array(X_list)

        # novelty detector
        X_np_trn, X_np_test, y_is_novel_trn, y_is_novel_test = train_test_split(X_np, y_is_novel, stratify=y_is_novel)
        classifier = RandomForestClassifier(500)
        classifier.fit(X_np_trn, y_is_novel_trn)
        y_pred = classifier.predict_proba(X_np_test)[:, 1]
        logger.info(f'Novelty detection mAP: {average_precision_score(y_is_novel_test, y_pred):.3f}')
        novelty_detector = RandomForestClassifier(500)
        novelty_detector.fit(X_np, y_is_novel)
        novel_domain_detectors.append(novelty_detector)

        #domain type classifier
        X_np_trn, X_np_test, y_trn, y_test = train_test_split(X_np, y, stratify=y)
        label_binarizer = MultiLabelBinarizer()
        y_trn = label_binarizer.fit_transform(
                        y_trn
                    )
        y_test = label_binarizer.transform(
                        y_test
                    )
        classifier = RandomForestClassifier(500)
        classifier.fit(X_np_trn, y_trn)
        y_pred = classifier.predict_proba(X_np_test)
        y_pred_all = np.array([y_pred_class[:, 1] for y_pred_class in y_pred]).T
        logger.info(f'Domain type classification mAP: {average_precision_score(y_test, y_pred_all):.3f}')

        classifier = RandomForestClassifier(500)
        classifier.fit(X_np, y)
        domain_type_classifiers.append(classifier)

    with open("data/domain_type_predictors.pkl", "wb") as file:
        pickle.dump([novel_domain_detectors, domain_type_classifiers], file)
