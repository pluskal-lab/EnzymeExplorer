"""This script trains domain type classifiers and novelty detectors based on the TMScore distances between the detected domains and the known ones."""


from sklearn.ensemble import RandomForestClassifier  # type: ignore
import pickle
import logging
import numpy as np  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.metrics import average_precision_score  # type: ignore
from sklearn.preprocessing import MultiLabelBinarizer  # type: ignore
from sklearn.metrics import precision_recall_curve  # type: ignore
import matplotlib.pyplot as plt # type: ignore
from sklearn.metrics import confusion_matrix, classification_report  # type: ignore
import seaborn as sns  # type: ignore
from sklearn.metrics import precision_score, recall_score, f1_score  # type: ignore

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def plot_performance_metrics_bars(y_true_per_fold: list[np.ndarray],
                                y_pred_per_fold: list[np.ndarray],
                                classes_per_fold: list[np.ndarray]) -> None:
    """
    Create bar plots for precision, recall, and F1-score for each domain type.
    """
    # Get unique classes across all folds
    all_classes = sorted(set().union(*[set(classes) for classes in classes_per_fold]))
    n_classes = len(all_classes)
    logger.info(f"Total number of unique classes: {n_classes}")
    
    # Create class index mapping
    class_to_idx = {class_name: idx for idx, class_name in enumerate(all_classes)}
    
    # Calculate metrics for each fold
    metrics_per_fold = []
    for fold_idx, (y_true_classes, y_pred, classes) in enumerate(zip(y_true_per_fold, y_pred_per_fold, classes_per_fold)):
        logger.info(f"Processing fold {fold_idx}")
        logger.info(f"y_true shape: {y_true_classes.shape}, y_pred shape: {y_pred.shape}")
            
        fold_metrics = {}
        
        # Convert predictions to class predictions
        y_pred_classes = classes[np.argmax(y_pred, axis=1)]
        
        # Calculate metrics for each class
        for class_name in all_classes:
            try:
                true_mask = (y_true_classes == class_name)
                pred_mask = (y_pred_classes == class_name)
                
                # Only calculate metrics if the class appears in the true labels
                if np.any(true_mask):
                    precision = precision_score(true_mask, pred_mask)
                    recall = recall_score(true_mask, pred_mask)
                    f1 = f1_score(true_mask, pred_mask)
                else:
                    precision = recall = f1 = 0.0
                    
                fold_metrics[class_name] = {
                    'Precision': precision,
                    'Recall': recall,
                    'F1-score': f1
                }
            except Exception as e:
                logger.error(f"Error calculating metrics for class {class_name} in fold {fold_idx}: {str(e)}")
                fold_metrics[class_name] = {
                    'Precision': 0.0,
                    'Recall': 0.0,
                    'F1-score': 0.0
                }
                
        metrics_per_fold.append(fold_metrics)
    
    if not metrics_per_fold:
        logger.error("No valid folds to process!")
        return
        
    # Calculate mean and std for each metric
    mean_metrics = {class_name: {'Precision': [], 'Recall': [], 'F1-score': []} 
                   for class_name in all_classes}
    std_metrics = {class_name: {'Precision': [], 'Recall': [], 'F1-score': []} 
                  for class_name in all_classes}
    
    for class_name in all_classes:
        for metric in ['Precision', 'Recall', 'F1-score']:
            values = [fold[class_name][metric] for fold in metrics_per_fold]
            mean_metrics[class_name][metric] = np.mean(values)
            std_metrics[class_name][metric] = np.std(values)
    
    # Calculate macro average F1 score across all classes
    macro_f1 = np.mean([mean_metrics[class_name]['F1-score'] for class_name in all_classes])
    
    # Create the plot
    _, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    x = np.arange(n_classes)
    width = 0.8
    
    metrics = ['Precision', 'Recall', 'F1-score']
    axes = [ax1, ax2, ax3]
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    for ax, metric, color in zip(axes, metrics, colors):
        means = [mean_metrics[class_name][metric] for class_name in all_classes]
        stds = [std_metrics[class_name][metric] for class_name in all_classes]
        
        bars = ax.bar(x, means, width, yerr=stds, capsize=5, 
                     color=color, alpha=0.7, 
                     error_kw={'elinewidth': 2, 'capthick': 2})
        
        ax.set_ylabel(metric, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(all_classes, rotation=45, ha='right')
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim(0, 1.1)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom')
    
    plt.suptitle(f'Domain Type Classification Performance Metrics\n(Detection Threshold = 0.5, Macro F1 = {macro_f1:.2f})', 
                fontsize=16)
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(f'outputs/domain_type_performance_metrics.{ext}', 
                    bbox_inches="tight", dpi=300)
    plt.close()

if __name__ == "__main__":

    with open('data/classifier_domain_and_plm_checkpoints.pkl', 'rb') as file:
        fold_classifiers = pickle.load(file)
    with open("data/clustering__domain_dist_based_features_foldseek.pkl", "rb") as file:
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

    y_novelty_test_per_fold = []
    y_novelty_pred_test_per_fold = []
    classes_per_fold = []
    y_test_per_fold = []
    y_pred_test_per_fold = []

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
        ref_types = {domain_module_id_2_domain_type.get(mod_id, 'negative') for mod_id in classifier.order_of_domain_modules}
        y = np.array([domain_module_id_2_domain_type.get(mod_id, 'negative') for mod_id in new_fold_domains])
        y_is_novel = np.array([int(dom_type not in ref_types and dom_type != 'negative') for dom_type in y])

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
            X_np_trn, X_np_test, y_trn, y_test = train_test_split(X_np_known, y_known, stratify=y_known, random_state=42)
            X_np_test_for_novelty = np.concatenate((X_np_test, X_np_novel))
            y_is_novel_test = np.concatenate((np.zeros(len(y_test)), np.ones(len(y_novel))))

            dom_classifier = RandomForestClassifier(500)
            dom_classifier.fit(X_np_trn, y_trn)
            classes_per_fold.append(dom_classifier.classes_)
            y_pred = dom_classifier.predict_proba(X_np_test)
            y_pred_all = dom_classifier.predict_proba(X_np_test_for_novelty)
            y_pred_test_per_fold.append(y_pred)
            logger.info(f'# of predictions: {len(y_pred)}')
            logger.info(f'# of gt cases: {len(y_test)}')
            y_test_per_fold.append(y_test)
            y_pred = 1 - y_pred_all.max(axis=1)
            y_is_novel_test_all.append(y_is_novel_test)
            y_pred_novel_all.append(y_pred)
        except ValueError:
            logger.warning(f'Not enough un-covered domain types for fold {FOLD} (it does not influence the final results, the fold is just excluded from the novelty detection evaluation metric)')

        y_novelty_test_per_fold.append(y)
        y_novelty_pred_test_per_fold.append(dom_classifier.predict_proba(X_np))

    if y_is_novel_test_all:
        # Concatenate predictions and ground truth from all folds
        y_true_concat = np.concatenate(y_is_novel_test_all)
        y_pred_concat = np.concatenate(y_pred_novel_all)
        
        # Calculate single PR curve
        precision, recall, _ = precision_recall_curve(y_true_concat, y_pred_concat)
        ap_score = average_precision_score(y_true_concat, y_pred_concat)
        
        # Create the plot
        plt.figure(figsize=(10, 8))
        
        # Plot PR curve
        plt.plot(recall, precision, 
                color='blue', 
                label=f'AP={ap_score:.3f}')
        
        # Customize the plot
        plt.xlabel("Recall", fontsize=15)
        plt.ylabel("Precision", fontsize=15)
        plt.title("Domain Novelty Detection\nPrecision-Recall Curve", fontsize=20)
        plt.legend(fontsize=12, labelspacing=0.9)
        
        # Set axis limits
        plt.xlim([-0.01, 1.05])
        plt.ylim([-0.01, 1.05])
        plt.grid(True, alpha=0.3)
        
        # Save the plots
        for ext in ['png', 'pdf']:
            plt.savefig(f'outputs/novelty_detection_pr_curve.{ext}', bbox_inches="tight", dpi=300)
        plt.close()
        
        logger.info(f'Novelty detection AP: {ap_score:.3f}')

    # Add performance metrics bar plots
    plot_performance_metrics_bars(y_test_per_fold, y_pred_test_per_fold, classes_per_fold)

    with open("data/domain_type_predictors_foldseek.pkl", "wb") as file:
        pickle.dump(domain_type_classifiers, file)
