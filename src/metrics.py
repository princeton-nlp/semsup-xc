"""
Metrics for multi-label text classification
"""

import numpy as np
from scipy.special import expit
import itertools
import copy
import torch
import pickle
import os
import json
# Metrics
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    precision_score,
    recall_score,
    label_ranking_average_precision_score,
    coverage_error
)
import time


def multilabel_metrics(data_args, id2label, label2id, fbr, training_args = None):
    """
    Metrics function used for multilabel classification.

    :fbr : A dict containing global thresholds to be used for selecting a class.
    We use global thresholds because we want to handle unseen classes,
    for which the threshold is not known in advance.
    """

    func_call_counts = 0

    def compute_metrics(p):

        # global func_call_counts
        # Save the predictions, maintaining a global counter
        if training_args is not None and training_args.local_rank <= 0:
            preds_fol = os.path.join(training_args.output_dir, 'predictions')
            os.makedirs(preds_fol, exist_ok = True)
            func_call_counts = time.time() #np.random.randn()
            pickle.dump(p.predictions, open(os.path.join(preds_fol, f'predictions_{(func_call_counts+1) * training_args.eval_steps }.pkl'), 'wb'))
            pickle.dump(p.label_ids, open(os.path.join(preds_fol, f'label_ids_{(func_call_counts+1) * training_args.eval_steps }.pkl'), 'wb'))


        # func_call_counts += 1
        # print(func_call_counts)

        # Collect the logits
        print('Here we go!')
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions

        # Compute the logistic sigmoid
        preds = expit(preds)

        # METRIC 0: Compute P@1, P@3, P@5
        if 'precision' not in fbr.keys():
            top_values = [1,3,5]
            # denoms = {k:0 for k in top_values}
            tops = {k:0 for k in top_values}
            i = 0
            preds_preck = None
            print(p.label_ids)
            for i, (logit, label) in enumerate(zip(p.predictions[0], p.label_ids)):
                logit = torch.from_numpy(logit)
                label = torch.from_numpy(label)
                _, indexes = torch.topk(logit.float(), k = max(top_values))
                for val in top_values:
                    if preds_preck is None:
                        tops[val] += len([x for x in indexes[:val] if label[x]!=0])
                    else:
                        tops[val] += len([x for x in preds_preck[i][indexes[:val]] if label[x]!=0])
                    # denoms[val] += min(val, label.nonzero().shape[0])
            
            precisions_at_k =  {k:v/((i+1)*k) for k,v in tops.items()}
            # rprecisions_at_k =  {k:v/denoms[v] for k,v in tops.items()}
        
            print('Evaluation Result: precision@{} = {}'.format(top_values, precisions_at_k))
            # print('Evaluation Result: rprecision@{} = {}'.format(top_values, rprecisions_at_k))
        
        # p.predictions = p.predictions[0]
        # p.label_ids = p.label_ids[0]

        # METRIC 1: Compute accuracy
        if 'accuracy' not in fbr.keys():
            performance = {}
            for threshold in np.arange(0.1, 1, 0.1):
                accuracy_preds = np.where(preds > threshold, 1, 0)
                performance[threshold] = np.sum(p.label_ids == accuracy_preds) / accuracy_preds.size * 100
            # Choose the best threshold
            best_threshold = max(performance, key=performance.get)
            fbr['accuracy'] = best_threshold
            accuracy = performance[best_threshold]
        else:
            accuracy_preds = np.where(preds > fbr['accuracy'], 1, 0)
            accuracy = np.sum(p.label_ids == accuracy_preds) / accuracy_preds.size * 100

        # METRIC 2: Compute the subset accuracy
        if 'subset_accuracy' not in fbr.keys():
            performance = {}
            for threshold in np.arange(0.1, 1, 0.1):
                subset_accuracy_preds = np.where(preds > threshold, 1, 0)
                performance[threshold] = accuracy_score(p.label_ids, subset_accuracy_preds)
            # Choose the best threshold
            best_threshold = max(performance, key=performance.get)
            fbr['subset_accuracy'] = best_threshold
            subset_accuracy = performance[best_threshold]
        else:
            subset_accuracy_preds = np.where(preds > fbr['subset_accuracy'], 1, 0)
            subset_accuracy = accuracy_score(p.label_ids, subset_accuracy_preds)

        # METRIC 3: Macro F-1
        if 'macro_f1' not in fbr.keys():
            performance = {}
            for threshold in np.arange(0.1, 1, 0.1):
                macro_f1_preds = np.where(preds > threshold, 1, 0)
                performance[threshold] = f1_score(p.label_ids, macro_f1_preds, average='macro')
            # Choose the best threshold
            best_threshold = max(performance, key=performance.get)
            fbr['macro_f1'] = best_threshold
            macro_f1 = performance[best_threshold]
        else:
            macro_f1_preds = np.where(preds > fbr['macro_f1'], 1, 0)
            macro_f1 = f1_score(p.label_ids, macro_f1_preds, average='macro')

        # METRIC 4: Micro F-1
        if 'micro_f1' not in fbr.keys():
            performance = {}
            for threshold in np.arange(0.1, 1, 0.1):
                micro_f1_preds = np.where(preds > threshold, 1, 0)
                performance[threshold] = f1_score(p.label_ids, micro_f1_preds, average='micro')
            # Choose the best threshold
            best_threshold = max(performance, key=performance.get)
            fbr['micro_f1'] = best_threshold
            micro_f1 = performance[best_threshold]
        else:
            micro_f1_preds = np.where(preds > fbr['micro_f1'], 1, 0)
            micro_f1 = f1_score(p.label_ids, micro_f1_preds, average='micro')

        # Multi-label classification report
        # Optimized for Micro F-1
        try:
            report = classification_report(p.label_ids, micro_f1_preds, target_names=[id2label[i] for i in range(len(id2label))])
            print('Classification Report: \n', report)
        except:
            report = classification_report(p.label_ids, micro_f1_preds)
            print('Classification Report: \n', report)
        return_dict  = {
            "accuracy": accuracy,
            "subset_accuracy": subset_accuracy,
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
            # "hier_micro_f1": hier_micro_f1,
            "fbr": fbr
        }
        for k in precisions_at_k:
            return_dict[f'P@{k}'] = precisions_at_k[k]
    
        if training_args is not None and training_args.local_rank <= 0:
            try:
                metrics_fol = os.path.join(training_args.output_dir, 'metrics')
                os.makedirs(metrics_fol, exist_ok = True)
                json.dump(return_dict, open(os.path.join(metrics_fol, f'metrics_{(func_call_counts+1) * training_args.eval_steps }.json'), 'w'), indent = 2)
            except Exception as e:
                print('Error in metrics', e)

        return return_dict

    return compute_metrics