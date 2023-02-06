import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report
from typing import Type, Dict, Tuple, List
from transformers import EvalPrediction

from taskloader.taskloader_utils import TaskLoaderBase, TaskInfo, TaskLoaderOutput

class HANSLoader(TaskLoaderBase):
    def _get_metric_compute_function(self):
        def per_class_accuracy_with_names(id_to_label: Dict = None):
            def _per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray):
                classes = np.unique(y_true)
                acc_dict = {}
                for c in classes:
                    indices = (y_true == c)
                    y_true_c = y_true[indices]
                    y_pred_c = y_pred[indices]
                    class_name = id_to_label[c] if id_to_label is not None else c
                    acc_dict[f'accuracy_{class_name}'] = accuracy_score(y_true=y_true_c, y_pred=y_pred_c)
                return acc_dict

            return _per_class_accuracy

        per_class_accuracy = per_class_accuracy_with_names()

        def compute_metrics_wrap(compute_metrics_fn, preprocess_fn):
            def wrapper(pred):
                new_pred = preprocess_fn(pred)
                return compute_metrics_fn(new_pred)

            return wrapper

        def binerize_fn(pred: EvalPrediction):
            print(f'Binerizing dataset HANS')
            preds = pred.predictions.argmax(-1)
            # (Entailment, Neutral, Contradiction)

            # Neutral => Contradiction
            preds[preds == 1] = 2
            # Contradiction (2) => Contradiction (1)
            preds[preds == 2] = 1

            return EvalPrediction(predictions=preds, label_ids=pred.label_ids)

        def compute_metrics_binerized(pred):
            y_true = pred.label_ids
            y_pred = pred.predictions
            labels = np.unique(pred.label_ids)
            report = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True, digits=3,
                                            labels=labels)
            return {
                    **per_class_accuracy(y_true, y_pred),
                    'accuracy': report['accuracy'],
            }

        return compute_metrics_wrap(compute_metrics_binerized, binerize_fn)
        
    def _load_task(self) -> TaskLoaderOutput:
        hans_ds = load_dataset("hans", split="validation")

        preprocess_function = self._get_preprocessing_function(sentence1_key="premise", sentence2_key="hypothesis", label_to_id=None)

        hans_ds = hans_ds.map(preprocess_function)

        task_info = TaskInfo(
            num_labels=2, 
            label_list=hans_ds.features["label"].names, 
            is_regression=False, 
            label_to_id={'entailment': 0, 'non-entailment': 1}, 
            id_to_label={0: 'entailment', 1: 'non-entailment', 2: 'non-entailment'}
        )

        metric_compute_function = self._get_metric_compute_function()
        eval_datasets = {
            "hans-validation": hans_ds 
        }

        if self.data_args.max_eval_samples is not None:
            eval_datasets = self._slice_datasets(eval_datasets, self.data_args.max_eval_samples)

        return TaskLoaderOutput(
            task_info=task_info,
            eval_datasets=eval_datasets,
            compute_metrics=metric_compute_function
        )