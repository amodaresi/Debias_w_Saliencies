import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report
from typing import Type, Dict, Tuple, List
from transformers import EvalPrediction

from taskloader.taskloader_utils import TaskLoaderBase, TaskInfo, TaskLoaderOutput

class FEVERLoader(TaskLoaderBase):
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

        def compute_metrics_with_name(pred):
            labels = pred.label_ids
            preds = pred.predictions.argmax(-1)
            acc = accuracy_score(labels, preds)
            results = {
                'accuracy': acc,
                **per_class_accuracy(labels, preds)
            }
            return results

        def mapped_compute_metrics(pred: EvalPrediction):
            def _fever_to_fever_symmetric_mapper(preds):
                # "SUPPORTS", "NOT ENOUGH INFO", "REFUTES" => "SUPPORTS", "REFUTES", "NOT ENOUGH INFO"
                preds[:, [1, 2]] = preds[:, [2, 1]]
                return preds
            pred = EvalPrediction(predictions=_fever_to_fever_symmetric_mapper(pred.predictions), label_ids=pred.label_ids)
            return compute_metrics_with_name(pred)
            
        return mapped_compute_metrics


    def _load_task(self) -> TaskLoaderOutput:
        train_ds = load_dataset("json", data_files="./taskloader/tasks/fever_data/fever.train.jsonl")['train']
        dev_ds = load_dataset("json", data_files="./taskloader/tasks/fever_data/fever.dev.jsonl")['train']
        sym_ds = load_dataset("json", data_files="./taskloader/tasks/fever_data/fever_symmetric_generated.jsonl")['train']

        train_ds = train_ds.add_column("idx", [i+1 for i in range(len(train_ds))])
        # dev_ds.add_column("idx", [i for i in range(len(dev_ds))])
        # sym_ds.add_column("idx", [i for i in range(len(sym_ds))])

        preprocess_function = self._get_preprocessing_function(sentence1_key="evidence", sentence2_key="claim", label_to_id={'SUPPORTS': 0, 'REFUTES': 1})
        dev_ds = dev_ds.map(preprocess_function, batched=True)
        print(dev_ds[0])

        preprocess_function = self._get_preprocessing_function(sentence1_key="evidence_sentence", sentence2_key="claim", label_to_id={'SUPPORTS': 0, 'REFUTES': 1})
        sym_ds = sym_ds.map(preprocess_function, batched=True)
        print(sym_ds[0])

        preprocess_function = self._get_preprocessing_function(sentence1_key="evidence", sentence2_key="claim", label_to_id={'SUPPORTS': 0, "NOT ENOUGH INFO":1, 'REFUTES': 2})
        train_ds = train_ds.map(preprocess_function, batched=True)
        print(train_ds[0])

        task_info = TaskInfo(
            num_labels=3, 
            label_list=['SUPPORTS', "NOT ENOUGH INFO", 'REFUTES'], 
            is_regression=False, 
            label_to_id={'SUPPORTS': 0, "NOT ENOUGH INFO":1, 'REFUTES': 2}, 
            id_to_label={0: 'SUPPORTS', 1: "NOT ENOUGH INFO", 2:'REFUTES'}
        )

        metric_compute_function = self._get_metric_compute_function()
        eval_datasets = {
            "FEVER-dev":  dev_ds,
            "FEVER-symV1": sym_ds
        }

        if self.data_args.train_samples_idxes:
            idxes = np.load(self.data_args.train_samples_idxes).astype(dtype=np.int)
            train_ds = train_ds.select(list(idxes))

        if self.data_args.max_train_samples is not None:
            train_ds = train_ds.select(range(self.data_args.max_train_samples))

        if self.data_args.max_eval_samples is not None:
            eval_datasets = self._slice_datasets(eval_datasets, self.data_args.max_eval_samples)

        return TaskLoaderOutput(
            task_info=task_info,
            train_dataset=train_ds,
            eval_datasets=eval_datasets,
            compute_metrics=metric_compute_function
        )