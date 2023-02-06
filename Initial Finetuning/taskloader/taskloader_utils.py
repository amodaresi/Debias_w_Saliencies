from os import stat
import numpy as np
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from datasets.arrow_dataset import Dataset
from transformers import EvalPrediction

@dataclass
class TaskInfo:
    num_labels: int
    label_list: List
    is_regression: bool
    label_to_id: Optional[Dict] = field(
        default=None
    )
    id_to_label: Optional[Dict] = field(
        default=None
    )


@dataclass
class TaskLoaderOutput:
    task_info: TaskInfo
    train_dataset: Optional[Dataset] = field(
        default=None
    )
    eval_datasets: Optional[Dict[str, Dataset]] = field(
        default=None
    )
    predict_datasets: Optional[Dict[str, Dataset]] = field(
        default=None
    )
    compute_metrics: Optional[Callable] = field(
        default=None
    )


class TaskLoaderBase:
    def __init__(self, task_name, cache_dir, data_args, training_args, tokenizer, max_length):
        self.task_name = task_name
        self.data_args = data_args
        self.training_args = training_args
        self.cache_dir = cache_dir
        self.tokenizer = tokenizer
        self.max_length = max_length

        if self.data_args.pad_to_max_length:
            self.padding = "max_length"
        else:
            # We will pad later, dynamically at batch creation, to the max sequence length in each batch
            self.padding = False

    @abstractmethod
    def _load_task(self) -> TaskLoaderOutput:
        raise NotImplementedError("Please run a task-specific taskloader")

    def _get_preprocessing_function(
        self, 
        sentence1_key: str, 
        sentence2_key: str = None, 
        label_to_id: dict = None):

        def preprocess_function(examples):
            # Tokenize the texts
            args = (
                (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
            )
            result = self.tokenizer(*args, padding=self.padding, max_length=self.max_length, truncation=True)

            # Map labels to IDs (not necessary for GLUE tasks)
            if label_to_id is not None and "label" in examples:
                result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
            return result
        return preprocess_function


    def _get_metric_compute_function(self, metric, is_regression: bool):
        def compute_metrics(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
            if self.data_args.task_name is not None:
                result = metric.compute(predictions=preds, references=p.label_ids)
                if len(result) > 1:
                    result["combined_score"] = np.mean(list(result.values())).item()
                return result
            elif is_regression:
                return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
            else:
                return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}
        return compute_metrics

    def _slice_datasets(self, datasets, size):
        for k in datasets.keys():
            datasets[k] = datasets[k].select(range(size))
        return datasets