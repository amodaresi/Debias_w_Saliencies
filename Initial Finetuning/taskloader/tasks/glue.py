import numpy as np
from datasets import load_dataset, load_metric
from taskloader.taskloader_utils import TaskLoaderBase, TaskInfo, TaskLoaderOutput

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


class GLUELoader(TaskLoaderBase):
    def _glue_preloader(self):
        # Load dataset
        raw_datasets = load_dataset("glue", self.task_name, cache_dir=self.cache_dir)

        # Build dataset info
        is_regression = self.task_name == "stsb"
        label_to_id, id_to_label = None, None
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
            label_to_id = {v: i for i, v in enumerate(label_list)}
            id_to_label = {id: label for label, id in label_to_id.items()}
        else:
            num_labels = 1

        task_info = TaskInfo(
            num_labels=num_labels, 
            label_list=label_list, 
            is_regression=is_regression, 
            label_to_id=label_to_id, 
            id_to_label=id_to_label
        )

        # Preprocessing
        sentence1_key, sentence2_key = task_to_keys[self.task_name]

        preprocess_function = self._get_preprocessing_function(sentence1_key=sentence1_key, sentence2_key=sentence2_key, label_to_id=None)

        with self.training_args.main_process_first(desc="dataset map pre-processing"):
            raw_datasets = raw_datasets.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )

        train_dataset = raw_datasets["train"]
        if self.data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(self.data_args.max_train_samples))

        #### REMAINING SPLITS IN EXCLUSIVE FUNCTIONS ONLY!

        # Load metric
        metric = load_metric("glue", self.task_name)
        metric_compute_function = self._get_metric_compute_function(metric=metric, is_regression=is_regression)

        return raw_datasets, train_dataset, metric_compute_function, task_info
    
    def _load_task(self) -> TaskLoaderOutput:
        raw_datasets, train_dataset, metric_compute_function, task_info = self._glue_preloader()
        eval_datasets = {
            "validation": raw_datasets["validation"]
        }
        predict_datasets = {
            "test": raw_datasets["test"]
        }

        if self.data_args.max_eval_samples is not None:
            eval_datasets = self._slice_datasets(eval_datasets, self.data_args.max_eval_samples)

        if self.data_args.max_predict_samples is not None:
            predict_datasets = self._slice_datasets(predict_datasets, self.data_args.max_predict_samples)

        return TaskLoaderOutput(
            train_dataset=train_dataset,
            task_info=task_info,
            eval_datasets=eval_datasets,
            predict_datasets=predict_datasets,
            compute_metrics=metric_compute_function
        )


class MNLILoader(GLUELoader):
    def _load_task(self) -> TaskLoaderOutput:
        raw_datasets, train_dataset, metric_compute_function, task_info = self._glue_preloader()
        eval_datasets = {
            "validation-m": raw_datasets["validation_matched"],
            "validation-mm": raw_datasets["validation_mismatched"]
        }
        predict_datasets = {
            "test-m": raw_datasets["test_matched"],
            "test-mm": raw_datasets["test_mismatched"]
        }

        if self.data_args.train_samples_idxes:
            idxes = np.load(self.data_args.train_samples_idxes).astype(dtype=np.int)
            train_dataset = train_dataset.select(list(idxes))

        if self.data_args.max_eval_samples:
            eval_datasets = self._slice_datasets(eval_datasets, self.data_args.max_eval_samples)

        if self.data_args.max_predict_samples:
            predict_datasets = self._slice_datasets(predict_datasets, self.data_args.max_predict_samples)

        return TaskLoaderOutput(
            train_dataset=train_dataset,
            task_info=task_info,
            eval_datasets=eval_datasets,
            predict_datasets=predict_datasets,
            compute_metrics=metric_compute_function
        )