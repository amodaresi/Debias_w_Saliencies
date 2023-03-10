import os
import sys
import random
import logging
import shutil
import numpy as np
from taskloader.taskloader import load_task

from src.transformers.utils import ModelArguments, DataTrainingArguments
from src.modeling.bert_w_weak import BertWithWeakLearnerConfig, BertWithWeakLearner

import datasets
import transformers

from transformers import (
    AutoConfig,
    AutoTokenizer,
    PretrainedConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    default_data_collator,
    set_seed,
    HfArgumentParser
)

# Add custom trainer
from src.transformers.trainer import Trainer
from src.transformers.training_args import TrainingArguments

from transformers.trainer_utils import get_last_checkpoint

logger = logging.getLogger(__name__)

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    # Parsing arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    tl_outputs = load_task(data_args.task_name, model_args.cache_dir, data_args, training_args, tokenizer, max_seq_length)
    train_dataset, eval_datasets, predict_datasets, compute_metrics, task_info = tl_outputs.train_dataset, tl_outputs.eval_datasets, tl_outputs.predict_datasets, tl_outputs.compute_metrics, tl_outputs.task_info
    num_labels, is_regression, label_list, label_to_id, id_to_label = task_info.num_labels, task_info.is_regression, task_info.label_list, task_info.label_to_id, task_info.id_to_label

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = BertWithWeakLearnerConfig.from_pretrained(model_args.model_name_or_path,
                                                               num_labels=num_labels,
                                                               cache_dir=model_args.cache_dir,
                                                               poe_alpha=training_args.poe_alpha,
                                                               poe_beta=training_args.poe_beta,
                                                               poe_mix_type=training_args.poe_mix_type,
                                                               dfl_gamma=training_args.dfl_gamma,
                                                               loss_fn=training_args.loss_fn,
                                                               weak_model_name_or_path=model_args.weak_model_name_or_path,
                                                               weak_model_sals_path=model_args.weak_model_sals_path)
    model = BertWithWeakLearner.from_pretrained(model_args.model_name_or_path, config=config, cache_dir=model_args.cache_dir)

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        # if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
        #     label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        # else:
        #     logger.warning(
        #         "Your model seems to have been trained with labels, but they don't match the dataset: ",
        #         f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
        #         "\nIgnoring the model labels as a result.",
        #     )
    elif data_args.task_name is None and not is_regression:
        label_to_id = label_to_id

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif data_args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}


    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    first_eval_dataset_key = list(eval_datasets.keys())[0]
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_datasets[first_eval_dataset_key] if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        for task, eval_dataset in eval_datasets.items():
            metrics = trainer.evaluate(eval_dataset=eval_dataset, metric_key_prefix=task)

            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            )
            metrics[f"{task}_samples"] = min(max_eval_samples, len(eval_dataset))
            trainer.log_metrics(f"{task}", metrics)
            trainer.save_metrics(f"{task}", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        for task, predict_dataset in predict_datasets.items():
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            predict_dataset = predict_dataset.remove_columns("label")
            predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
            predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

            output_predict_file = os.path.join(training_args.output_dir, f"predict_results_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    logger.info(f"***** Predict results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}
    if data_args.task_name is not None:
        kwargs["language"] = "en"
        kwargs["dataset_tags"] = ""
        kwargs["dataset_args"] = data_args.task_name
        kwargs["dataset"] = f"{data_args.task_name.upper()}"

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        shutil.move(os.path.abspath(sys.argv[1]), os.path.join(training_args.output_dir, f"run_arguments.json"))

if __name__ == '__main__':
    main()