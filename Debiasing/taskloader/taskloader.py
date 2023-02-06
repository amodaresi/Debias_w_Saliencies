from transformers import TrainingArguments, PreTrainedTokenizer

from taskloader.tasks.glue import GLUELoader, MNLILoader
from taskloader.tasks.hans import HANSLoader
from taskloader.tasks.fever import FEVERLoader

task_loaders = {
    "mnli": MNLILoader,
    "sst2": GLUELoader,
    "sts-b": GLUELoader,
    "wnli": GLUELoader,
    "qnli": GLUELoader,
    "rte": GLUELoader,
    "qqp": GLUELoader,
    "cola": GLUELoader,
    "hans": HANSLoader,
    "fever": FEVERLoader
}
tasks = list(task_loaders.keys())

from src.transformers.utils import DataTrainingArguments

def load_task (
    task_name: str, 
    cache_dir: str, 
    data_args: DataTrainingArguments, 
    training_args: TrainingArguments, 
    tokenizer: PreTrainedTokenizer, 
    max_length: int
):
    if task_name not in task_loaders.keys():
        raise ValueError("Unknown task, you should pick one in " + ",".join(task_loaders.keys()))
    tk = task_loaders[task_name](task_name, cache_dir, data_args, training_args, tokenizer, max_length)
    return tk._load_task()