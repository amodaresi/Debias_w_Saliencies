import torch
import numpy as np

import datasets
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding
)

MAX_LENGTH = 128
BATCH_SIZE = 128
TASK_NAME = "mnli"

tokenizer = AutoTokenizer.from_pretrained(
        "google/bert_uncased_L-2_H-128_A-2"
)

def _get_preprocessing_function(
    sentence1_key: str, 
    sentence2_key: str = None, 
    label_to_id: dict = None):

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=False, max_length=MAX_LENGTH, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result
    return preprocess_function

original_ds = datasets.load_dataset("glue", "mnli", split="train")

preprocess_function = _get_preprocessing_function(sentence1_key="premise", sentence2_key="hypothesis")
train_ds = original_ds.map(preprocess_function, batched=True)

train_ds.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
collator = DataCollatorWithPadding(tokenizer, True, MAX_LENGTH, return_tensors="pt")
dataloader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, collate_fn=collator)
dataset_size = len(train_ds)
steps = int(np.ceil(dataset_size / BATCH_SIZE))
num_labels = 3

for seed in [52, 62, 72, 82, 87, 92, 97]:
    base_loc = f"./output/tinybert/mnli/0001/{seed}"
    
    model = AutoModelForSequenceClassification.from_pretrained(base_loc)
    model.to(torch.device("cuda:0"))
    model.eval()

    all_sals = np.zeros((dataset_size, MAX_LENGTH))
    it = iter(dataloader)

    for i in tqdm(range(steps)):
        batch = next(it)
        batch = {k: v.to(torch.device('cuda:0')) for k, v in batch.items()}
        inputs = {
                'input_ids': batch['input_ids'].to(torch.device('cuda:0')),
                'attention_mask': batch['attention_mask'].to(torch.device('cuda:0')),
                'token_type_ids': batch['token_type_ids'].to(torch.device('cuda:0')),
                # 'labels': labels
        }
        labels = batch['labels'].to(torch.device('cuda:0'))
        output = model(**batch, output_hidden_states=True)
        output.hidden_states[0].retain_grad()
        logits = output.logits
        l_sum = torch.gather(logits, 1, labels.unsqueeze(-1)).sum()
        l_sum.backward()
        
        inputXgradient = output.hidden_states[0].grad * output.hidden_states[0]
        saliencies = torch.norm(inputXgradient, dim=-1).detach()

        length = saliencies.size()[1]

        model.zero_grad()
        all_sals[i*BATCH_SIZE:(i+1)*BATCH_SIZE, :length] = saliencies.cpu().numpy()

    np.save(base_loc + "/sals.npy", all_sals)
    del model