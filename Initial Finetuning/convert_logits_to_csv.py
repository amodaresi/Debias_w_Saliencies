import pandas as pd
import torch
import numpy as np

def convert_preds(y, location):
    df = pd.DataFrame(y)
    df['id'] = np.arange(len(df))
    df.columns = ['label_0','label_1','label_2','id']
    df.to_csv(location,
          columns=['id','label_0','label_1','label_2'], index=False)

for seed in [42, 52, 62, 72, 82, 92, 87, 97]:
    print("Loading logits...")
    all_logits = torch.load(f"./output/tinybert/mnli/0001/{seed}/training_dynamics_after_epoch/epoch2_after_epoch_train_logits_[392702, 3].pt")
    rearranged_all_logits = torch.zeros_like(all_logits)
    idxs = torch.load(f"./output/tinybert/mnli/0001/{seed}/training_dynamics_after_epoch/epoch2_after_epoch_train_idxs_[392702].pt")
    rearranged_all_logits[idxs] = all_logits
    convert_preds(rearranged_all_logits.numpy(), f"./output/tinybert/mnli/0001/{seed}/{seed}_predictions.csv")