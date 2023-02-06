import json
from copy import deepcopy

with open("mnli_training_args_tiny.json", 'r') as f:
    data = json.load(f)

file_names = []

for seed in [42, 52, 62, 72, 82, 92, 87, 97]:
    new_data = deepcopy(data)
    new_data['output_dir'] = f"./output/tinybert/mnli/0001/{seed}/"
    new_data['seed'] = seed

    with open(f"train_mnli_{seed}.json", 'w') as f:
        f.write(json.dumps(new_data, indent=4))

    file_names.append(f"train_mnli_{seed}.json")

with open(f"run.sh", 'w') as f:
    for file_name in file_names:
        f.write('python -u training.py ' + file_name + '\n')