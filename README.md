# Guide the Learner: Controlling Product of Experts Debiasing Method Based on Token Attribution Similarities

_Accepted as a conference paper for EACL 2023_

<!-- [Arxiv]() -->

> **Abstract**: Several proposals have been put forward in recent years for improving out-of-distribution (OOD) performance through mitigating dataset biases. A popular workaround is to train a robust model by re-weighting training examples based on a secondary biased model. Here, the underlying assumption is that the biased model resorts to shortcut features. Hence, those training examples that are correctly predicted by the biased model are flagged as being biased and are down-weighted during the training of the main model. However, assessing the importance of an instance merely based on the  predictions of the biased model may be too naive. It is possible that the prediction of the main model can be derived from another decision-making process that is distinct from the behavior of the biased model. To circumvent this, we introduce a fine-tuning strategy that incorporates the similarity between the main and biased model attribution scores in a Product of Experts (PoE) loss function to further improve OOD performance. With experiments conducted on natural language inference and fact verification benchmarks, we show that our method improves OOD results while maintaining in-distribution (ID) performance.

## Requirements

To install the required dependencies for this repo you can use `requirements.txt`:

```shell script
pip install -r requirements.txt
```

## Initial Finetuning
In the "Initial Finetuning" folder:
1. Create run arguments for training with the following command:
```
python write_jsons_mnli_tiny.py
```
2. Start the training:
```
run.sh
```
3. Convert logits to csv:
```
python convert_logits_to_csv.py
```
4. Compute and store saliencies:
```
python compute_saliencies_tiny.py
```

## Debias Finetuning
In the "Debiasing" folder:
1. Set hyperparameters in `training_args.json`
2. Run training with the following command:
##### *Note: Change the `weak_model_name_or_path` and `weak_model_sals_path` parameters based on the logits and saliencies files that are produced in the previous phase.*
```
python training.py training_args.json
```


3. (For HANS) Evaluate HANS with the following command:
```
python training.py hans_evaluation_args.json
```

## Credits:
This repository is influenced by the [Huggingface trainer](https://huggingface.co/docs/transformers/main_classes/trainer) and [bias-probing](https://github.com/technion-cs-nlp/bias-probing) implementations.