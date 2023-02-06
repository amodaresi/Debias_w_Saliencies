import os
import re
from os.path import abspath
import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

import torch
from torch import nn

from transformers import AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert import BertConfig, BertForSequenceClassification, BertPreTrainedModel, BertModel
from src.losses.poe import ProductOfExpertsLoss

class BertWithWeakLearnerConfig(BertConfig):
    EXPERT_POLICIES = {'freeze', 'e2e'}
    LOSS_FUNCTIONS = {'poe', 'dfl'}

    def __init__(self,
                 weak_model_name_or_path='google/bert_uncased_L-2_H-128_A-2',
                 weak_model_sals_path='',
                 loss_fn='poe',
                 poe_alpha=1.0,
                 poe_beta=1.0,
                 poe_mix_type=1,
                 dfl_gamma=2.0,
                 poe_kappa=1e+7,
                 *args, **kwargs
                 ):

        assert loss_fn in BertWithWeakLearnerConfig.LOSS_FUNCTIONS, \
            f'The loss functions must be one of {", ".join(BertWithWeakLearnerConfig.LOSS_FUNCTIONS)},' \
            f'but got {loss_fn}'

        super().__init__(*args, **kwargs)
        self.weak_model_name_or_path = weak_model_name_or_path
        self.weak_model_sals_path = weak_model_sals_path
        self.loss_fn = loss_fn
        self.poe_alpha = poe_alpha
        self.poe_kappa = poe_kappa
        self.poe_beta = poe_beta
        self.poe_mix_type = poe_mix_type
        self.dfl_gamma = dfl_gamma

def _select_loss(config: BertWithWeakLearnerConfig):
    name = config.loss_fn
    if name == 'poe':
        return ProductOfExpertsLoss(config.poe_alpha, config.poe_beta, config.poe_mix_type)
    elif name == 'dfl':
        return DebiasedFocalLoss(gamma=config.dfl_gamma)

    raise ValueError(f'Unknown loss function ${name}')

class BertWithWeakLearner(BertPreTrainedModel):
    config_class = BertWithWeakLearnerConfig

    def __init__(self, config: BertWithWeakLearnerConfig):
        super().__init__(config)
        # self.weak_learner = weak_learner
        # self.bert = bert
        self.config = config
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.bert = BertModel(config)
        self.loss_fn = _select_loss(config)
        self.bias_loss_fn = nn.CrossEntropyLoss()
        self.have_correlations = False
        self.weak_logits = self._load_df(config.weak_model_name_or_path)
        if config.weak_model_sals_path != '':
            self.weak_sals = np.load(config.weak_model_sals_path)
            self.have_correlations = True

    @staticmethod
    def _load_df(path):
        print(f'Loading probabilities from CSV file ({abspath(path)})...')
        return pd.read_csv(path).set_index('id')

    def _forward_weak(self, inputs_with_labels, ids):
        if self.weak_logits is not None:
            # Return logits from loaded file
            # (N, C)
            inferred_device = inputs_with_labels['input_ids'].device

            return torch.from_numpy(np.array(self.weak_logits.loc[ids.cpu()-1])).to(inferred_device)

        with torch.no_grad():
            _, weak_logits = self.weak_model(**inputs_with_labels, return_dict=False)
        return weak_logits

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, idx=None, **kwargs):
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            # 'labels': labels
        }

        if self.training:
            _, pooled_output, hidden_states = self.bert(**inputs, return_dict=False, output_hidden_states=True)
            main_logits = self.classifier(self.dropout(pooled_output))
            weak_logits = self._forward_weak(inputs, idx)

            if self.have_correlations:
                hidden_states[0].retain_grad()

                l_sum = torch.gather(main_logits, 1, labels.unsqueeze(-1)).sum()
                l_sum.backward(retain_graph=True)

                inputXgradient = hidden_states[0].grad * hidden_states[0]
                saliencies = torch.norm(inputXgradient, dim=-1)

        else:
            _, pooled_output = self.bert(**inputs, return_dict=False)
            main_logits = self.classifier(self.dropout(pooled_output))
            weak_logits = None
        
        if self.have_correlations and self.training:
            inferred_device = input_ids.device
            length = saliencies.size()[1]
            weak_sals = torch.from_numpy(self.weak_sals[idx.cpu()-1]).to(inferred_device)
            valid_main_sals = saliencies * (weak_sals[:, :length] > 0)
            cos = torch.nn.CosineSimilarity(dim=-1)
            corrs = cos(valid_main_sals, weak_sals[:, :length]).detach()
            self.zero_grad()

            loss, logits = self.loss_fn(main_logits, weak_logits, labels, corrs)
        else:
            loss, logits = self.loss_fn(main_logits, weak_logits, labels)
            
        return SequenceClassifierOutput(logits=main_logits, loss=loss)