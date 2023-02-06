import torch
from torch import nn


class ProductOfExpertsLoss(nn.Module):
    """Implements the Product of Experts loss (PoE) for a single expert (teacher) and main model"""
    def __init__(self, alpha=1.0, beta=1.0, mcorr_mix_type=0):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.alpha = alpha
        self.beta = beta
        self.mcorr_mix_type = mcorr_mix_type

    def __repr__(self):
        return f'{self.__class__.__name__}(alpha={self.alpha})'

    def forward(self, inputs, expert_input, target, corrs=None):
        """Computes the PoE cross-entropy loss for a batch size N and number of classes C.
        During training, we use the expert_input for the PoE loss. During evaluation, the expert logits are
        ignored and standard cross-entropy loss is used

        :param inputs: The main model raw outputs (before applying Softmax), of shape (N, C)
        :param expert_input: The expert model raw outputs (before applying Softmax), of shape (N, C)
        :param target: Target labels, of shape (N,)
        :return:
            loss: The calculated loss tensor
            inputs: The normalized logits. If this is called during evaluation, the original input logits are returned
        """
        orig_loss = self.ce_loss(inputs, target)
        if self.training:
            assert expert_input is not None, "During training, expert_input must not be None"
            if corrs is not None:
                poe_logits = self.log_softmax(inputs) + self.log_softmax(expert_input)
                poe_losses = self.ce_loss(
                        poe_logits,
                        target
                )
                if self.mcorr_mix_type == 0:
                    m_corrs = (corrs + 1e-10) ** (torch.exp(-self.beta * poe_losses))
                    losses = poe_losses * m_corrs + self.alpha * (1-m_corrs) * orig_loss
                elif self.mcorr_mix_type == 1:
                    m_corrs = 1.0 - (1.0 - corrs) * (torch.exp(-self.beta * poe_losses))
                    losses = poe_losses * m_corrs + self.alpha * (1-m_corrs) * orig_loss
                elif self.mcorr_mix_type == 2:
                    m_corrs = (corrs + 1e-10) ** (torch.exp(-self.beta * poe_losses))
                    losses = (poe_losses + self.alpha * orig_loss) * m_corrs + (1-m_corrs) * orig_loss
                elif self.mcorr_mix_type == 3:
                    m_corrs = 1.0 - (1.0 - corrs) * (torch.exp(-self.beta * poe_losses))
                    losses = (poe_losses + self.alpha * orig_loss) * m_corrs + (1-m_corrs) * orig_loss
            else:
                poe_logits = self.log_softmax(inputs) + self.log_softmax(expert_input)
                poe_losses = self.ce_loss(
                        poe_logits,
                        target
                )
                losses = poe_losses + (self.alpha) * orig_loss
            return torch.mean(losses), poe_logits
        else:
            # During evaluation, we use standard cross-entropy for the main model only
            return torch.mean(orig_loss), inputs
