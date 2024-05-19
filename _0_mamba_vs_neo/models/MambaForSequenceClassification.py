from transformers import MambaModel
from transformers.models.mamba.modeling_mamba import MambaCache
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Optional
import torch.nn as nn
import torch


class MambaForSequenceClassification(MambaModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0)

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.LongTensor] = None,
            cache_params: Optional[MambaCache] = None,
            use_cache: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = super().forward(
            input_ids,
            inputs_embeds,
            cache_params,
            use_cache,
            output_hidden_states,
            return_dict,
        )
        hidden_states = outputs.last_hidden_state
        if self.pad_token_id is not None and input_ids is not None:
            pad_mask = input_ids == self.pad_token_id
            if pad_mask.any():
                last_non_pad_positions = pad_mask.size(1) - pad_mask.flip(dims=[1]).int().argmax(dim=1) - 1

            else:
                last_non_pad_positions = torch.full((input_ids.shape[0],), pad_mask.size(1) - 1, dtype=torch.long,
                                                    device=input_ids.device)
        else:
            last_non_pad_positions = torch.full((hidden_states.shape[0],), hidden_states.shape[1] - 1, dtype=torch.long,
                                                device=hidden_states.device)
        last_hidden_states = hidden_states[torch.arange(hidden_states.size(0)), last_non_pad_positions]
        logits = self.classifier(last_hidden_states)
        return SequenceClassifierOutput(
            loss=None,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )