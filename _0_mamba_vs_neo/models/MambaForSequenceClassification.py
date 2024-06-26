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
        self.classifier = nn.Linear(config.hidden_size, config.num_labels, bias=False)
        self.post_init()

    def _find_last_non_pad_position(self, input_ids):
        if self.pad_token_id is not None:
            pad_mask = input_ids == self.pad_token_id
            if pad_mask.any():
                if not (~pad_mask).any():
                    raise ValueError(f"No input found, only pad tokens, input {input_ids}")
                last_non_pad_positions = pad_mask.size(1) - pad_mask.flip(dims=[1]).int().argmin(dim=1) - 1
                return last_non_pad_positions

        last_non_pad_positions = torch.full((input_ids.shape[0],), input_ids.size(1) - 1, dtype=torch.long,
                                            device=input_ids.device)
        return last_non_pad_positions

    def gpt2_find_last_non_pad_position(self, input_ids):
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
            else:
                sequence_lengths = -1
        return sequence_lengths

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
        last_non_pad_positions = self._find_last_non_pad_position(input_ids)
        last_hidden_states = hidden_states[torch.arange(hidden_states.size(0)), last_non_pad_positions]
        logits = self.classifier(last_hidden_states)
        # TODO: currently we return the last hidden state ONLY, not all layers hidden states
        # TODO: currently we dont support loss calculation
        return SequenceClassifierOutput(
            loss=None,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )
