import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, PreTrainedModel
from longformer.longformer import Longformer, LongformerConfig
from longformer.sliding_chunks import pad_to_window_size


class AvgPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.pooler = torch.nn.AdaptiveAvgPool2d((1, config.hidden_size))

    def forward(self, hidden_states):
        return self.pooler(hidden_states).view(-1, self.hidden_size)


class MaxPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.pooler = torch.nn.AdaptiveMaxPool2d((1, config.hidden_size))

    def forward(self, hidden_states):
        return self.pooler(hidden_states).view(-1, self.hidden_size)


class ClassifyHeader(nn.Module):
    """
    use averaging pooling across tokens to replace first_token_pooling
    """

    def __init__(self, config, num_class):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.pooler = AvgPooler(config)

        # self.dense = nn.Linear(config.hidden_size * 5, config.hidden_size)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # hidden_dropout_prob 0.1 for longformer
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_layer = nn.Linear(config.hidden_size, num_class)

    def forward(self, title_hidden, text_hidden, code_hidden):
        pool_hidden = self.pooler(title_hidden)
        x = self.dropout(pool_hidden)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x


class LongBert(PreTrainedModel):
    def __init__(self, config, code_bert, num_class):
        super().__init__(config)
        self.tbert  = Longformer.from_pretrained(code_bert, config=config)
        self.cls = ClassifyHeader(config, num_class=num_class)

    def forward(
            self,
            text_ids=None,
            text_attention_mask=None,
    ):
        t_hidden = self.tbert(
            text_ids, attention_mask=text_attention_mask)[0]

        logits = self.cls(title_hidden=t_hidden)
        return logits


