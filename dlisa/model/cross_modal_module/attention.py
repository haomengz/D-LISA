import lightning.pytorch as pl
from torch import nn
import torch
import math


class CustomSoftmax(nn.Module):
    def __init__(self):
        super(CustomSoftmax, self).__init__()

    def forward(self, x):
        mask = x != float('-inf')
        maxes = torch.max(x, dim=-1, keepdim=True)[0]
        exps = torch.exp(x - maxes)
        exps = torch.where(mask, exps, torch.tensor(0., device=x.device))
        exps_masked_sum = exps.sum(dim=-1, keepdim=True)
        softmax_probs = exps / (exps_masked_sum + 1e-9)
        return softmax_probs


class MultiHeadAttention(pl.LightningModule):
    """
    Reference: https://github.com/zlccccc/3DVG-Transformer
    """
    def __init__(self, d_model, d_k, d_v, h, dropout):
        super().__init__()
        self.attention = ScaledDotProductAttention(d_model=d_model, d_k=d_k, d_v=d_v, h=h)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None, score=None):
        out = self.attention(queries, keys, values, attention_mask, attention_weights, score)
        out = self.dropout(out)
        out = self.layer_norm(queries + out)
        return out


class ScaledDotProductAttention(pl.LightningModule):
    """
    Reference: https://github.com/zlccccc/3DVG-Transformer
    """
    def __init__(self, d_model, d_k, d_v, h):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        """
        super().__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.softmax = CustomSoftmax()

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None, score=None):
        """
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :param score: balancing score for attention and weights (b_s, h, nq, nk).
        :return: out: (b_s, nq, d_model)
        """
        batch_size, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries)
        q = q.view(batch_size, nq, self.h, self.d_k).permute(0, 2, 1, 3)
        k = self.fc_k(keys).view(batch_size, nk, self.h, self.d_k).permute(0, 2, 3, 1)
        v = self.fc_v(values).view(batch_size, nk, self.h, self.d_v).permute(0, 2, 1, 3)

        att = torch.matmul(q, k) / math.sqrt(self.d_k)
        
        if attention_weights is not None:
            if score is not None:
                att = (1 - score) * att + score * attention_weights
            else:
                att = att + attention_weights

        if attention_mask is not None:
            att = att.masked_fill(~attention_mask, float('-inf'))
        
        att = self.softmax(att)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(batch_size, nq, self.h * self.d_v)
        out = self.fc_o(out)
        return out
