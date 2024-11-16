import lightning.pytorch as pl
import torch
import torch.nn as nn
from dlisa.model.cross_modal_module.attention import MultiHeadAttention


class MatchModule(pl.LightningModule):
    def __init__(self, feat_channel, input_channel, head, depth):
        super().__init__()
        self.head = head
        self.depth = depth - 1
        self.features_concat = nn.Sequential(
            nn.Conv1d(input_channel, feat_channel, 1),
            nn.BatchNorm1d(feat_channel),
            nn.PReLU(feat_channel),
            nn.Conv1d(feat_channel, feat_channel, 1),
        )
        self.word_self_attn = MultiHeadAttention(
                d_model=feat_channel,
                h=head,
                d_k=feat_channel // head,
                d_v=feat_channel // head,
                dropout=0.1
            )
        
        self.self_attn = nn.ModuleList(
            MultiHeadAttention(
                d_model=feat_channel,
                h=head,
                d_k=feat_channel // head,
                d_v=feat_channel // head,
                dropout=0.1
            ) for _ in range(depth)
        )
        self.cross_attn = nn.ModuleList(
            MultiHeadAttention(
                d_model=feat_channel,
                h=head,
                d_k=feat_channel // head,
                d_v=feat_channel // head,
                dropout=0.1
            ) for _ in range(depth)
        )
        self.match = nn.Sequential(
            nn.Conv1d(feat_channel, feat_channel, 1),
            nn.BatchNorm1d(feat_channel),
            nn.PReLU(),
            nn.Conv1d(feat_channel, feat_channel, 1),
            nn.BatchNorm1d(feat_channel),
            nn.PReLU(),
            nn.Conv1d(feat_channel, 1, 1)
        )
        self.score_fc = nn.Linear(2 * feat_channel, 1)

    def forward(self, data_dict, output_dict):
        batch_size, chunk_size = data_dict["ann_id"].shape[0:2]
        num_proposals = output_dict["pred_aabb_min_max_bounds"].shape[1]
        word_features = output_dict["word_features"]
        aabb_features = output_dict['aabb_features']
        pred_aabb_min_max_bounds = output_dict["pred_aabb_min_max_bounds"]
        box_centers = pred_aabb_min_max_bounds.mean(dim=2)
        score = None

        # word features
        word_features = self.word_self_attn(word_features, word_features, word_features)

        # visual features
        aabb_features = self.features_concat(aabb_features.permute(0, 2, 1)).permute(0, 2, 1)
        output_dict["aabb_features_inter"] = aabb_features
        aabb_features = aabb_features.unsqueeze(dim=1).expand(-1, chunk_size, -1, -1).flatten(start_dim=0, end_dim=1)

        # attention weight
        attention_weights = self._calculate_spatial_weight(output_dict)
        attention_weights = attention_weights.unsqueeze(dim=1).expand(-1, chunk_size, -1, -1, -1).reshape(
            batch_size * chunk_size, attention_weights.shape[1], num_proposals, num_proposals
        )
        attention_weights = torch.repeat_interleave(attention_weights, repeats=self.head//4, dim=1)
   
        # attention mask
        tmp_unsqueezed = output_dict["proposal_masks_dense"].unsqueeze(-1)
        attention_mask = tmp_unsqueezed.transpose(1, 2) * tmp_unsqueezed
        attention_mask = attention_mask.unsqueeze(dim=1).expand(-1, self.head, -1, -1)
        attention_mask = attention_mask.unsqueeze(dim=1).expand(-1, chunk_size, -1, -1, -1).reshape(
            batch_size * chunk_size, attention_mask.shape[1], num_proposals, num_proposals
        )
        
        # get spatial scores
        sentence_feature = output_dict['sentence_features']
        sentence_feature = sentence_feature.unsqueeze(dim=1).unsqueeze(dim=1).expand(-1, num_proposals, num_proposals, -1)
        obj = aabb_features.unsqueeze(dim=2).expand(-1, -1, num_proposals, -1)
        concate_feature = torch.cat((sentence_feature, obj), dim=-1)
        score = torch.sigmoid(self.score_fc(concate_feature)).squeeze(-1)
        score = score.unsqueeze(dim=1).expand(-1, self.head, -1, -1)

        for i in range(0, self.depth + 1):
            aabb_features = self.self_attn[i](aabb_features, aabb_features, aabb_features, attention_mask=attention_mask, attention_weights=attention_weights, score=score)
            aabb_features = self.cross_attn[i](aabb_features, word_features, word_features)
        
        # match
        aabb_features = aabb_features.permute(0, 2, 1).contiguous()
        output_dict["pred_aabb_scores"] = self.match(aabb_features).flatten(start_dim=0, end_dim=1)

    def _calculate_spatial_weight(self, output_dict):
        objects_center = output_dict["pred_aabb_min_max_bounds"].mean(dim=2)
        num_proposals = objects_center.shape[1]
        center_a = objects_center.unsqueeze(dim=1).repeat(1, num_proposals, 1, 1)
        center_b = objects_center.unsqueeze(dim=2).repeat(1, 1, num_proposals, 1)
        dist = (center_a - center_b).pow(2)
        dist = torch.sqrt(dist.sum(dim=-1)).unsqueeze(dim=1)
        dist_weights = 1 / (dist + 1e-2)

        dist_weights += torch.finfo(torch.float32).eps
        norm = dist_weights.sum(dim=2, keepdim=True)
        dist_weights = dist_weights / norm
        zeros = torch.zeros_like(dist_weights)
        dist_weights = torch.cat([dist_weights, -dist, zeros, zeros], dim=1).detach()
        return dist_weights

