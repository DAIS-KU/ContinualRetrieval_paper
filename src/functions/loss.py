import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .similarities import calculate_S_qd_regl_logits


class InfoNCETermLoss(nn.Module):
    def __init__(self):
        super(InfoNCETermLoss, self).__init__()

    # (batch_size, 254, 768)  # (batch_size, positive_k + negative_k, 254, 768)
    def forward(self, q_embeddings, d_embeddings):
        # print(f"q_embeddings:{q_embeddings.shape}, d_embeddings:{d_embeddings.shape}")
        logits = calculate_S_qd_regl_logits(q_embeddings, d_embeddings)
        labels = torch.zeros(
            d_embeddings.size(0), d_embeddings.size(1), device=q_embeddings.device
        )
        labels[:, :1] = 1.0
        labels = labels.long()
        # print(f"logits:{logits.shape}/{logits}, labels:{labels.shape}")
        loss = F.cross_entropy(logits, labels)
        return loss


class InfoNCELoss(nn.Module):
    def __init__(self):
        super(InfoNCELoss, self).__init__()

    def forward(self, query_emb, positive_emb, negative_emb):
        # query_emb: (batch_size, embedding_dim)
        # positive_emb: (batch_size, positive_k, embedding_dim)
        # negative_emb: (batch_size, negative_k, embedding_dim)
        # 내적 계산
        pos_sim = torch.matmul(
            query_emb.unsqueeze(1), positive_emb.transpose(-1, -2)
        ).squeeze(
            1
        )  # (batch_size, positive_k)
        neg_sim = torch.matmul(
            query_emb.unsqueeze(1), negative_emb.transpose(-1, -2)
        ).squeeze(
            1
        )  # (batch_size, negative_k)
        # 모든 유사도를 결합 후 temperature 적용
        logits = torch.cat(
            (pos_sim, neg_sim), dim=1
        )  # (batch_size, positive_k + negative_k)
        # 정답 레이블 생성 (첫 번째 positive 샘플을 정답으로 설정)
        labels = torch.zeros(
            query_emb.size(0), dtype=torch.long, device=query_emb.device
        )
        # Cross-entropy loss 계산
        # print(f"logits:{logits.shape}/{logits}, labels:{labels.shape}")
        loss = F.cross_entropy(logits, labels)
        return loss


class SimpleContrastiveLoss:
    def __call__(
        self, x: Tensor, y: Tensor, target: Tensor = None, reduction: str = "mean"
    ):
        if target is None:
            target_per_qry = y.size(0) // x.size(0)
            target = torch.arange(
                0,
                x.size(0) * target_per_qry,
                target_per_qry,
                device=x.device,
                dtype=torch.long,
            )
        logits = torch.matmul(x, y.transpose(0, 1))
        return F.cross_entropy(logits, target, reduction=reduction)
