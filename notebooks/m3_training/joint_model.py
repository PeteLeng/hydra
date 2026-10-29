"""A shared BERT encoder, two heads, and explicit losses. Inference takes no labels."""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertModel

from snips_data import IGNORE_SLOT, Labels


@dataclass
class Predictions:
    """B = batch size; L = token length, including special tokens and padding."""

    intent_logits: Tensor      # float [B, num_intents]; raw scores per example
    slot_logits: Tensor        # float [B, L, num_slots]; raw scores per token


@dataclass
class Losses:
    """Scalar tensors retain their autograd graph; [] denotes a scalar shape."""

    intent: Tensor             # float []; mean intent cross-entropy
    slot: Tensor               # float []; mean supervised-token cross-entropy
    total: Tensor              # float []; intent + slot_weight * slot


class JointModel(nn.Module):
    def __init__(self, encoder: BertModel, labels: Labels, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = encoder
        self.labels = labels
        self.dropout = nn.Dropout(dropout)
        self.intent_head = nn.Linear(encoder.config.hidden_size, len(labels.intent_names))
        self.slot_head = nn.Linear(encoder.config.hidden_size, len(labels.slot_names))

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Predictions:
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        # BERT's pooled output is dense+tanh on [CLS]; token states cover every position.
        return Predictions(
            intent_logits=self.intent_head(self.dropout(encoded.pooler_output)),
            slot_logits=self.slot_head(self.dropout(encoded.last_hidden_state)),
        )


def compute_loss(
    predictions: Predictions, intent_ids: Tensor, slot_ids: Tensor, slot_weight: float = 1.0,
) -> Losses:
    """Targets: int64 intent_ids [B], slot_ids [B, L]; IGNORE_SLOT positions are ignored."""
    intent = F.cross_entropy(predictions.intent_logits, intent_ids)
    # Flatten [B, L, num_slots] to [B*L, num_slots], keeping classes on the last axis.
    slot = F.cross_entropy(predictions.slot_logits.flatten(0, 1), slot_ids.flatten(),
                           ignore_index=IGNORE_SLOT)
    return Losses(intent=intent, slot=slot, total=intent + slot_weight * slot)


def save_checkpoint(model: JointModel, path: Path) -> None:
    """An inference snapshot; optimizer/RNG state is not included."""
    torch.save({"weights": model.state_dict(), "encoder_config": model.encoder.config.to_dict(),
                "intents": model.labels.intent_names, "slots": model.labels.slot_names,
                "dropout": model.dropout.p}, path)


def load_checkpoint(path: Path, device: torch.device | str = "cpu") -> JointModel:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    # Rebuild the architecture directly from the saved config: no downloads or pretrained load.
    encoder = BertModel(BertConfig.from_dict(saved["encoder_config"]))
    model = JointModel(encoder, Labels(intent_names=saved["intents"], slot_names=saved["slots"]), saved["dropout"])
    model.load_state_dict(saved["weights"])
    return model.to(device).eval()
