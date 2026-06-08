from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src" / "model"))

import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModel

from config_model import Config


class PhoBertLoraQA(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or Config.from_yaml(path="experiments/phobert/config.yaml")
        self.basemodel = AutoModel.from_pretrained(self.config.model_name)

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias=self.config.lora_bias,
        )
        self.encoder = get_peft_model(self.basemodel, lora_config)
        self.dropout = nn.Dropout(self.config.dropout_rate)
        self.relu = nn.ReLU()
        self.qa_outputs = nn.Linear(self.basemodel.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state

        x = self.dropout(hidden_state)
        x = self.relu(x)
        logits = self.qa_outputs(x)

        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)
