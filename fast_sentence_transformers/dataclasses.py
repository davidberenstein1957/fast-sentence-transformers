from typing import TypedDict

import torch


class ModelInput(TypedDict):
    outputs: torch.Tensor
    attention_mask: torch.Tensor


class ModelOutputs(TypedDict):
    outputs: torch.Tensor
    attention_mask: torch.Tensor
