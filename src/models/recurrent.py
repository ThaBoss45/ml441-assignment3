
from __future__ import annotations

import torch
from torch import nn


class ElmanNetwork(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.hidden = hidden
        self.input_to_hidden = nn.Linear(1, hidden)
        self.hidden_to_hidden = nn.Linear(hidden, hidden, bias=False)
        self.hidden_to_output = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, return_trace: bool = False):
        h = x.new_zeros((x.shape[0], self.hidden))
        trace = []
        for step in range(x.shape[1]):
            h = torch.tanh(self.input_to_hidden(x[:, step, :]) + self.hidden_to_hidden(h))
            output = self.hidden_to_output(h)
            if return_trace:
                trace.append((h, output))
        return (output, trace) if return_trace else output


class JordanNetwork(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.hidden = hidden
        self.input_to_hidden = nn.Linear(1, hidden)
        self.output_to_hidden = nn.Linear(1, hidden, bias=False)
        self.hidden_to_output = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, return_trace: bool = False):
        previous_output = x.new_zeros((x.shape[0], 1))
        trace = []
        for step in range(x.shape[1]):
            h = torch.tanh(self.input_to_hidden(x[:, step, :])
                           + self.output_to_hidden(previous_output))
            previous_output = self.hidden_to_output(h)
            if return_trace:
                trace.append((h, previous_output))
        return (previous_output, trace) if return_trace else previous_output


class MultiRecurrentNetwork(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.hidden = hidden
        self.input_to_hidden = nn.Linear(1, hidden)
        self.hidden_to_hidden = nn.Linear(hidden, hidden, bias=False)
        self.output_to_hidden = nn.Linear(1, hidden, bias=False)
        self.hidden_to_output = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, return_trace: bool = False):
        h = x.new_zeros((x.shape[0], self.hidden))
        previous_output = x.new_zeros((x.shape[0], 1))
        trace = []
        for step in range(x.shape[1]):
            h = torch.tanh(self.input_to_hidden(x[:, step, :])
                           + self.hidden_to_hidden(h)
                           + self.output_to_hidden(previous_output))
            previous_output = self.hidden_to_output(h)
            if return_trace:
                trace.append((h, previous_output))
        return (previous_output, trace) if return_trace else previous_output


MODELS = {"elman": ElmanNetwork, "jordan": JordanNetwork,
          "multi": MultiRecurrentNetwork}


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
