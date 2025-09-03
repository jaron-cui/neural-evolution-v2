from dataclasses import dataclass
from typing import List, Callable, Dict, Tuple

import torch
import torch.nn as nn

from neuron import NeuronDataFields, NeuronDataField


@dataclass
class Genome:
  initial_latent_states: Dict[str, torch.Tensor]  # each will have a neuron instantiated in its own chamber
  gestation_duration: int
  gestation_hormone: torch.Tensor
  population_hormone: torch.Tensor
  step_network: 'StepNetwork'
  connection_phenotype_network: 'ConnectionPhenotypeNetwork'
  connection_change_network: 'ConnectionChangeNetwork'
  mitosis_network: 'MitosisNetwork'
  max_chamber_capacity: int = 20
  chamber_count: int = 10


class GenomicNetwork(nn.Module):
  def __init__(
    self,
    d: NeuronDataFields,
    input_fields: List[NeuronDataField],
    output_fields: List[NeuronDataField],
    apply_input_mask: bool = True,
    reference_prior: List[Callable[[torch.Tensor], torch.Tensor]] = None,
    hidden_size: int = 100,
    hidden_layers: int = 1
  ):
    super().__init__()
    self.d = d
    self.input_fields = input_fields
    self.output_fields = output_fields
    self.input_indices = d.indices(self.input_fields)
    self.reference_prior = reference_prior
    self.apply_input_mask = apply_input_mask

    input_size = sum([field.size for field in input_fields])
    output_size = sum([field.size for field in output_fields])
    self.layers = self._create_network(input_size, output_size, hidden_size, hidden_layers)

    if len(output_fields) != len(reference_prior):
      raise ValueError('The reference prior list length must correspond to the number of output fields.')

  @staticmethod
  def _create_network(input_size: int, output_size: int, hidden_size: int, hidden_layers: int) -> nn.Module:
    return nn.Sequential(
      nn.Linear(input_size, hidden_size),
      nn.LeakyReLU(inplace=True),
      *sum([[
        nn.Linear(hidden_size, hidden_size),
        nn.LeakyReLU(inplace=True)
      ] for _ in range(hidden_layers)], []),
      nn.Linear(hidden_size, output_size)
    )

  def forward(self, neuron_data: torch.Tensor, split_result: bool = False) -> Tuple[torch.Tensor, ...] | torch.Tensor:
    d = self.d
    x = neuron_data[..., self.input_indices] if self.apply_input_mask else neuron_data
    out: torch.Tensor = self.layers(x)

    if split_result:
      return tuple(out[d.indices(f, self.output_fields)] for f in self.output_fields)
    else:
      return out


class StepNetwork(GenomicNetwork):
  def __init__(self, d: NeuronDataFields):
    super().__init__(
      d,
      input_fields=[
        d.total_input_connection_strength,
        d.total_output_connection_strength,
        d.external_sodium_level,
        d.internal_potassium_level,
        d.max_internal_potassium_level,
        d.latent_state,
        d.hormone_level,
        d.mitosis_progress,
        d.apoptosis_progress,
        d.group_affinities,

        d.base_output_signal_rate,
        d.input_signal_rate_multiplier,
        d.potassium_recovery_rate
      ],
      output_fields=[
        d.latent_state,
        d.mitosis_progress,
        d.apoptosis_progress,

        d.base_output_signal_rate,
        d.input_signal_rate_multiplier,
        d.potassium_recovery_rate
      ],
      reference_prior=[
        lambda t: t[..., d.indices(d.latent_state)],
        lambda t: torch.full((*t.shape[:-1], 1), 0.05),
        lambda t: torch.full((*t.shape[:-1], 1), -0.1),

        lambda t: torch.full((*t.shape[:-1], 1), 1.0),
        lambda t: torch.full((*t.shape[:-1], 1), 1.0),
        lambda t: torch.full((*t.shape[:-1], 1), 0.1)
      ]
    )


class ConnectionPhenotypeNetwork(GenomicNetwork):
  def __init__(self, d: NeuronDataFields):
    super().__init__(
      d,
      input_fields=[
        d.total_input_connection_strength,
        d.total_output_connection_strength,
        d.external_sodium_level,
        d.internal_potassium_level,
        d.max_internal_potassium_level,
        d.latent_state,
        d.hormone_level,
        d.mitosis_progress,
        d.apoptosis_progress,
        d.group_affinities,

        d.base_output_signal_rate,
        d.input_signal_rate_multiplier,
        d.potassium_recovery_rate
      ],
      output_fields=[
        d.connection_phenotype
      ],
      reference_prior=[
        lambda t: t[..., d.latent_state][..., :d.connection_phenotype.size]
      ]
    )


class ConnectionChangeNetwork(GenomicNetwork):
  def __init__(self, d: NeuronDataFields):
    super().__init__(
      d,
      input_fields=[
        d.connection_phenotype,
        d.connection_phenotype
      ],
      output_fields=[
        d.connection_strength_change
      ],
      reference_prior=[
        lambda _: torch.zeros(1)
      ],
      apply_input_mask=False
    )


class MitosisNetwork(GenomicNetwork):
  def __init__(self, d: NeuronDataFields):
    super().__init__(
      d,
      input_fields=[
        d.total_input_connection_strength,
        d.total_output_connection_strength,
        d.external_sodium_level,
        d.internal_potassium_level,
        d.max_internal_potassium_level,
        d.latent_state,
        d.hormone_level,
        d.mitosis_progress,
        d.apoptosis_progress,
        d.group_affinities,

        d.base_output_signal_rate,
        d.input_signal_rate_multiplier,
        d.potassium_recovery_rate
      ],
      output_fields=[
        d.latent_state,
        d.latent_state,
        d.group_affinities,
        d.connection_strength_change,
        d.connection_strength_change
      ],
      reference_prior=[
        lambda t: t[..., d.indices(d.latent_state)],
        lambda t: t[..., d.indices(d.latent_state)],
        lambda t: t[..., d.indices(d.group_affinities)],
        lambda t: torch.full((*t.shape[:-1], 1), 1.0),
        lambda t: torch.full((*t.shape[:-1], 1), 0.0)
      ]
    )

