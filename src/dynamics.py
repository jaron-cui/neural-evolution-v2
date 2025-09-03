from typing import Tuple

import torch

from genome import Genome
from neuron import NeuronDataFields


class NeuralSimulation:
  def __init__(self, d: NeuronDataFields, genome: Genome):
    max_neuron_count = genome.chamber_count * genome.max_chamber_capacity
    neuron_data_size = d.size()

    self.neurons = torch.zeros((max_neuron_count, neuron_data_size))
    self.living_neuron_mask = torch.zeros(max_neuron_count, dtype=torch.bool)

    # (chamber, src, dst)
    self.chamber_connection_strength = torch.zeros(
      (genome.chamber_count, genome.max_chamber_capacity, genome.max_chamber_capacity))
    # (src_chamber, dst_chamber, connection)
    self.inter_chamber_connection_strength = torch.zeros(
      (genome.chamber_count, genome.chamber_count, genome.max_chamber_capacity))
    # (src_chamber, dst_chamber, connection, src-dst)
    self.inter_chamber_connections = torch.full(
      (genome.chamber_count, genome.chamber_count, genome.max_chamber_capacity, 2), -1, dtype=torch.int16)

    self.d = d
    self.genome = genome

    self._add_initial_neurons()

  def _add_initial_neurons(self):
    d = self.d
    for chamber_number, initial_latent in enumerate(self.genome.initial_latent_states.values()):
      neuron_index = chamber_number * self.genome.max_chamber_capacity

      neuron_data = torch.zeros(d.size())
      neuron_data[d.indices(d.latent_state)] = initial_latent
      neuron_data[d.indices(d.group_affinities)] = torch.nn.functional.one_hot(
        torch.tensor(chamber_number, dtype=torch.int), self.genome.chamber_count)

      self.neurons[neuron_index] = neuron_data

  def _allocate_neurons(self, chamber_numbers: torch.Tensor) -> torch.Tensor:
    allocated_neuron_indices = torch.full_like(chamber_numbers, -1)
    for chamber_number in range(self.genome.chamber_count):
      chamber_start = chamber_number * self.genome.max_chamber_capacity
      chamber_end = chamber_start + self.genome.max_chamber_capacity

      chamber_mask = torch.zeros_like(self.living_neuron_mask, dtype=torch.bool)
      chamber_mask[chamber_start:chamber_end] = True

      chamber_requests = chamber_numbers.eq(chamber_number)
      chamber_request_count = chamber_requests.int().sum()

      free_spaces = ~self.living_neuron_mask & chamber_mask
      cumulative_sum = torch.cumsum(free_spaces.int(), dim=-1)
      if cumulative_sum[-1] < chamber_request_count:
        raise ValueError('Cannot allocate neurons: chamber is full.')

      free_spaces[cumulative_sum > chamber_request_count] = False
      allocated_indices = torch.nonzero(free_spaces, as_tuple=False).squeeze()

      self.living_neuron_mask[allocated_indices] = False
      allocated_neuron_indices[chamber_requests] = allocated_indices
    self.neurons[allocated_neuron_indices].zero_()
    return allocated_neuron_indices

  def modulate_chamber_selection(self, chamber_affinity: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    chamber_affinity = chamber_affinity.clone()
    chamber_living_mask = self.living_neuron_mask.view((self.genome.chamber_count, self.genome.max_chamber_capacity))
    chamber_selection = torch.full((chamber_affinity.size(0),), -1, dtype=torch.int)
    while chamber_selection.eq(-1).sum() > 0:
      naive_chamber_selection = torch.argmax(chamber_affinity, dim=-1)
      reproductive_round = first_n_unique_indices(naive_chamber_selection)

      population_counts = chamber_living_mask.sum(dim=-1)
      population_ratios = population_counts / self.genome.max_chamber_capacity

      chamber_affinity[reproductive_round] -= population_ratios
      chamber_selection[reproductive_round] = naive_chamber_selection[reproductive_round]
    return chamber_affinity, chamber_selection

  def step(self, genome: Genome):
    d = self.d
    new_neurons = torch.zeros_like(self.neurons)

    # TODO: hormones for population

    # calculate systemic statistics TODO: perhaps move this after connection and other updates
    inter_chamber_connection_mask = self.inter_chamber_connections[:, :, :, 0] >= 0
    inter_chamber_connection_src = self.inter_chamber_connections[inter_chamber_connection_mask, 0].flatten()
    inter_chamber_connection_dst = self.inter_chamber_connections[inter_chamber_connection_mask, 1].flatten()
    inter_chamber_connection_strength = self.inter_chamber_connection_strength[inter_chamber_connection_mask].flatten()

    total_output_connection_strength = self.chamber_connection_strength.sum(dim=2).flatten()
    total_output_connection_strength.scatter_add_(0, inter_chamber_connection_src, inter_chamber_connection_strength)
    total_input_connection_strength = self.chamber_connection_strength.sum(dim=1).flatten()
    total_input_connection_strength.scatter_add_(0, inter_chamber_connection_dst, inter_chamber_connection_strength)

    new_neurons[:, d.indices(d.total_output_connection_strength)] = total_input_connection_strength
    new_neurons[:, d.indices(d.total_input_connection_strength)] = total_input_connection_strength

    # step network -> latent state, mitosis/apoptosis change, signal rate, multiplier, potassium recovery
    old_living_neurons = self.neurons[self.living_neuron_mask]
    (
      new_latent_state,
      mitosis_progress,
      apoptosis_progress,
      new_base_output_signal_rate,
      new_input_signal_rate_multiplier,
      new_potassium_recovery_rate
    ) = self.genome.step_network.forward(old_living_neurons, split_result=True)
    new_neurons[self.living_neuron_mask, d.indices(d.latent_state)] = new_latent_state
    new_neurons[self.living_neuron_mask, d.indices(d.mitosis_progress)] = (
      old_living_neurons[:, d.indices(d.mitosis_progress)] + mitosis_progress
    )
    new_neurons[self.living_neuron_mask, d.indices(d.apoptosis_progress)] = (
      old_living_neurons[:, d.indices(d.apoptosis_progress) + apoptosis_progress]
    )
    new_neurons[self.living_neuron_mask, d.indices(d.base_output_signal_rate)] = new_base_output_signal_rate
    new_neurons[self.living_neuron_mask, d.indices(d.input_signal_rate_multiplier)] = new_input_signal_rate_multiplier
    new_neurons[self.living_neuron_mask, d.indices(d.potassium_recovery_rate)] = new_potassium_recovery_rate

    # connection phenotype, connection change network per group
    connection_phenotype = torch.zeros((self.neurons.size(0), d.size(d.connection_phenotype)))
    connection_phenotype[self.living_neuron_mask] = self.genome.connection_phenotype_network.forward(old_living_neurons)
    chamber_count = self.genome.chamber_count
    for chamber_number in range(chamber_count):
      chamber_start, chamber_end = chamber_number * chamber_count, (chamber_number + 1) * chamber_count
      chamber_living_mask = self.living_neuron_mask[chamber_start:chamber_end]

      chamber_neuron_indices = torch.arange(
        self.neurons.size(0), dtype=torch.int)[chamber_start:chamber_end][chamber_living_mask]

      src_in_chamber = (inter_chamber_connection_src >= chamber_start) & (inter_chamber_connection_src < chamber_end)
      dst_in_chamber = (inter_chamber_connection_dst >= chamber_start) & (inter_chamber_connection_dst < chamber_end)
      external_neuron_indices = torch.cat((
        inter_chamber_connection_src[dst_in_chamber],
        inter_chamber_connection_dst[src_in_chamber])
      ).unique()
      # external_phenotypes = connection_phenotype[external_neurons]
      internal_pairs = torch.cartesian_prod(chamber_neuron_indices, chamber_neuron_indices)
      internal_to_external_pairs = torch.cartesian_prod(chamber_neuron_indices, external_neuron_indices)
      external_to_internal_pairs = torch.cartesian_prod(external_neuron_indices, chamber_neuron_indices)

      all_pairs = torch.cat((internal_pairs, internal_to_external_pairs, external_to_internal_pairs))

      phenotype_pairs = connection_phenotype[all_pairs].flatten(start_dim=1)
      connection_strength_changes = self.genome.connection_change_network.forward(phenotype_pairs)
      internal_changes = connection_strength_changes[:internal_pairs.numel()].view(
        (chamber_neuron_indices.size(0), chamber_neuron_indices.size(0)))
      internal_to_external_changes = (
        connection_strength_changes[internal_pairs.numel():internal_pairs.numel() + internal_to_external_pairs.numel()]
      ).view((chamber_neuron_indices.size(0), external_neuron_indices.size(0)))
      external_to_internal_changes = (
        connection_strength_changes[internal_pairs.numel() + internal_to_external_pairs.numel():]
      ).view((external_neuron_indices.size(0), chamber_neuron_indices.size(0)))

      # WARNING: THIS DIRECTLY MUTATES STATE. TODO: CHECK THAT THIS IS FINE
      self.chamber_connection_strength[chamber_number] += internal_changes
      external_to_internal_mask = inter_chamber_connection_mask.clone()
      external_to_internal_mask[external_to_internal_mask] = dst_in_chamber
      self.inter_chamber_connection_strength[external_to_internal_mask] += external_to_internal_changes
      internal_to_external_mask = inter_chamber_connection_mask.clone()
      internal_to_external_mask[internal_to_external_mask] = src_in_chamber
      self.inter_chamber_connection_strength[internal_to_external_mask] += internal_to_external_changes

    # -- process signals per group --
    raw_output_signal_rate = self.neurons[:, d.indices(d.output_signal_rate)]
    raw_output_signal_rate[~self.living_neuron_mask] = 0
    # internal input signal rate
    input_signal_rate = self.chamber_connection_strength * torch.stack(
      raw_output_signal_rate.chunk(self.genome.chamber_count)
    ).unsqueeze(2).sum(dim=1).flatten()

    # add inter-chamber signals
    inter_chamber_signal_rate = inter_chamber_connection_strength * raw_output_signal_rate[inter_chamber_connection_src]
    input_signal_rate.scatter_add_(0, inter_chamber_connection_dst, inter_chamber_signal_rate)

    potassium = self.neurons[:, d.indices(d.internal_potassium_level)]
    potassium_recovery = self.neurons[:, d.indices(d.potassium_recovery_rate)]
    output_signal_rate = torch.minimum((
      input_signal_rate * self.neurons[:, d.indices(d.input_signal_rate_multiplier)]
      + self.neurons[:, d.indices(d.base_output_signal_rate)]
    ), potassium)  # TODO: sodium?
    new_neurons[:, d.indices(d.output_signal_rate)] = output_signal_rate
    new_neurons[:, d.indices(d.internal_potassium_level)] = potassium - output_signal_rate + potassium_recovery


    # handle apoptosis and mitosis
    apoptosis_initiated = self.neurons[:, d.indices(d.apoptosis_progress)] >= 1
    new_neurons[apoptosis_initiated].zero_()
    self.living_neuron_mask[apoptosis_initiated] = False

    mitosis_initiated = (self.neurons[:, d.indices(d.mitosis_progress)] >= 1) & ~apoptosis_initiated
    parent_data = self.neurons[mitosis_initiated]
    parent_indices = torch.nonzero(mitosis_initiated, as_tuple=False).squeeze()
    (
      parent_latent,
      child_latent,
      group_affinities,
      parent_child_connection_strength,
      child_parent_connection_strength
    ) = self.genome.mitosis_network.forward(parent_data)
    chamber_affinity, chamber_selection = self.modulate_chamber_selection(group_affinities)
    new_neurons[mitosis_initiated, d.indices(d.latent_state)] = parent_latent
    child_indices = self._allocate_neurons(chamber_selection)
    new_neurons[child_indices, d.indices(d.latent_state)] = child_latent
    new_neurons[child_indices, d.indices(d.group_affinities)] = chamber_affinity

    parent_chamber_number = parent_indices // self.genome.max_chamber_capacity
    parent_chamber_index = parent_indices % self.genome.max_chamber_capacity
    child_chamber_index = child_indices % self.genome.max_chamber_capacity
    internal_mitosis = chamber_selection.eq(parent_chamber_number)
    self.chamber_connection_strength[
      parent_chamber_number[internal_mitosis],
      parent_chamber_index[internal_mitosis],
      child_chamber_index[internal_mitosis]
    ] = parent_child_connection_strength
    self.chamber_connection_strength[
      parent_chamber_number[internal_mitosis],
      child_chamber_index[internal_mitosis],
      parent_chamber_index[internal_mitosis]
    ] = child_parent_connection_strength

    self.inter_chamber_connections[
      parent_chamber_number[~internal_mitosis],
      chamber_selection[~internal_mitosis],

    ]

def unique_prefix_length(tensor):
  return (torch.cumsum(torch.zeros_like(tensor).scatter_(0, tensor, 1), dim=0).eq(1)).sum()

def first_n_unique_indices(tensor):
  return torch.unique(tensor, return_inverse=True)[1].sort()[1]
