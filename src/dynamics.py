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

  def step(self, genome: Genome):
    d = self.d
    new_neurons = torch.zeros_like(self.neurons)

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

    # process signals per group


    # handle apoptosis and mitosis
    pass
