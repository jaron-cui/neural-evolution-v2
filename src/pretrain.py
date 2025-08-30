import time

import torch
from genome import GenomicNetwork, StepNetwork
from neuron import NeuronDataFields


def pretrain_genomic_network(
  d: NeuronDataFields,
  network: GenomicNetwork,
  iterations: int = 10000,
  lr: float = 0.00015,
  batch_size: int = 1024
):
  if network.reference_prior is None:
    raise ValueError('No reference prior associated with genomic network.')

  input_size = d.size(network.input_fields) if network.apply_input_mask else d.size()

  optimizer = torch.optim.Adam(params=network.parameters(), lr=lr)
  criterion = torch.nn.MSELoss()
  for i in range(iterations):
    dummy_neurons = (torch.rand((batch_size, input_size)) * 2 - 1) * 20
    target = torch.cat([prior(dummy_neurons) for prior in network.reference_prior], dim=-1)

    out = network(dummy_neurons)
    loss = criterion(out, target)

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()


def _main():
  # torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')
  d = NeuronDataFields()
  network = StepNetwork(d)

  input_size = d.size(network.input_fields) if network.apply_input_mask else d.size()
  test_input = (torch.rand(input_size) * 2 - 1) * 20
  target = torch.cat([prior(test_input) for prior in network.reference_prior], dim=-1)
  print('Test input:', test_input)
  print('Target output:', target)
  print('Before pretrain:', network(test_input))

  start_time = time.time()
  pretrain_genomic_network(d, network)
  print(time.time() - start_time)

  print('After pretrain:', network(test_input))


if __name__ == '__main__':
  _main()
