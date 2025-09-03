from dataclasses import dataclass, field
from typing import List, Sequence

_id_counter = 0


def _unique_id():
  global _id_counter
  _id_counter += 1
  return _id_counter


@dataclass
class NeuronDataField:
  size: int
  id: int = field(default_factory=_unique_id)


@dataclass
class NeuronDataFields:
  # environmental statistics
  total_output_connection_strength = NeuronDataField(size=1)
  total_input_connection_strength = NeuronDataField(size=1)
  output_signal_rate = NeuronDataField(size=1)

  # neuron state
  external_sodium_level = NeuronDataField(size=1)  # TODO: possibly implement sodium mechanic as a signal buffer
  internal_potassium_level = NeuronDataField(size=1)
  max_internal_potassium_level = NeuronDataField(size=1)
  latent_state = NeuronDataField(size=8)
  hormone_level = NeuronDataField(size=4)
  mitosis_progress = NeuronDataField(size=1)
  apoptosis_progress = NeuronDataField(size=1)
  group_affinities = NeuronDataField(size=10)

  # derived neuron parameters
  base_output_signal_rate = NeuronDataField(size=1)
  input_signal_rate_multiplier = NeuronDataField(size=1)
  potassium_recovery_rate = NeuronDataField(size=1)

  # extra output data
  connection_phenotype = NeuronDataField(size=4)
  connection_strength_change = NeuronDataField(size=1)

  def indices(
    self,
    fields: NeuronDataField | Sequence[NeuronDataField],
    data_fields: Sequence[NeuronDataField] = None
  ) -> slice | List[int]:
    if isinstance(fields, NeuronDataField):
      fields = [fields]
    else:
      fields = sorted(fields, key=lambda f: f.id)

    field_id_set = set(f.id for f in fields)
    if data_fields is None:
      data_fields = self.get_neuron_data_fields()
    data_field_id_set = set(df.id for df in data_fields)

    if any(fid not in data_field_id_set for fid in field_id_set):
      raise ValueError('Non-data fields do not have corresponding indices.')

    first_field_number = next(i for i, fid in enumerate(data_field_id_set) if fid in field_id_set)
    start_index = sum(df.size for df in data_fields[:first_field_number])
    # return a simple slice if all fields are in a continuous block
    if (
      len(data_fields) >= first_field_number + len(fields)
      and all(f.id == df.id for f, df in zip(
        fields, data_fields[first_field_number:first_field_number + len(fields)]
      ))
    ):
      end_index = start_index + sum(f.size for f in fields)
      return slice(start_index, end_index)

    # return a list of indices if there are gaps in the data region specified by the fields
    indices = []
    for df in data_fields[first_field_number:]:
      if df.id in field_id_set:
        indices.extend(range(start_index, start_index + df.size))
      start_index += df

    return indices

  def size(self, fields: Sequence[NeuronDataField] = None) -> int:
    return sum(f.size for f in (fields or self.get_neuron_data_fields()))

  def get_neuron_data_fields(self) -> List[NeuronDataField]:
    return [
      self.total_output_connection_strength,
      self.total_input_connection_strength,
      self.output_signal_rate,
      self.external_sodium_level,
      self.internal_potassium_level,
      self.max_internal_potassium_level,
      self.latent_state,
      self.hormone_level,
      self.mitosis_progress,
      self.apoptosis_progress,
      self.group_affinities,
      self.base_output_signal_rate,
      self.input_signal_rate_multiplier,
      self.potassium_recovery_rate
    ]
