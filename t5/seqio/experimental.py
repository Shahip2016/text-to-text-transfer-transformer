# Copyright 2021 The T5 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Experimental utilities for SeqIO."""
import inspect
from typing import Mapping

from t5.seqio import dataset_providers


CacheDatasetPlaceholder = dataset_providers.CacheDatasetPlaceholder
Mixture = dataset_providers.Mixture
MixtureRegistry = dataset_providers.MixtureRegistry
Task = dataset_providers.Task
TaskRegistry = dataset_providers.TaskRegistry


def _get_fully_cached_name(
    original_name: str,
    sequence_length: Mapping[str, int]
) -> str:
  """Generates name for fully-cached task or mixture."""
  new_name = f'{original_name}_'
  if len(sequence_length) == 1:
    new_name += str(next(iter(sequence_length.values())))
  else:
    new_name += '_'.join(
        f'{feat}{sequence_length[feat]}' for feat in sequence_length)
  return new_name


def add_fully_cached_task(
    task_name: str,
    sequence_length: Mapping[str, int]
) -> Task:
  """Adds fully-cached version of the task for given sequence lengths."""
  task = TaskRegistry.get(task_name)
  new_name = _get_fully_cached_name(task_name, sequence_length)

  try:
    return TaskRegistry.get(new_name)
  except ValueError:
    pass

  fixed_sequence_length = sequence_length

  new_preprocessors = []
  for prep in task.preprocessors:
    if isinstance(prep, CacheDatasetPlaceholder):
      continue

    def wrapped_prep(ds, output_features, prep=prep):
      prep_args = inspect.signature(prep).parameters.keys()
      extra_kwargs = {}
      if 'sequence_length' in prep_args:
        extra_kwargs['sequence_length'] = fixed_sequence_length
      if 'output_features' in prep_args:
        extra_kwargs['output_features'] = output_features
      return prep(ds, **extra_kwargs)

    new_preprocessors.append(wrapped_prep)

  # Cache at the end of the pipeline.
  new_preprocessors.append(CacheDatasetPlaceholder(required=True))

  # Add post-cache preprocessor to ensure the runtime sequence length is valid.
  def validate_sequence_length(ds, sequence_length):
    if sequence_length is not None and sequence_length != fixed_sequence_length:
      raise ValueError(
          f"Fully-cached task '{new_name}' can only be loaded with "
          f'`sequence_length={fixed_sequence_length}` or `None`.')
    return ds
  new_preprocessors.append(validate_sequence_length)

  return TaskRegistry.add(
      new_name,
      source=task.source,
      preprocessors=new_preprocessors,
      output_features=task.output_features,
      metric_fns=task.metric_fns,
  )


def add_fully_cached_mixture(
    mixture_name: str,
    sequence_length: Mapping[str, int]
) -> Mixture:
  """Adds fully-cached version of the mixture for given sequence lengths."""
  mixture = MixtureRegistry.get(mixture_name)
  new_name = _get_fully_cached_name(mixture_name, sequence_length)

  # Register fully-cached tasks for the mixture.
  new_tasks = [
      add_fully_cached_task(task.name, sequence_length)
      for task in mixture.tasks]
  return MixtureRegistry.add(
      new_name,
      [(new_t.name, mixture._task_to_rate[old_t.name])  # pylint:disable=protected-access
       for old_t, new_t in zip(mixture.tasks, new_tasks)])

