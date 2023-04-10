# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer

import jax
from flax import core
from flax.core import freeze, unfreeze

import optax
import jax.numpy as jnp
from flax.training import dynamic_scale as dynamic_scale_lib
from flax import struct
from typing import Any, Callable

import torch


class TrainState(struct.PyTreeNode):
  step: int
  apply_fn: Callable = struct.field(pytree_node=False)
  params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
  quant_params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
  batch_stats: Any
  tx: optax.GradientTransformation = struct.field(pytree_node=False)
  opt_state: optax.OptState = struct.field(pytree_node=True)
  dynamic_scale: dynamic_scale_lib.DynamicScale = None

  @classmethod
  def create(cls, *, apply_fn, params, quant_params, tx, **kwargs):
    """Creates a new instance with `step=0` and initialized `opt_state`."""
    return cls(
        step=0,
        apply_fn=apply_fn,
        params=params,
        quant_params=quant_params,
        tx=None,
        opt_state=None,
        **kwargs,
    )


map_dict = {
    '11': '0',
    '12': '1',
    '21': '2',
    '22': '3',
    '31': '4',
    '32': '5',
    '41': '6',
    '42': '7',
}


def resnet_load_pretrained_weights(state, location):

  # Load torch style resnet18.
  torch_state = torch.load(location, map_location=torch.device('cpu'))

  torch_weights = unfreeze(jax.tree_util.tree_map(
      lambda x: jnp.zeros(x.shape), state.params))
  torch_bn_stats = unfreeze(jax.tree_util.tree_map(
      lambda x: jnp.zeros(x.shape), state.batch_stats))
  for key, value in torch_state.items():

    # init block

    if key == 'features.init_block.conv.conv.weight':
      torch_weights['conv_init']['kernel'] = jnp.moveaxis(
          jnp.array(value), (0, 1, 2, 3), (3, 2, 0, 1))
      continue
    if key == 'features.init_block.conv.bn.weight':
      torch_weights['bn_init']['scale'] = jnp.array(value)
      continue
    if key == 'features.init_block.conv.bn.bias':
      torch_weights['bn_init']['bias'] = jnp.array(value)
      continue
    if key == 'features.init_block.conv.bn.running_mean':
      torch_bn_stats['bn_init']['mean'] = jnp.array(value)
      continue
    if key == 'features.init_block.conv.bn.running_var':
      torch_bn_stats['bn_init']['var'] = jnp.array(value)
      continue

    # output block

    if key == 'output.weight':
      torch_weights['Dense_0']['kernel'] = jnp.transpose(jnp.array(value))
    if key == 'output.bias':
      torch_weights['Dense_0']['bias'] = jnp.array(value)

    # body blocks

    if 'features.stage' in key:
      stage_id = key.split('.')[1][-1]
      unit_id = key.split('.')[2][-1]

      map_k = map_dict[stage_id + unit_id]

      if key == 'features.stage' + stage_id + '.unit' + unit_id + \
              '.body.conv1.conv.weight':
        torch_weights['ResNetBlock_' + map_k]['Conv_0']['kernel'
                                                        ] = jnp.moveaxis(
            jnp.array(value), (0, 1, 2, 3), (3, 2, 0, 1))
        continue
      if key == 'features.stage' + stage_id + '.unit' + unit_id + \
              '.body.conv1.bn.weight':
        torch_weights['ResNetBlock_'
                      + map_k]['BatchNorm_0']['scale'] = jnp.array(value)
        continue
      if key == 'features.stage' + stage_id + '.unit' + unit_id + \
              '.body.conv1.bn.bias':
        torch_weights['ResNetBlock_'
                      + map_k]['BatchNorm_0']['bias'] = jnp.array(value)
        continue
      if key == 'features.stage' + stage_id + '.unit' + unit_id + \
              '.body.conv1.bn.running_mean':
        torch_bn_stats['ResNetBlock_'
                       + map_k]['BatchNorm_0']['mean'] = jnp.array(value)
        continue
      if key == 'features.stage' + stage_id + '.unit' + unit_id + \
              '.body.conv1.bn.running_var':
        torch_bn_stats['ResNetBlock_'
                       + map_k]['BatchNorm_0']['var'] = jnp.array(value)
        continue

      if key == 'features.stage' + stage_id + '.unit' + unit_id + \
              '.body.conv2.conv.weight':
        torch_weights['ResNetBlock_' + map_k]['Conv_1']['kernel'
                                                        ] = jnp.moveaxis(
            jnp.array(value), (0, 1, 2, 3), (3, 2, 0, 1))
        continue
      if key == 'features.stage' + stage_id + '.unit' + unit_id + \
              '.body.conv2.bn.weight':
        torch_weights['ResNetBlock_'
                      + map_k]['BatchNorm_1']['scale'] = jnp.array(value)
        continue
      if key == 'features.stage' + stage_id + '.unit' + unit_id + \
              '.body.conv2.bn.bias':
        torch_weights['ResNetBlock_'
                      + map_k]['BatchNorm_1']['bias'] = jnp.array(value)
        continue
      if key == 'features.stage' + stage_id + '.unit' + unit_id + \
              '.body.conv2.bn.running_mean':
        torch_bn_stats['ResNetBlock_'
                       + map_k]['BatchNorm_1']['mean'] = jnp.array(value)
        continue
      if key == 'features.stage' + stage_id + '.unit' + unit_id + \
              '.body.conv2.bn.running_var':
        torch_bn_stats['ResNetBlock_'
                       + map_k]['BatchNorm_1']['var'] = jnp.array(value)
        continue

      if key == 'features.stage' + stage_id + '.unit' + unit_id + \
              '.identity_conv.conv.weight':
        torch_weights['ResNetBlock_' + map_k]['conv_proj']['kernel'
                                                           ] = jnp.moveaxis(
            jnp.array(value), (0, 1, 2, 3), (3, 2, 0, 1))
        continue
      if key == 'features.stage' + stage_id + '.unit' + unit_id + \
              '.identity_conv.bn.weight':
        torch_weights['ResNetBlock_'
                      + map_k]['norm_proj']['scale'] = jnp.array(value)
        continue
      if key == 'features.stage' + stage_id + '.unit' + unit_id + \
              '.identity_conv.bn.bias':
        torch_weights['ResNetBlock_'
                      + map_k]['norm_proj']['bias'] = jnp.array(value)
        continue
      if key == 'features.stage' + stage_id + '.unit' + unit_id + \
              '.identity_conv.bn.running_mean':
        torch_bn_stats['ResNetBlock_'
                       + map_k]['norm_proj']['mean'] = jnp.array(value)
        continue
      if key == 'features.stage' + stage_id + '.unit' + unit_id + \
              '.identity_conv.bn.running_var':
        torch_bn_stats['ResNetBlock_'
                       + map_k]['norm_proj']['var'] = jnp.array(value)
        continue

  general_params = torch_weights
  batch_stats = torch_bn_stats

  return TrainState.create(
      apply_fn=state.apply_fn,
      params=freeze(general_params),
      quant_params=state.quant_params,
      tx=state.tx,
      batch_stats=freeze(batch_stats),
  )
