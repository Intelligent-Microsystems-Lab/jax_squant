# Author: Clemens JS Schaefer
# Implementation of SQuant
# https://arxiv.org/pdf/2202.07471.pdf
# based on https://github.com/clevercool/SQuant

import functools
import numpy as np

import jax
import jaxlib
import jax.numpy as jnp

from flax import linen as nn


def SQuant_func(flip_number, flip_up_number, flip_down_number,
                rounding_error_sum, rounding_number, rounding_error,
                up_number, up_error, up_priority, up_order, down_number,
                down_error, down_priority, down_order,
                ):

  fround_func = functools.partial(round_func, flip_number=flip_number,
                                  flip_up_number=flip_up_number,
                                  flip_down_number=flip_down_number)

  res1 = jax.vmap(jax.vmap(fround_func))(
      rounding_error_sum=rounding_error_sum,
      rounding_number_=rounding_number,
      rounding_error_=rounding_error,
      up_number_=up_number,
      up_error_=up_error,
      up_priority_=up_priority,
      up_order_=up_order,
      down_number_=down_number,
      down_error_=down_error,
      down_priority_=down_priority,
      down_order_=down_order,
  )

  rounding_number = res1[0]
  rounding_error = res1[1]
  up_priority = res1[2]
  down_priority = res1[3]

  return rounding_number, rounding_error, up_priority, down_priority


def round_func(
    flip_number,
    flip_up_number,
    flip_down_number,
    rounding_error_sum,
    rounding_number_,
    rounding_error_,
    up_number_,
    up_error_,
    up_priority_,
    up_order_,
    down_number_,
    down_error_,
    down_priority_,
    down_order_,
):

  number_ = jnp.where(rounding_error_sum < 0, up_number_, down_number_)
  error_ = jnp.where(rounding_error_sum < 0, up_error_, down_error_)
  priority_ = jnp.where(rounding_error_sum < 0, up_priority_, down_priority_)
  order_ = jnp.where(rounding_error_sum < 0, up_order_, down_order_)
  # error_1 = jnp.where(rounding_error_sum < 0, down_error_, up_error_)
  priority_1 = jnp.where(rounding_error_sum < 0, down_priority_, up_priority_)
  # flip_number_ = jnp.where(rounding_error_sum < 0,
  #                          flip_up_number, flip_down_number)
  is_up = jnp.where(rounding_error_sum < 0, True, False)

  rounding_error_sum = jnp.abs(rounding_error_sum)
  topk = rounding_error_sum.round()
  over_squant = (topk >= rounding_error_sum)

  rounding_error_ = rounding_error_.at[order_].get()
  rounding_error_ = jnp.where(jnp.arange(
      rounding_error_.shape[0]) < topk, error_.at[order_].get(),
      rounding_error_)
  rounding_error_ = rounding_error_.at[jnp.argsort(order_)].get()

  rounding_number_ = rounding_number_.at[order_].get()
  rounding_number_ = jnp.where(jnp.arange(
      rounding_number_.shape[0]) < topk, number_.at[order_].get(),
      rounding_number_)
  rounding_number_ = rounding_number_.at[jnp.argsort(order_)].get()

  priority_1 = jnp.where(over_squant,
                         priority_1.at[order_.at[topk.astype(jnp.int32) - 1
                                                 ].get().astype(
                             jnp.int32)].set(
                             jnp.abs(rounding_error_[order_.at[
                                 topk.astype(jnp.int32) - 1].get().astype(
                                 jnp.int32)])),
                         priority_1.at[order_.at[topk.astype(jnp.int32)
                                                 ].get().astype(
                             jnp.int32)].set(
                             jnp.abs(rounding_error_[order_.at[
                                 topk.astype(jnp.int32)].get().astype(
                                 jnp.int32)]))
                         )

  up_priority_ = jnp.where(is_up, priority_, priority_1)
  down_priority_ = jnp.where(is_up, priority_1, priority_)

  return rounding_number_, rounding_error_, up_priority_, down_priority_


def sigma_fn(tensor, is_signed):
  if not is_signed:
    return tensor[tensor > 0].std()
  return tensor.std()


def asymmetric_linear_quantization_params(num_bits,
                                          saturation_min,
                                          saturation_max,
                                          integral_zero_point=True,
                                          signed=True):
  """
  Compute the scaling factor and zeropoint with the given quantization range.
  saturation_min: lower bound for quantization range
  saturation_max: upper bound for quantization range
  """
  n = 2 ** num_bits - 1
  scale = n / jnp.clip((saturation_max - saturation_min), a_min=1e-8)
  # hard to gurantee numerical equivalence to pytorch here.
  zero_point = scale * saturation_min

  if integral_zero_point:
    if isinstance(zero_point, jaxlib.xla_extension.ArrayImpl):
      zero_point = jnp.round(zero_point)  # .round()
    else:
      zero_point = jnp.round(zero_point)  # float(round(zero_point))
  if signed:
    zero_point += 2**(num_bits - 1)
  return scale, zero_point


def linear_quantize(input, scale, zero_point, inplace=False):
  """
  Quantize single-precision input tensor to integers with the given
    scaling factor and zeropoint.
  input: single-precision input tensor to be quantized
  scale: scaling factor for quantization
  zero_pint: shift for quantization
  """

  # reshape scale and zeropoint for convolutional weights and activation
  if len(input.shape) == 4:
    scale = jnp.reshape(scale, (-1, 1, 1, 1))
    zero_point = jnp.reshape(zero_point, (-1, 1, 1, 1))
  # reshape scale and zeropoint for linear weights
  elif len(input.shape) == 2:
    scale = jnp.reshape(scale, [-1, 1])
    zero_point = jnp.reshape(zero_point, [-1, 1])
  # mapping single-precision input to integer values with the given
  # scale and zeropoint
  return (scale * input - zero_point)


def linear_dequantize(input, scale, zero_point, inplace=False):
  """
  Map integer input tensor to fixed point float point with given
    scaling factor and zeropoint.
  input: integer input tensor to be mapped
  scale: scaling factor for quantization
  zero_pint: shift for quantization
  """

  # reshape scale and zeropoint for convolutional weights and activation
  if len(input.shape) == 4:
    scale = jnp.reshape(scale, (-1, 1, 1, 1))
    zero_point = jnp.reshape(zero_point, (-1, 1, 1, 1))
  # reshape scale and zeropoint for linear weights
  elif len(input.shape) == 2:
    scale = jnp.reshape(scale, [-1, 1])
    zero_point = jnp.reshape(zero_point, [-1, 1])
  # mapping integer input to fixed point float point value with given
  # scaling factor and zeropoint
  return (input + zero_point) / scale


def adaptive_round(x, t_min=None, t_max=None, squant_k='make that required',
                   squant_c='make that required', bit='make this required'):
  # Get the rounding integer and fraction.
  rounding_number = jnp.round(x)
  rounding_error = rounding_number - x

  up_number = rounding_number
  up_error = rounding_error
  up_error = jnp.where(x >= t_max, 0, up_error)
  up_error = jnp.where(up_error > 0, 0, up_error)
  up_priority = jnp.abs(up_error)

  up_error = jnp.where(up_error != 0, up_error + 1, up_error)
  up_number = jnp.where(up_error != 0, up_number + 1, up_number)

  down_number = rounding_number
  down_error = rounding_error
  down_error = jnp.where(x <= t_min, 0, down_error)
  down_error = jnp.where(down_error <= 0, 0, down_error)
  down_priority = jnp.abs(down_error)

  down_error = jnp.where(down_error != 0, down_error - 1, down_error)
  down_number = jnp.where(down_error != 0, down_number - 1, down_number)

  flip_number = jnp.array([0.0])
  flip_up_number = jnp.array([0.0])
  flip_down_number = jnp.array([0.0])

  conver_shape = jnp.reshape(x, [x.shape[0], x.shape[1], -1]).shape
  local_squant_k = squant_k
  if conver_shape[2] == 1:
    local_squant_k = False

  if squant_k and local_squant_k:
    rounding_error_sum = jnp.reshape(rounding_error, conver_shape).sum(axis=-1)

    # maintaining order of zero is hard (not guranteed yet).
    up_order = jnp.flip(jnp.argsort(
        jnp.reshape(up_priority, conver_shape)), axis=-1)
    down_order = jnp.flip(jnp.argsort(
        jnp.reshape(down_priority, conver_shape)), axis=-1)
    up_priority *= 0.0
    down_priority *= 0.0

    res = SQuant_func(
        flip_number,
        flip_up_number,
        flip_down_number,

        rounding_error_sum,
        jnp.reshape(rounding_number, conver_shape),
        jnp.reshape(rounding_error, conver_shape),

        jnp.reshape(up_number, conver_shape),
        jnp.reshape(up_error, conver_shape),
        jnp.reshape(up_priority, conver_shape),
        up_order,

        jnp.reshape(down_number, conver_shape),
        jnp.reshape(down_error, conver_shape),
        jnp.reshape(down_priority, conver_shape),
        down_order,
    )

    rounding_number = jnp.reshape(res[0], rounding_number.shape)
    rounding_error = jnp.reshape(res[1], rounding_error.shape)
    up_priority = jnp.reshape(res[2], up_priority.shape)
    down_priority = jnp.reshape(res[3], down_priority.shape)

  if squant_c:
    conver_shape = (1, x.shape[0], -1)
    rounding_error_sum = jnp.reshape(rounding_error, conver_shape).sum(axis=-1)
    up_order = jnp.flip(jnp.argsort(
        jnp.reshape(up_priority, conver_shape)), axis=-1)
    down_order = jnp.flip(jnp.argsort(
        jnp.reshape(down_priority, conver_shape)), axis=-1)

    res = SQuant_func(
        flip_number,
        flip_up_number,
        flip_down_number,

        rounding_error_sum,
        jnp.reshape(rounding_number, conver_shape),
        jnp.reshape(rounding_error, conver_shape),

        jnp.reshape(up_number, conver_shape),
        jnp.reshape(up_error, conver_shape),
        jnp.reshape(up_priority, conver_shape),
        up_order,

        jnp.reshape(down_number, conver_shape),
        jnp.reshape(down_error, conver_shape),
        jnp.reshape(down_priority, conver_shape),
        down_order
    )
    rounding_number = jnp.reshape(res[0], rounding_number.shape)
    rounding_error = jnp.reshape(res[1], rounding_error.shape)
    up_priority = jnp.reshape(res[2], up_priority.shape)
    down_priority = jnp.reshape(res[3], down_priority.shape)

  rounding_number = jnp.clip(rounding_number, a_min=t_min, a_max=t_max)
  # jnp.unique is not jit compatible.
  # assert (np.prod(jnp.unique(rounding_number).shape) <= 2 ** bit)
  return rounding_number


def squant_fn(tensor, bit, is_perchannel, squant_k, squant_c, scale_off=False,
              shape_c=False):

  if shape_c is False:
    # reshuffle axis to match pytorch.
    if len(tensor.shape) == 4:
      tensor = jnp.moveaxis(tensor, (0, 1, 2, 3), (3, 2, 0, 1))
    else:
      tensor = jnp.transpose(tensor)

  if is_perchannel:
    x_max = jnp.expand_dims(
        jnp.max(jnp.reshape(tensor, [tensor.shape[0], -1]), axis=1), axis=1)
    x_min = jnp.expand_dims(
        jnp.min(jnp.reshape(tensor, [tensor.shape[0], -1]), axis=1), axis=1)
  else:
    x_max = tensor.max()
    x_min = tensor.min()

  scale, zero_point = asymmetric_linear_quantization_params(bit, x_min, x_max)

  if scale_off is not False:
    scale = np.load(scale_off)

  quant_tensor = linear_quantize(tensor, scale, zero_point, inplace=False)

  n = 2 ** (bit - 1)

  quant_tensor = adaptive_round(
      quant_tensor, -n, n - 1, squant_k, squant_c, bit)

  quant_tensor = jnp.clip(quant_tensor, a_min=-n, a_max=n - 1)
  if scale_off is False:
    quant_tensor = linear_dequantize(
        quant_tensor, scale, zero_point, inplace=False)

  if shape_c is False:
    # bring tensor back into og shape
    if len(tensor.shape) == 4:
      quant_tensor = jnp.moveaxis(quant_tensor, (0, 1, 2, 3), (2, 3, 1, 0))
    else:
      quant_tensor = jnp.transpose(quant_tensor)

  return quant_tensor


class uniform_static(nn.Module):
  bits: int = 4
  percent: float = 12.
  sign: bool = True

  @nn.compact
  def __call__(self, x, no_quant=False):
    if type(self.bits) == int:
      assert (
          self.bits > 1
      ), "Bit widths below 2 bits are not supported but got bits: "\
          + str(self.bits)

    if no_quant:
      return x

    xmax = self.variable(
        'quant_params', 'xmax', jnp.zeros, (1,))
    xmin = self.variable(
        'quant_params', 'xmin', jnp.zeros, (1,))

    if self.is_mutable_collection('quant_params'):

      x_max = jnp.max(x)
      alpha = self.percent * jnp.abs(x).max()

      if not self.sign:
        sigma = jnp.nanstd(jnp.where(x > 0, x, jnp.nan))
      else:
        sigma = jnp.std(x)

      alpha = sigma * self.percent
      if self.sign:
        alpha = self.percent * sigma / 1.25

      if self.bits < 6:
        # For small bit, need clip.
        alpha = jnp.minimum(alpha, x_max)

      if self.sign:
        xmin.value = -alpha
      else:
        xmin.value = jnp.zeros_like(alpha)
      xmax.value = alpha

    scale, zero_point = asymmetric_linear_quantization_params(
        self.bits, xmin.value, xmax.value, integral_zero_point=True,
        signed=self.sign)

    new_quant_x = jnp.round(linear_quantize(
        x, scale, zero_point, inplace=False))
    n = 2**(self.bits - 1)
    if self.sign:
      new_quant_x = jnp.clip(new_quant_x, -n, n - 1)
    else:
      new_quant_x = jnp.clip(new_quant_x, 0, 2 * n - 1)

    quant_x = linear_dequantize(new_quant_x,
                                scale,
                                zero_point,
                                inplace=False)

    return quant_x
