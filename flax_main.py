# Author: Clemens JS Schaefer
# Initially copied
# from https://github.com/google/flax/tree/main/examples/imagenet

import time
from absl import app
from absl import flags
from absl import logging
from clu import platform
import functools
from typing import Any, Callable, Sequence, Tuple
import numpy as np

import jax
from jax import lax
from jax import random
import jax.numpy as jnp

import flax
from flax import core
from flax import struct
from flax import jax_utils
from flax import linen as nn
from flax.training import checkpoints
from flax.training import common_utils
from flax.training import dynamic_scale as dynamic_scale_lib

import ml_collections
import optax
import tensorflow as tf
import tensorflow_datasets as tfds

import input_pipeline
from resnet_load_pretrained_weights import resnet_load_pretrained_weights as load_res18
from load_res50 import resnet_load_pretrained_weights as load_res50
from squant_flax import squant_fn, uniform_static


FLAGS = flags.FLAGS

flags.DEFINE_integer('rng', 69, 'Random seed.')
flags.DEFINE_string('model', 'ResNet50', 'Model type.')
flags.DEFINE_integer('batch_size', 256, 'Batch size.')
flags.DEFINE_string(
    'tfds_data_dir', '/afs/crc.nd.edu/user/c/cschaef6/tensorflow_datasets', '')
flags.DEFINE_string('dataset', 'imagenet2012:5.*.*', '')
flags.DEFINE_bool('cache', False, '')
flags.DEFINE_bool('half_precision', False, '')

flags.DEFINE_string('model_weights',
                    # 'unit_test_data/res18_w.pt',
                    'unit_test_data/res50.npy',
                    'Pretrained model weights location.')
flags.DEFINE_integer('wb', 4, 'Weight bits.')
flags.DEFINE_integer('ab', 4, 'Activation bits.')
flags.DEFINE_float('sigma', 12., 'Sigma for weight parameter init.')

NUM_CLASSES = 1000

ModuleDef = Any


class ResNetBlock(nn.Module):
  """ResNet block."""
  filters: int
  conv: ModuleDef
  norm: ModuleDef
  act: Callable
  strides: Tuple[int, int] = (1, 1)
  quant_fn: Callable = None

  @nn.compact
  def __call__(self, x, no_quant):
    residual = x

    # quant inpt
    x = self.quant_fn(sign=False)(x, no_quant=no_quant)

    if self.strides == (2, 2):
      y = self.conv(self.filters, (3, 3), self.strides,
                    padding=((1, 0), (1, 0)))(x)
      # block #3 Max absolute difference: 2.9087067e-05
    else:
      y = self.conv(self.filters, (3, 3), self.strides)(x)

    # block #1 Max absolute difference: 1.1444092e-05
    # block #2 Max absolute difference: 1.1444092e-05
    y = self.norm()(y)
    y = self.act(y)
    # block #1 Max absolute difference: 5.722046e-06

    # quant inpt
    y = self.quant_fn(sign=False)(y, no_quant=no_quant)

    y = self.conv(self.filters, (3, 3))(y)
    y = self.norm(scale_init=nn.initializers.zeros_init())(y)

    if residual.shape != y.shape:

      # quant inpt
      residual = self.quant_fn(sign=False)(residual, no_quant=no_quant)

      residual = self.conv(self.filters, (1, 1),
                           self.strides, name='conv_proj')(residual)
      residual = self.norm(name='norm_proj')(residual)
      # block #2 Max absolute difference: 6.0796738e-06

    # block #1 Max absolute difference: 9.059906e-06
    return self.act(residual + y)


class BottleneckResNetBlock(nn.Module):
  """Bottleneck ResNet block."""
  filters: int
  conv: ModuleDef
  norm: ModuleDef
  act: Callable
  strides: Tuple[int, int] = (1, 1)
  quant_fn: Callable = None

  @nn.compact
  def __call__(self, x, no_quant):
    residual = x

    # quant inpt
    x = self.quant_fn(sign=False)(x, no_quant=no_quant)
    y = self.conv(self.filters, (1, 1), strides=self.strides)(x)
    y = self.norm()(y)
    y = self.act(y)

    # quant inpt
    y = self.quant_fn(sign=False)(y, no_quant=no_quant)
    if self.strides == (2, 2):
      y = self.conv(self.filters, (3, 3), padding=((1, 1), (1, 1)))(y)
    else:
      y = self.conv(self.filters, (3, 3), self.strides)(y)
    y = self.norm()(y)
    y = self.act(y)

    # quant inpt
    y = self.quant_fn(sign=False)(y, no_quant=no_quant)
    y = self.conv(self.filters * 4, (1, 1))(y)
    y = self.norm(scale_init=nn.initializers.zeros_init())(y)

    if residual.shape != y.shape:
      residual = self.quant_fn(sign=False)(residual, no_quant=no_quant)
      residual = self.conv(self.filters * 4, (1, 1),
                           self.strides, name='conv_proj')(residual)
      residual = self.norm(name='norm_proj')(residual)

    return self.act(residual + y)


class ResNet(nn.Module):
  """ResNetV1."""
  stage_sizes: Sequence[int]
  block_cls: ModuleDef
  num_classes: int
  num_filters: int = 64
  dtype: Any = jnp.float32
  act: Callable = nn.relu
  conv: ModuleDef = nn.Conv
  quant_fn: Callable = None

  @nn.compact
  def __call__(self, x, no_quant: bool = False):
    train = False
    conv = functools.partial(self.conv, use_bias=False, dtype=self.dtype)
    norm = functools.partial(nn.BatchNorm,
                             use_running_average=not train,
                             momentum=0.9,
                             epsilon=1e-5,
                             dtype=self.dtype,
                             axis_name='batch')

    # quant inpt
    # x = self.quant_fn(sign=False)(x, no_quant=no_quant)

    _ = self.variable('quant_params', 'placeholder', jnp.zeros, (1,))

    x = conv(self.num_filters, (7, 7), (2, 2),
             padding=[(3, 3), (3, 3)],
             name='conv_init')(x)

    # Debugging
    # test = np.load('../inter.npy')
    # np.testing.assert_allclose(jnp.moveaxis(jnp.array(test),(0, 1, 2, 3),
    #      (0, 3, 1, 2)), x)

    x = norm(name='bn_init')(x)
    x = nn.relu(x)
    # Max absolute difference: 3.8146973e-06 ResNet18 (ones)
    # Max absolute difference: 9.536743e-07 ResNet50

    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding=((1, 0), (1, 0)),)
    # Max absolute difference: 3.8146973e-06 ResNet18
    # Max absolute difference: 9.536743e-07 ResNet50 (ones)

    for i, block_size in enumerate(self.stage_sizes):
      for j in range(block_size):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)
        x = self.block_cls(self.num_filters * 2 ** i,
                           strides=strides,
                           conv=conv,
                           norm=norm,
                           act=self.act,
                           quant_fn=self.quant_fn)(x, no_quant=no_quant)
        # block1 Max absolute difference: 2.1457672e-06 ResNet50 (ones)
        # block2 Max absolute difference: 6.1392784e-06 ResNet50 (ones)
        # block3 Max absolute difference: 7.688999e-06 ResNet50 (ones)

    x = jnp.mean(x, axis=(1, 2))

    # quant inpt - always 8 bit
    x = uniform_static(bits=8, percent=FLAGS.sigma, sign=False,)(x, no_quant=no_quant)
    # x = self.quant_fn(sign=False)(x, no_quant=no_quant)

    x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
    x = jnp.asarray(x, self.dtype)
    return x


ResNet18 = functools.partial(ResNet, stage_sizes=[2, 2, 2, 2],
                             block_cls=ResNetBlock)
ResNet34 = functools.partial(ResNet, stage_sizes=[3, 4, 6, 3],
                             block_cls=ResNetBlock)
ResNet50 = functools.partial(ResNet, stage_sizes=[3, 4, 6, 3],
                             block_cls=BottleneckResNetBlock)
ResNet101 = functools.partial(ResNet, stage_sizes=[3, 4, 23, 3],
                              block_cls=BottleneckResNetBlock)
ResNet152 = functools.partial(ResNet, stage_sizes=[3, 8, 36, 3],
                              block_cls=BottleneckResNetBlock)
ResNet200 = functools.partial(ResNet, stage_sizes=[3, 24, 36, 3],
                              block_cls=BottleneckResNetBlock)


def create_model(*, model_cls, quant_fn, half_precision, **kwargs):
  platform = jax.local_devices()[0].platform
  if half_precision:
    if platform == 'tpu':
      model_dtype = jnp.bfloat16
    else:
      model_dtype = jnp.float16
  else:
    model_dtype = jnp.float32
  return model_cls(num_classes=NUM_CLASSES, quant_fn=quant_fn,
                   dtype=model_dtype, **kwargs)


def initialized(key, image_size, model):
  input_shape = (1, image_size, image_size, 3)

  @jax.jit
  def init(*args):
    return model.init(*args)
  variables = init({'params': key}, jnp.ones(input_shape, model.dtype))
  return variables['params'], variables['quant_params'], \
      variables['batch_stats']


def cross_entropy_loss(logits, labels):
  one_hot_labels = common_utils.onehot(labels, num_classes=NUM_CLASSES)
  xentropy = optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels)
  return jnp.mean(xentropy)


def compute_metrics(logits, labels):
  loss = cross_entropy_loss(logits, labels)
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  metrics = {
      'loss': loss,
      'accuracy': accuracy,
  }
  metrics = lax.pmean(metrics, axis_name='batch')
  return metrics


def eval_step(state, batch, no_quant):
  variables = {'params': state.params, 'quant_params': state.quant_params,
               'batch_stats': state.batch_stats}
  logits = state.apply_fn(
      variables, batch['image'], mutable=False, no_quant=no_quant)
  return compute_metrics(logits, batch['label'])


def prepare_tf_data(xs):
  """Convert a input batch from tf Tensors to numpy arrays."""
  local_device_count = jax.local_device_count()

  def _prepare(x):
    # Use _numpy() for zero-copy conversion between TF and NumPy.
    x = x._numpy()  # pylint: disable=protected-access

    # reshape (host_batch_size, height, width, 3) to
    # (local_devices, device_batch_size, height, width, 3)
    return x.reshape((local_device_count, -1) + x.shape[1:])

  return jax.tree_util.tree_map(_prepare, xs)


def create_input_iter(dataset_builder, batch_size, image_size, dtype, train,
                      cache):
  ds = input_pipeline.create_split(
      dataset_builder, batch_size, image_size=image_size, dtype=dtype,
      train=train, cache=cache)
  it = map(prepare_tf_data, ds)
  it = jax_utils.prefetch_to_device(it, 2)
  return it


def restore_checkpoint(state, workdir):
  return checkpoints.restore_checkpoint(workdir, state)


def save_checkpoint(state, workdir):
  state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state))
  step = int(state.step)
  logging.info('Saving checkpoint step %d.', step)
  checkpoints.save_checkpoint_multiprocess(workdir, state, step, keep=3)


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


def create_train_state(rng, config: ml_collections.ConfigDict,
                       model, image_size):
  """Create initial training state."""

  params, quant_params, batch_stats = initialized(rng, image_size, model)
  state = TrainState.create(
      apply_fn=model.apply,
      params=params,
      quant_params=quant_params,
      tx=None,
      batch_stats=batch_stats,
  )
  return state


def train_and_evaluate(config: ml_collections.ConfigDict,
                       # workdir: str
                       ) -> TrainState:
  """Execute model training and evaluation loop.
  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.
  Returns:
    Final TrainState.
  """

  rng = random.PRNGKey(FLAGS.rng)

  image_size = 224

  if config.batch_size % jax.device_count() > 0:
    raise ValueError('Batch size must be divisible by the number of devices')
  local_batch_size = config.batch_size // jax.process_count()

  platform = jax.local_devices()[0].platform

  if config.half_precision:
    if platform == 'tpu':
      input_dtype = tf.bfloat16
    else:
      input_dtype = tf.float16
  else:
    input_dtype = tf.float32

  dataset_builder = tfds.builder(config.dataset, data_dir=config.tfds_data_dir)
  eval_iter = create_input_iter(
      dataset_builder, local_batch_size, image_size, input_dtype, train=False,
      cache=config.cache)

  num_validation_examples = dataset_builder.info.splits[
      'validation'].num_examples
  steps_per_eval = num_validation_examples // config.batch_size

  quant_fn = functools.partial(
      uniform_static, bits=FLAGS.ab, percent=FLAGS.sigma, sign=False,)

  model_cls = globals()[config.model]
  model = create_model(
      model_cls=model_cls, quant_fn=quant_fn,
      half_precision=config.half_precision)

  rng, rng_key = jax.random.split(rng, 2)
  state = create_train_state(
      rng_key, config, model, image_size)
  if FLAGS.model == 'ResNet18':
    state = load_res18(state, FLAGS.model_weights)
  elif FLAGS.model == 'ResNet50':
    state = load_res50(state, FLAGS.model_weights)
  else:
    raise Exception('Loading model method not implemented for: ' + FLAGS.model)
  logging.info('Model loaded successfully.')
  state = jax_utils.replicate(state)
  p_eval_step = jax.pmap(functools.partial(
      eval_step, no_quant=True), axis_name='batch')

  eval_metrics = []
  for _ in range(steps_per_eval):
    eval_batch = next(eval_iter)

    metrics = p_eval_step(state, eval_batch)
    eval_metrics.append(metrics)
  eval_metrics = common_utils.get_metrics(eval_metrics)
  summary = jax.tree_util.tree_map(lambda x: x.mean(), eval_metrics)
  logging.info('Pre-quant, loss: %.4f, accuracy: %.2f',
               summary['loss'], summary['accuracy'] * 100)

  # 100-26.94 = 73.06
  # I0406 15:52:14.762888 22563244531456 flax_main.py:374]
  # Pre-quant, loss: 1.2219, accuracy: 73.07

  state = jax_utils.unreplicate(state)

  jsquant_fn = jax.jit(functools.partial(
      squant_fn, bit=FLAGS.wb, is_perchannel=True, squant_k=True,
      squant_c=True, scale_off=False))

  def quant_single(path, x):
    if ('kernel' in path):  # bias quant maybe not done?
      if ('BatchNorm' not in '.'.join(path)) and ('bn_init' not in path) and \
              ('stem_bn' not in path) and ('head_bn' not in path) and \
              ('norm_proj' not in path):
        logging.info("QUANT %d bit: %s shape %s num_params: %s" %
                     (4, '.'.join(path), str(x.shape), str(np.prod(x.shape))))
        start = time.perf_counter()
        x = jsquant_fn(x)
        elapsed = (time.perf_counter() - start)
        logging.info("Quantzation time: %f ms" % (elapsed * 1000))

    return x

  qweights = flax.traverse_util.path_aware_map(quant_single, state.params)

  state = TrainState.create(
      apply_fn=state.apply_fn,
      params=qweights,
      quant_params=state.quant_params,
      tx=state.tx,
      batch_stats=state.batch_stats,
  )

  # calibrate activation quant.
  rng, rng_key1, rng_key2 = jax.random.split(rng, 3)
  init_noise = jax.random.uniform(rng_key1, (FLAGS.batch_size, 224, 224, 3))

  _, new_state = state.apply_fn({'params': state.params,
                                 'quant_params': state.quant_params,
                                 'batch_stats': state.batch_stats,
                                 },
                                init_noise,
                                mutable=['quant_params',],
                                no_quant=False,
                                )
  state = TrainState.create(
      apply_fn=state.apply_fn,
      params=state.params,
      quant_params=new_state['quant_params'],
      tx=None,
      batch_stats=state.batch_stats,
  )

  state = jax_utils.replicate(state)
  p_eval_step = jax.pmap(functools.partial(
      eval_step, no_quant=False), axis_name='batch')

  # eval quant
  eval_metrics = []
  for _ in range(steps_per_eval):
    eval_batch = next(eval_iter)

    metrics = p_eval_step(state, eval_batch)
    eval_metrics.append(metrics)
  eval_metrics = common_utils.get_metrics(eval_metrics)
  summary = jax.tree_util.tree_map(lambda x: x.mean(), eval_metrics)
  logging.info('Post-quant, loss: %.4f, accuracy: %.2f',
               summary['loss'], summary['accuracy'] * 100)

  # Wait until computations are done before exiting
  jax.random.normal(jax.random.PRNGKey(FLAGS.rng), ()).block_until_ready()

  return state


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], 'GPU')

  logging.info('JAX process: %d / %d',
               jax.process_index(), jax.process_count())
  logging.info('JAX local devices: %r', jax.local_devices())

  # Add a note so that we can tell which task is which JAX host.
  # (Depending on the platform task 0 is not guaranteed to be host 0)
  platform.work_unit().set_task_status(f'process_index: {jax.process_index()},'
                                       f'process_count: {jax.process_count()}')

  train_and_evaluate(FLAGS)


if __name__ == '__main__':
  app.run(main)
