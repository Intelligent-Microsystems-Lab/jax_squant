# JAX/FLAX SQuant

JAX/FLAX implementation of the SQuant [paper](https://arxiv.org/pdf/2202.07471.pdf).

A 4-bit weight+activation quantization on ResNet18 yields 68.28% on ImageNet compared to 66.14% in the paper and 68.07+/-0.31 (over 10 runs) from the original implementation.

A 4-bit weight+activation quantization on ResNet50 yields 71.44% on ImageNet compared to 70.80% in the paper.


Note: Inputs to first layer are not quantized and inputs to last layer (FC) are always at 8 bit.
