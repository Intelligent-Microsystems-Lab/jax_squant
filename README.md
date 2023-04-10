# JAX/FLAX SQuant

JAX/FLAX implementation of the SQuant [paper](https://arxiv.org/pdf/2202.07471.pdf).

A 4-bit weight quantization on ResNet18 yields 70.32% on ImageNet where the quantization process takes 11.68s.

A 4-bit weight+activation quantization on ResNet18 yields 67.03+/-0.16 (over 10 runs) on ImageNet compared to 66.14 in the paper.
