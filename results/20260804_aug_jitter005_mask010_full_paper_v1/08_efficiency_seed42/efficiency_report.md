# Model efficiency v1

- Trainable parameters: 12,149,928
- MACs: 1,461,515,072
- FLOPs (2 x MACs): 2,923,030,144
- Checkpoint size: 46.458 MiB
- State-dict size: 46.423 MiB

- Batch 1: median=7.9331 ms, P95=11.4631 ms, peak GPU memory=57.0 MiB
- Batch 64: median=15.6323 ms, P95=15.7712 ms, peak GPU memory=213.2 MiB

MAC scope: Conv1d and Linear only. Normalization, activations, pooling reductions, interpolation, sigmoid, and elementwise FiLM/GRN operations are excluded.
