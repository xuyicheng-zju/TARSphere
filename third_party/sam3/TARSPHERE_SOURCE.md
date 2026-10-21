# SAM3 source

- Upstream: [facebookresearch/sam3](https://github.com/facebookresearch/sam3).
- Paper: [SAM 3: Segment Anything with Concepts](https://arxiv.org/abs/2511.16719).
- Source: a bundled source snapshot; its upstream commit is not recorded.
- License: [SAM License](LICENSE), including its original copyright and use terms.

The local [checkpoint-loading patch](../patches/0001-sam3-load-checkpoint-on-target-device.patch) allocates and loads the image model on its configured device and validates consumed detector parameters. For SAM3.1 image inference it excludes the explicitly named video-only neck parameters and allows only the unused fourth FPN level to lack weights. It is already applied. TARSphere serves image segmentation through its own service adapter. Checkpoints are installed separately. Original copyright and dependency notices are retained.
