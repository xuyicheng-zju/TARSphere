# Long-CLIP source

- Upstream: [beichenzbc/Long-CLIP](https://github.com/beichenzbc/Long-CLIP).
- Paper: [Long-CLIP: Unlocking the Long-Text Capability of CLIP](https://arxiv.org/abs/2403.15378).
- Source revision: [`3966af9ae9331666309a22128468b734db4672a7`](https://github.com/beichenzbc/Long-CLIP/tree/3966af9ae9331666309a22128468b734db4672a7).
- License: [Apache-2.0](LICENSE), with original per-file notices retained.

Only the inference implementation and tokenizer data used by InternNav are bundled. Local adaptation imports `packaging` directly instead of the removed `pkg_resources` API, and package builds include the tokenizer data. Model weights are installed separately.
