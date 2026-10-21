# Bundled Long-CLIP inference code

Source: [beichenzbc/Long-CLIP](https://github.com/beichenzbc/Long-CLIP), revision `3966af9ae9331666309a22128468b734db4672a7`.

Only the model implementation and tokenizer data required by the bundled InternNav encoders are included. Model weights are installed separately. The source retains its [Apache-2.0 license](LICENSE) and per-file notices.

Local adaptation: import the supported `packaging` module directly instead of the removed `pkg_resources` API. Tokenizer data are included in package builds.
