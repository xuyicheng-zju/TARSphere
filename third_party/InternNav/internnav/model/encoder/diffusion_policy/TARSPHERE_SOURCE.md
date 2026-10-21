# Diffusion Policy source

- Upstream: [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy), incorporated through InternNav.
- Paper: [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137).
- Source: a nested source snapshot; its upstream source commit is not recorded.
- License: [MIT](LICENSE), retaining the Columbia Artificial Intelligence and Robotics Lab copyright.

InternNav adapts imports to its `internnav.model.encoder.diffusion_policy` package. TARSphere retains that layout. The license text is reproduced verbatim from [the upstream license at revision 5ba07ac](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/LICENSE); this identifies the license copy, not a source revision pin.

Inherited implementations and notices include:

- [Planning with Diffusion](https://github.com/jannerm/diffuser) ([project and paper](https://diffusion-planning.github.io/)): the conditional U-Net is adapted through Diffusion Policy; its [MIT notice](model/diffusion/LICENSE_DIFFUSER) is reproduced from [the upstream license](https://github.com/jannerm/diffuser/blob/7ea422860cc0106e5ca5949d980f04b799d5462c/LICENSE).
- [Behavior Transformers](https://github.com/notmahi/bet) ([paper](https://arxiv.org/abs/2206.11251)): the nested `model/bet` baseline retains its [MIT notice](model/bet/LICENSE), reproduced from [the upstream license](https://github.com/notmahi/bet/blob/3d9d42352c0a0a6fb170369e95dde4ec3cfea951/LICENSE).
- [minGPT](https://github.com/karpathy/minGPT): the nested implementation retains its [MIT license](model/bet/libraries/mingpt/LICENSE).
- [Relay Policy Learning](https://github.com/google-research/relay-policy-learning) and [ROBEL](https://github.com/google-research/robel): inherited kitchen source retains per-file notices, an [Apache-2.0 model license](env/kitchen/relay_policy_learning/adept_models/LICENSE), and the separate [Franka model license](env/kitchen/relay_policy_learning/third_party/franka/LICENSE).
- [imagecodecs](https://github.com/cgohlke/imagecodecs) and [Pymunk](https://github.com/viblo/pymunk): adapted files retain their full [BSD notice](codecs/imagecodecs_numcodecs.py) and [MIT notice](env/pusht/pymunk_override.py), respectively.

These notices apply to their respective incorporated source; InternNav's license does not replace them. External libraries, model weights, and assets retain their own terms.
