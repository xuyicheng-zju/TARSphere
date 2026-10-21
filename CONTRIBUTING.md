# Contributing

Use a focused branch and keep changes small enough to review. Include a minimal reproduction for bug reports and explain the resulting behavior in pull requests.

## Development

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m ruff check tarsphere scripts tests examples
python -m ruff format --check tarsphere scripts tests examples
python -m pytest -q
python -m build
```

Simulator-dependent tests require the corresponding SDK or assets. Keep hardware and model API calls out of the default unit suite.

## Benchmark changes

Update specifications, indexes, and documentation together. State the effect on task semantics, model inputs, scoring, and result comparability. Preserve unsuccessful and interrupted runs as such.

## Contributions and licenses

Original contributions use Apache-2.0. Preserve upstream copyright and license notices, and check the terms before adding third-party material. Keep credentials, local host configuration, generated output, and model weights out of commits.
