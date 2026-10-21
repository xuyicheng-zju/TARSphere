"""Preload the default navigation dependencies before Isaac starts."""

from __future__ import annotations


def preload_open_frontier_clustering_dependencies() -> dict[str, str]:
    """Load and exercise HDBSCAN before Kit adds its vendored SciPy paths.

    Isaac Sim ships a partial SciPy tree for its own extensions. Once Kit is
    live that tree can be resolved together with the environment's sklearn
    and hdbscan, producing an ABI-incoherent mixed import. OpenFrontier's
    clustering is CPU-only, so load its declared dependency set before the
    SimulationApp exists and execute one tiny fit to resolve lazy imports.
    """

    try:
        import hdbscan
        import numpy as np
        import scipy
        import sklearn

        probe = np.column_stack(
            [
                np.linspace(0.0, 1.0, 32, dtype=np.float64),
                np.zeros((32, 4), dtype=np.float64),
            ]
        )
        labels = hdbscan.HDBSCAN(
            min_cluster_size=2,
            min_samples=1,
        ).fit_predict(probe)
        if np.asarray(labels).shape != (32,):
            raise RuntimeError("HDBSCAN smoke fit returned an invalid shape.")
    except Exception as exc:
        raise RuntimeError(
            "OpenFrontier requires one coherent numpy/scipy/sklearn/hdbscan "
            "environment before Isaac starts."
        ) from exc
    return {
        "hdbscan": str(getattr(hdbscan, "__file__", "")),
        "scipy": str(getattr(scipy, "__file__", "")),
        "sklearn": str(getattr(sklearn, "__file__", "")),
    }
