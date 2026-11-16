"""PLS regression with a switchable X-scaling convention.

The environment variable EVOO_SCALING selects how the spectral variables are scaled inside
every PLS model of the study:
  unit (default) : centre each variable and scale it to unit variance (scikit-learn's default;
                   used for all main results)
  mean           : mean-centring only, the usual convention for spectra
Reading the setting at call time keeps worker processes consistent with the parent.
"""
import os

from sklearn.cross_decomposition import PLSRegression as _PLSRegression

ENV = "EVOO_SCALING"
MODES = ("unit", "mean")


def get_scaling():
    mode = os.environ.get(ENV, "unit")
    if mode not in MODES:
        raise ValueError(f"{ENV} must be one of {MODES}, got {mode!r}")
    return mode


def set_scaling(mode):
    if mode not in MODES:
        raise ValueError(f"scaling must be one of {MODES}, got {mode!r}")
    os.environ[ENV] = mode


def PLSRegression(n_components=2, **kwargs):
    kwargs.setdefault("scale", get_scaling() == "unit")
    return _PLSRegression(n_components=n_components, **kwargs)
