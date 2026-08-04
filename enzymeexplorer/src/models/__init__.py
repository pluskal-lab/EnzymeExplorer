"""Predictive models, exposed via PEP 562 lazy attribute resolution.

Eagerly re-exporting every model class at package import time made
``import enzymeexplorer.src.models`` (transitively reached by
``import enzymeexplorer.src``) drag in a long tail of heavy or
optional dependencies that production prediction does not need:

  * ``CLEAN`` baseline -> ``gdown`` for model fetches, ProFun
  * ``DomainsXgb`` -> ``xgboost``
  * ``Foldseek`` / ``Blastp`` / ``HMM`` / ``PfamSUPFAM`` -> wrappers
    around external command-line tools that we do not ship in the
    production env

Production prediction unpickles two fold-checkpoint bundles
(:data:`pipeline.DEFAULT_PLM_DOMAINS_BUNDLE`,
:data:`pipeline.DEFAULT_PLM_ONLY_BUNDLE`). Pickle resolves classes by
their *defining* module path
(``enzymeexplorer.src.models.plm_randomforest:PlmRandomForest`` etc.),
so only the specific submodule for the pickled class needs to be
importable — never the package's re-exports.

The two non-prediction callers that DO use the re-exports
(``experiments_orchestration.experiment_runner``) do so via
``getattr(models, "<ModelName>")`` (single-class dispatch from CLI
config) and ``inspect.getmembers(models, inspect.isclass)`` (error
message listing available models). Both still work after the lazy
conversion: ``__getattr__`` triggers on attribute access, ``__dir__``
advertises the known names for ``inspect.getmembers``.

Adding a new model class: append a single row to ``_LAZY_IMPORTS``.
"""

from __future__ import annotations

import importlib
from typing import Any

# Mapping ``<class name>`` -> ``<submodule path relative to this package>``.
# The submodule is imported on first attribute access and the resolved
# class is cached on the module globals so subsequent accesses skip the
# importlib round-trip.
_LAZY_IMPORTS: dict[str, str] = {
    # Domain-feature models (require structural features).
    "DomainsRandomForest":           ".domain_comparisons_randomforest",
    "DomainsXgb":                    ".domain_comparisons_xgb",
    "PlmDomainsRandomForest":        ".plm_domain_comparison_randomforest",
    "PlmDomainsMLP":                 ".plm_domains_mlp",
    "PlmDomainsLogisticRegression":  ".plm_domains_logistic_regression",
    # PLM-only models.
    "PlmRandomForest":               ".plm_randomforest",
    # External-tool baselines (each pulls in its own subprocess wrapper).
    "Blastp":                        ".baselines",
    "Foldseek":                      ".baselines",
    "HMM":                           ".baselines",
    "PfamSUPFAM":                    ".baselines",
    "CLEAN":                         ".baselines",
}


def __getattr__(name: str) -> Any:
    """PEP 562 hook — import the relevant submodule on first access."""
    submodule_path = _LAZY_IMPORTS.get(name)
    if submodule_path is None:
        raise AttributeError(
            f"module 'enzymeexplorer.src.models' has no attribute {name!r}"
        )
    submodule = importlib.import_module(submodule_path, package=__name__)
    cls = getattr(submodule, name)
    # Cache on the package's globals so future lookups bypass __getattr__
    # and so ``inspect.getmembers`` sees a real attribute, not a descriptor.
    globals()[name] = cls
    return cls


def __dir__() -> list[str]:
    """Advertise the lazy names for ``inspect.getmembers`` /
    interactive completion. Does NOT trigger the imports — only
    listing the keys."""
    return sorted(set(_LAZY_IMPORTS.keys()) | set(globals().keys()))
