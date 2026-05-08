"""Statistical / exploratory audit + train-test-split tooling.

The audit (``eda.audit``) depends on the ``[eda]`` extras (matplotlib, seaborn,
statsmodels, scikit-learn). The split helper (``eda.split``) is dependency-light
and runnable from any environment that has pandas. We therefore avoid eagerly
importing the audit at package-import time so ``eda.split`` stays usable even
when the heavy plotting deps aren't installed (e.g. in default CI).

Use :func:`run_audit` directly via ``from eda.audit import run_audit``.
"""

__all__ = ["audit", "split"]
