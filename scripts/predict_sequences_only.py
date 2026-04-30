"""Thin shim — delegates to ``enzymeexplorer.src.prediction.predict_sequences_only``.

The canonical CLI lives inside the package so it can be installed as the
``predict_sequences_only`` console-script entry point. This file is kept so
``python scripts/predict_sequences_only.py …`` invocations continue to work.
"""

from enzymeexplorer.src.prediction.predict_sequences_only import main


if __name__ == "__main__":
    main()
