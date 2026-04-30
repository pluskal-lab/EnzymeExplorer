"""Thin shim — delegates to ``enzymeexplorer.src.prediction.predict_with_structures``.

The canonical CLI lives inside the package so it can be installed as the
``predict_with_structures`` console-script entry point. This file is kept so
``python scripts/predict_with_structures.py …`` invocations continue to work.
"""

from enzymeexplorer.src.prediction.predict_with_structures import main


if __name__ == "__main__":
    main()
