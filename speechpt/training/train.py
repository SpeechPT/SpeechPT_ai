"""SageMaker container entrypoint router.

- If `--output-s3-uri` is present, run preprocessing job logic.
- Otherwise, run AE training logic.
"""
from __future__ import annotations

import sys


def main():
    if "--output-s3-uri" in sys.argv or "--output-dir" in sys.argv:
        from ae_preprocess_trainjob import main as preprocess_main

        preprocess_main()
        return

    from ae_probe_train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
