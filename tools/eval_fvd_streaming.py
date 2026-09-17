#!/usr/bin/env python3
"""Public FVD CLI; implementation and tests live in tools/fvd_paired.py."""
try:
    from tools.fvd_paired import main
except ModuleNotFoundError:
    from fvd_paired import main

if __name__ == "__main__":
    raise SystemExit(main())
