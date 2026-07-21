"""Root wrapper for `python benchmark_verify.py --results ...`."""

from adv_lab.eval.benchmark_verify import main


if __name__ == "__main__":
    raise SystemExit(main())