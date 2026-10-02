"""

Minor utilities.

"""

import argparse
from typing import Generator
from pathlib import Path
from contextlib import contextmanager
from time import perf_counter

# top-level data directory
DATA_DIR = Path("/cephfs2/brs/pop2-prime/cc_512_no_dust_continue")
DEFAULT_LEDGER_DIR = Path.home() / "mphys" / "data" / "pop3_ledger.parquet"

def common_parser() -> argparse.ArgumentParser:
    """
    Arguments common to every parser. Wrap this function call in [] and pass to parents= arg on subparsers.
    """
    parser = argparse.ArgumentParser(add_help=False)  # need to set add_help=False for parent parsers
    parser.add_argument(
        "-l", 
        "--ledger", 
        type=Path, 
        default=DEFAULT_LEDGER_DIR, 
        help="Path to ledger parquet file."
    )
    parser.add_argument("-o", "--outdir", type=Path, default=Path("."), help="Path to output directory.")
    parser.add_argument("-s", "--stars", nargs="+", help="Stars to process (list labels).")

    return parser


@contextmanager
def timer(label: str) -> Generator[None, None, None]:
    """
    Context manager around perf_counter().
    """
    t0 = perf_counter()
    yield
    elapsed = perf_counter() - t0
    print(f"{label} completed in {elapsed:.1f}s.")