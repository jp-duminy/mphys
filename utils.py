"""

Minor utilities.

"""

from typing import Generator
from pathlib import Path
from contextlib import contextmanager
from time import perf_counter

# top-level data directory
DATA_DIR = Path("/cephfs2/brs/pop2-prime/cc_512_no_dust_continue")

@contextmanager
def timer(label: str) -> Generator[None, None, None]:
    """
    Context manager around perf_counter().
    """
    t0 = perf_counter()
    yield
    elapsed = perf_counter() - t0
    print(f"{label} completed in {elapsed:.1f}s.")