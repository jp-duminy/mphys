"""

Minor utilities.

"""

from typing import Generator
from contextlib import contextmanager
from time import perf_counter

@contextmanager
def timer(label: str) -> Generator[None, None, None]:
    """
    Context manager around perf_counter().
    """
    t0 = perf_counter()
    yield
    elapsed = perf_counter() - t0
    print(f"{label} completed in {elapsed:.1f}s.")