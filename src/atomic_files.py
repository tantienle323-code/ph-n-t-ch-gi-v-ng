
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def atomic_output_path(destination):
    """Yield a unique same-directory temporary path and atomically publish it."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)

    try:
        yield temporary_path
        os.replace(temporary_path, destination)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
