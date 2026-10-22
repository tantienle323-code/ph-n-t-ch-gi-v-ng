import os
import platform
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def market_data_refresh_lock(
    lock_path,
    timeout_seconds=300,
):
    """Đảm bảo chỉ một tiến trình cập nhật dữ liệu tại một thời điểm."""
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(lock_path, "a+b") as lock_file:
        if platform.system() == "Windows":
            import msvcrt

            lock_file.seek(0, os.SEEK_END)

            if lock_file.tell() == 0:
                lock_file.write(b"\0")
                lock_file.flush()

            deadline = time.monotonic() + timeout_seconds

            while True:
                lock_file.seek(0)

                try:
                    msvcrt.locking(
                        lock_file.fileno(),
                        msvcrt.LK_NBLCK,
                        1,
                    )
                    break

                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "Một tiến trình khác đang cập nhật "
                            "dữ liệu quá lâu."
                        ) from None

                    time.sleep(0.1)

            try:
                yield

            finally:
                lock_file.seek(0)
                msvcrt.locking(
                    lock_file.fileno(),
                    msvcrt.LK_UNLCK,
                    1,
                )

        else:
            import fcntl

            deadline = time.monotonic() + timeout_seconds

            while True:
                try:
                    fcntl.flock(
                        lock_file.fileno(),
                        fcntl.LOCK_EX | fcntl.LOCK_NB,
                    )
                    break

                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "Một tiến trình khác đang cập nhật "
                            "dữ liệu quá lâu."
                        ) from None

                    time.sleep(0.1)

            try:
                yield

            finally:
                fcntl.flock(
                    lock_file.fileno(),
                    fcntl.LOCK_UN,
                )