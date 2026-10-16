import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.atomic_files import atomic_output_path
from src.market_refresh import market_data_refresh_lock


class TestAtomicOutputPath(unittest.TestCase):
    def test_concurrent_writers_get_unique_paths_and_publish_atomically(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / 'market.csv'
            destination.write_text('old', encoding='utf-8')

            with atomic_output_path(destination) as first_path:
                with atomic_output_path(destination) as second_path:
                    self.assertNotEqual(first_path, second_path)
                    Path(first_path).write_text('first', encoding='utf-8')
                    Path(second_path).write_text('second', encoding='utf-8')

            self.assertEqual(destination.read_text(encoding='utf-8'), 'first')
            self.assertEqual(
                sorted(path.name for path in Path(temp_dir).iterdir()),
                ['market.csv'],
            )

    def test_failed_write_keeps_previous_file_and_removes_temporary_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / 'model.pkl'
            destination.write_bytes(b'old-model')

            with self.assertRaisesRegex(RuntimeError, 'write failed'):
                with atomic_output_path(destination) as temporary_path:
                    Path(temporary_path).write_bytes(b'incomplete')
                    raise RuntimeError('write failed')

            self.assertEqual(destination.read_bytes(), b'old-model')
            self.assertEqual(list(Path(temp_dir).iterdir()), [destination])

    def test_market_refresh_lock_can_be_reacquired_after_release(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir) / 'data' / '.market_refresh.lock'

            with market_data_refresh_lock(lock_path, timeout_seconds=1):
                self.assertTrue(lock_path.exists())
            with market_data_refresh_lock(lock_path, timeout_seconds=1):
                self.assertTrue(lock_path.exists())


if __name__ == '__main__':
    unittest.main()
